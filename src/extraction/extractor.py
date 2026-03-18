"""LLM事实抽取（多轮fallback检索版）

核心流程：
  检索相关chunks → 拼装prompt → 调LLM → Pydantic校验 → 检测缺失字段 → fallback

多轮fallback策略（extract_field函数内部）：

  Round 1 — 子字段组分别检索
    将原来的一个大杂烩query拆成4个聚焦的子组（parties/pricing/valuation/commitment），
    分别做向量检索，合并去重后送LLM做全字段抽取。
    大部分字段在这一轮就能提取到。

  Round 2 — HyDE（Hypothetical Document Embeddings）
    对Round 1中仍为空的字段，用LLM生成一段"假设性的报告原文"，
    再用这段文本的embedding去检索。这比直接用"估值方法"4个字检索效果好很多，
    因为假设文本包含了丰富的上下文关键词。
    检索到新的chunks后，做定向LLM抽取（只提取缺失字段）。

  Round 3 — 章节路由
    对Round 2后仍为空的字段，根据预定义的section关键词过滤出相关章节的chunks，
    在这个子集内做向量检索。
    例如 valuation_method → 只在section包含"评估/估值/定价"的chunks中搜索。
    这是最精确的定位手段，但依赖于chunker正确识别了章节标题。

调用关系：
  pipeline.py → extract_facts() → extract_field() （内含3轮fallback）
"""

from pydantic import ValidationError

from src.extraction.prompts import (
    FIELD_CONFIGS,
    FIELD_DESCRIPTIONS,
    FIELD_SUBGROUPS,
    HYDE_SYSTEM,
    HYDE_USER,
    SubFieldGroup,
    TARGETED_EXTRACT_SYSTEM,
    TARGETED_EXTRACT_USER,
)
from src.indexing.search import search_chunks, search_chunks_by_section
from src.indexing.store import VectorStore
from src.infra.config import settings
from src.infra.embedder import embed_text
from src.infra.llm_client import chat_json, chat_raw
from src.infra.logger import get_logger
from src.infra.models import (
    AcquisitionPurpose,
    Chunk,
    DealSummary,
    ExtractionResult,
    ExtractionStatus,
    FactRecord,
)

logger = get_logger(__name__)

# 字段名 → Pydantic模型
FIELD_MODELS = {
    "deal_summary": DealSummary,
    "acquisition_purpose": AcquisitionPurpose,
}


# =====================================================================
# 多轮fallback helper函数
#
# 这些函数被extract_field()内部调用，不对外暴露。
# 按职责分为4类：
#   1. 缺失检测：_get_missing_fields, _subgroups_for_missing
#   2. 结果合并：_merge_data, _deduplicate_chunks
#   3. 格式化：_format_chunks_text, _format_source_ids
#   4. 三轮检索：_round1_retrieve, _round2_hyde, _round3_section, _targeted_extract
# =====================================================================


def _get_missing_fields(data: dict, field_name: str) -> list[str]:
    """检测LLM返回结果中哪些主字段为空

    只检查target_fields中定义的主字段（如acquirer, valuation_method），
    不检查_quote后缀字段（_quote跟随主字段，主字段为空则_quote也无意义）。

    Args:
        data: LLM返回并经Pydantic校验后的字段字典
        field_name: 字段组名（deal_summary / acquisition_purpose）

    Returns:
        值为空字符串的主字段名列表，如 ["valuation_method", "performance_commitment"]
    """
    # 从所有SubFieldGroup中收集target_fields作为检查对象
    subgroups = FIELD_SUBGROUPS.get(field_name, [])
    all_target_fields = set()
    for sg in subgroups:
        all_target_fields.update(sg.target_fields)

    missing = []
    for f in all_target_fields:
        if not data.get(f, ""):
            missing.append(f)
    return missing


def _subgroups_for_missing(
    missing_fields: list[str], field_name: str
) -> list[SubFieldGroup]:
    """根据缺失字段找到需要重新检索的SubFieldGroup

    例如：missing=["valuation_method"] → 返回 valuation 组
    例如：missing=["valuation_method", "performance_commitment"] → 返回 valuation + commitment 两组

    通过集合交集判断：如果某个subgroup的target_fields与missing_fields有交集，就需要该组。
    """
    subgroups = FIELD_SUBGROUPS.get(field_name, [])
    result = []
    missing_set = set(missing_fields)
    for sg in subgroups:
        if missing_set & set(sg.target_fields):
            result.append(sg)
    return result


def _merge_data(
    existing: dict, new_data: dict, missing_fields: list[str]
) -> dict:
    """合并Round 2/3的定向抽取结果到已有数据中

    核心原则：只填空字段，绝不覆盖已有值。
    这确保了Round 1的高置信度结果不会被后续轮次的低置信度结果覆盖。

    同时处理_quote字段：如果主字段被填充，对应的_quote也一起填入。

    Args:
        existing: Round 1（或上一轮合并后）的数据字典
        new_data: 本轮定向抽取的结果字典
        missing_fields: 允许被填充的主字段名列表（安全边界，防止意外覆盖）

    Returns:
        合并后的新字典（不修改existing原对象）
    """
    merged = dict(existing)
    for field in missing_fields:
        new_val = new_data.get(field, "")
        # 只在"新值非空"且"已有值为空"时填充
        if new_val and not merged.get(field, ""):
            merged[field] = new_val
            # 同时填充对应的_quote字段（如target_valuation → target_valuation_quote）
            quote_key = f"{field}_quote"
            if quote_key in new_data and not merged.get(quote_key, ""):
                merged[quote_key] = new_data[quote_key]
    return merged


def _deduplicate_chunks(
    chunk_lists: list[list[tuple[Chunk, float]]],
) -> list[tuple[Chunk, float]]:
    """合并多组检索结果，按(doc_id, chunk_id)去重，保留最高相似度分数

    Round 1中4个子组分别检索会返回4组结果，可能有重叠的chunks。
    此函数去重后按分数降序排列，供后续拼装prompt使用。

    Args:
        chunk_lists: 多组检索结果，每组是 [(Chunk, score), ...] 列表

    Returns:
        去重后的结果列表，按score降序排列
    """
    best: dict[int, tuple[Chunk, float]] = {}
    for results in chunk_lists:
        for chunk, score in results:
            # 用(doc_id, chunk_id)的hash作为唯一标识（跨文档安全）
            composite_key = hash((chunk.doc_id, chunk.chunk_id))
            if composite_key not in best or score > best[composite_key][1]:
                best[composite_key] = (chunk, score)

    sorted_results = sorted(best.values(), key=lambda x: x[1], reverse=True)
    return sorted_results


def _format_chunks_text(results: list[tuple[Chunk, float]]) -> str:
    """将检索结果格式化为LLM prompt中的文本片段，每个chunk带页码和章节标注"""
    return "\n\n---\n\n".join(
        f"[页码:{c.page}, 章节:{c.section}]\n{c.text}" for c, _ in results
    )


def _format_source_ids(results: list[tuple[Chunk, float]]) -> list[str]:
    """提取来源标识列表，格式："doc_id:chunkN:pN"，用于ExtractionResult.source_chunks"""
    return [f"{c.doc_id}:chunk{c.chunk_id}:p{c.page}" for c, _ in results]


def _round1_retrieve(
    store: VectorStore, doc_id: str, field_name: str, top_k: int | None = None
) -> list[tuple[Chunk, float]]:
    """Round 1检索：按子字段组分别检索 + 合并去重

    与原来的单query检索对比：
    - 原来：一个query = "交易方案 交易金额 发行价格 支付方式 标的估值..."（9个概念混在一起）
    - 现在：4个query分别检索 parties/pricing/valuation/commitment，每个只包含2-3个相关概念

    每个子组检索 per_group_k = top_k // 子组数 个chunks（至少4个），
    合并去重后截断到total top_k，确保总chunks数不超过LLM上下文限制。
    """
    subgroups = FIELD_SUBGROUPS.get(field_name, [])
    if not subgroups:
        # 没有子组定义时fallback到原始单query（向后兼容）
        config = FIELD_CONFIGS[field_name]
        return search_chunks(store, config["query"], top_k=top_k, doc_id=doc_id)

    # 每个子组分配的检索数量，至少4个保证覆盖
    per_group_k = max(4, (top_k or settings.top_k) // len(subgroups))
    chunk_lists = []
    for sg in subgroups:
        results = search_chunks(store, sg.query, top_k=per_group_k, doc_id=doc_id)
        chunk_lists.append(results)

    deduped = _deduplicate_chunks(chunk_lists)
    # 截断到最终top_k，避免送入LLM的chunks过多
    final_k = top_k or settings.top_k
    return deduped[:final_k]


def _round2_hyde(
    store: VectorStore, doc_id: str, subgroup: SubFieldGroup, top_k: int | None = None
) -> list[tuple[Chunk, float]]:
    """Round 2检索：HyDE（Hypothetical Document Embeddings）

    流程：
    1. 用chat_raw()让LLM根据hyde_query生成一段假设性报告原文（约100-200字中文）
    2. 将这段假设文本做embed，得到一个"语义丰富"的查询向量
    3. 用这个向量在FAISS中检索

    为什么比直接query效果好：
    - 直接query "估值方法" → embedding只包含4个字的语义
    - HyDE生成 "本次交易采用收益法和资产基础法对标的公司100%股权进行评估，
      最终以收益法评估结果作为评估结论" → embedding包含丰富的上下文关键词，
      与真实报告文本的embedding更接近

    注意：这里调用了chat_raw()（不要求JSON格式），而非chat_json()
    """
    # Step 1: LLM生成假设文本
    hyde_text = chat_raw(
        HYDE_SYSTEM,
        HYDE_USER.format(hyde_query=subgroup.hyde_query),
    )
    if not hyde_text:
        logger.warning(f"HyDE生成失败: {subgroup.name}")
        return []

    logger.debug(f"HyDE生成文本({subgroup.name}): {hyde_text[:80]}...")

    # Step 2: 用假设文本的embedding做检索（注意这里直接调store.search而非search_chunks，
    # 因为我们已经有了文本，需要自己embed后传入向量）
    hyde_embedding = embed_text(hyde_text)
    results = store.search(hyde_embedding, top_k=top_k, doc_id=doc_id)
    return results


def _round3_section(
    store: VectorStore, doc_id: str, subgroup: SubFieldGroup, top_k: int | None = None
) -> list[tuple[Chunk, float]]:
    """Round 3检索：章节路由

    思路：并购报告的章节结构高度标准化，虽然不同报告的命名有差异
    （"评估与定价" vs "标的资产估值情况"），但关键词是稳定的。
    用section_keywords做子串匹配，把候选chunks缩小到目标章节内，
    再在这个子集中做向量检索。

    例如 valuation 组：
    - section_keywords = ["评估", "估值", "定价"]
    - 能匹配到 "第六节 交易标的的评估与定价" / "第六章 标的资产评估及定价情况"
    - 只在这些章节的chunks中搜索，避免被其他章节的噪音干扰
    """
    results = search_chunks_by_section(
        store,
        subgroup.query,
        subgroup.section_keywords,
        top_k=top_k,
        doc_id=doc_id,
    )
    return results


def _targeted_extract(
    chunks: list[tuple[Chunk, float]],
    field_name: str,
    missing_fields: list[str],
    doc_id: str,
) -> dict | None:
    """定向LLM抽取：只提取指定的缺失字段（Round 2/3专用）

    与Round 1全字段抽取的区别：
    - Round 1: 用完整的DEAL_SUMMARY_SYSTEM prompt，让LLM提取全部14个字段
    - 定向抽取: 用TARGETED_EXTRACT_SYSTEM prompt，只让LLM提取2-3个缺失字段

    好处：
    1. 省token——prompt中只描述需要的字段
    2. LLM更聚焦——不会在已有字段上浪费注意力
    3. 结果更可控——返回的JSON只包含目标字段

    Args:
        chunks: 检索到的chunks（来自HyDE或章节路由）
        field_name: 字段组名（用于日志）
        missing_fields: 需要抽取的主字段名列表，如 ["valuation_method"]
        doc_id: 文档ID（用于日志）

    Returns:
        抽取结果字典（如 {"valuation_method": "收益法", "valuation_method_quote": "..."}），
        失败返回None
    """
    if not chunks:
        return None

    # 根据缺失字段动态构建prompt中的字段描述
    # 例如 missing=["valuation_method"] → 只在prompt中描述 valuation_method 这个字段
    descriptions = []
    for f in missing_fields:
        desc = FIELD_DESCRIPTIONS.get(f, f"   - {f}")
        descriptions.append(f"   - {desc}")
    field_desc_text = "\n".join(descriptions)

    chunks_text = _format_chunks_text(chunks)
    system_prompt = TARGETED_EXTRACT_SYSTEM.format(field_descriptions=field_desc_text)
    user_prompt = TARGETED_EXTRACT_USER.format(chunks_text=chunks_text)

    logger.info(
        f"[{doc_id}] {field_name}: 定向抽取 {missing_fields}，使用 {len(chunks)} chunks"
    )

    result = chat_json(system_prompt, user_prompt)
    return result


def extract_field(
    store: VectorStore,
    doc_id: str,
    field_name: str,
    top_k: int | None = None,
) -> ExtractionResult:
    """抽取单个事实字段（支持多轮fallback检索）

    这是多轮fallback系统的入口函数。签名与原版完全一致，保持向后兼容。
    内部根据settings.enable_multi_round决定走单轮还是多轮流程。

    多轮流程概览（以deal_summary为例）：

      Round 1: 4个子组query分别检索 → 合并去重8个chunks → LLM全字段抽取
               → 得到 {"acquirer": "深赤湾", ..., "valuation_method": ""}
               → 检测到 valuation_method 为空

      Round 2: valuation子组 → HyDE生成假设文本 → embed → 检索 → 定向抽取valuation_method
               → 得到 {"valuation_method": "收益法"} → 合并到data中

      Round 3: （如果Round 2仍未填充）→ 章节路由 → section含"评估"的chunks → 定向抽取

    LLM调用次数分析（deal_summary 4个子组）：
      最佳情况：Round 1全部填充 → 1次chat_json
      典型情况：Round 1缺1-2个字段 → 1次chat_json + 1次chat_raw(HyDE) + 1次chat_json(定向)
      最差情况：全部fallback → 1 + 4*(1+1) + 4*1 = 13次（极端情况，实际不会发生）

    Args:
        store: 向量存储（包含FAISS索引和chunk元数据）
        doc_id: 文档标识（用于过滤检索结果，确保只检索本文档的chunks）
        field_name: 字段组名（"deal_summary" 或 "acquisition_purpose"）
        top_k: 检索返回的chunks数量，默认从settings.top_k读取

    Returns:
        ExtractionResult，其中：
        - status=SUCCESS: 所有字段都已填充
        - status=PARTIAL: 三轮后仍有缺失字段（新增状态）
        - status=FAILED: Round 1就失败了（无chunks或LLM调用失败）
    """
    config = FIELD_CONFIGS[field_name]
    model_cls = FIELD_MODELS[field_name]

    # ================================================================
    # Round 1: 子字段组分别检索 → 全字段LLM抽取
    # ================================================================
    if settings.enable_multi_round and field_name in FIELD_SUBGROUPS:
        # 多轮模式：拆分为多个聚焦query分别检索，覆盖更广
        results = _round1_retrieve(store, doc_id, field_name, top_k=top_k)
    else:
        # 单轮模式（或未配置子组）：使用原始的单一monolithic query
        results = search_chunks(store, config["query"], top_k=top_k, doc_id=doc_id)

    if not results:
        logger.warning(f"[{doc_id}] {field_name}: 未检索到相关chunks")
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.FAILED,
            error="未检索到相关chunks",
        )

    # 用原始完整prompt做LLM全字段抽取（保持与原版行为一致）
    chunks_text = _format_chunks_text(results)
    all_source_ids = _format_source_ids(results)
    user_prompt = config["user_prompt_template"].format(chunks_text=chunks_text)

    logger.info(f"[{doc_id}] {field_name}: Round 1 使用 {len(results)} chunks 调用LLM")

    raw_result = chat_json(config["system_prompt"], user_prompt)

    if raw_result is None:
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.FAILED,
            source_chunks=all_source_ids,
            error="LLM调用失败",
        )

    # Pydantic校验Round 1结果
    try:
        validated = model_cls.model_validate(raw_result)
        data = validated.model_dump()
    except ValidationError as e:
        logger.warning(f"[{doc_id}] {field_name}: Round 1 Pydantic校验失败: {e}")
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.FAILED,
            raw_response=str(raw_result),
            source_chunks=all_source_ids,
            error=f"校验失败: {e}",
        )

    # 如果未启用多轮fallback，直接返回Round 1结果（向后兼容）
    if not settings.enable_multi_round or field_name not in FIELD_SUBGROUPS:
        logger.info(f"[{doc_id}] {field_name}: 抽取成功")
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.SUCCESS,
            data=data,
            raw_response=str(raw_result),
            source_chunks=all_source_ids,
        )

    # 检测Round 1后哪些字段仍为空
    missing = _get_missing_fields(data, field_name)
    if not missing:
        logger.info(f"[{doc_id}] {field_name}: Round 1 全部字段已填充")
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.SUCCESS,
            data=data,
            raw_response=str(raw_result),
            source_chunks=all_source_ids,
        )

    logger.info(f"[{doc_id}] {field_name}: Round 1 缺失字段 {missing}，启动Round 2 HyDE")

    # ================================================================
    # Round 2: HyDE补全缺失字段
    #
    # 只对缺失字段所属的SubFieldGroup触发HyDE，不重复处理已填充的组。
    # 例如 parties组的字段都填好了，只有valuation组缺失 → 只对valuation组做HyDE。
    # 每个组：chat_raw(HyDE生成) + chat_json(定向抽取) = 2次LLM调用。
    # ================================================================
    missing_subgroups = _subgroups_for_missing(missing, field_name)
    for sg in missing_subgroups:
        # 该子组内实际缺失的字段（可能子组有3个target_fields但只缺1个）
        sg_missing = [f for f in sg.target_fields if f in missing]
        if not sg_missing:
            continue

        hyde_results = _round2_hyde(store, doc_id, sg, top_k=top_k)
        if not hyde_results:
            continue

        all_source_ids.extend(_format_source_ids(hyde_results))
        # 定向抽取：只让LLM提取sg_missing中的字段
        new_data = _targeted_extract(hyde_results, field_name, sg_missing, doc_id)
        if new_data:
            # 合并结果：只填空字段，不覆盖Round 1已有值
            data = _merge_data(data, new_data, sg_missing)

    # 重新检测缺失
    missing = _get_missing_fields(data, field_name)
    if not missing:
        logger.info(f"[{doc_id}] {field_name}: Round 2 全部字段已填充")
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.SUCCESS,
            data=data,
            raw_response=str(raw_result),
            source_chunks=all_source_ids,
        )

    logger.info(f"[{doc_id}] {field_name}: Round 2后仍缺失 {missing}，启动Round 3 章节路由")

    # ================================================================
    # Round 3: 章节路由补全
    #
    # 与Round 2的区别：不用HyDE，而是根据section_keywords精确定位章节。
    # 这是"最后手段"——如果连章节路由都找不到，说明报告中可能确实没有这个信息。
    # 每个组：1次chat_json(定向抽取)，不需要chat_raw（没有HyDE步骤）。
    # ================================================================
    missing_subgroups = _subgroups_for_missing(missing, field_name)
    for sg in missing_subgroups:
        sg_missing = [f for f in sg.target_fields if f in missing]
        if not sg_missing:
            continue

        section_results = _round3_section(store, doc_id, sg, top_k=top_k)
        if not section_results:
            continue

        all_source_ids.extend(_format_source_ids(section_results))
        new_data = _targeted_extract(section_results, field_name, sg_missing, doc_id)
        if new_data:
            data = _merge_data(data, new_data, sg_missing)

    # ================================================================
    # 最终结果：三轮都跑完后，再检测一次缺失
    # - 全部填充 → SUCCESS
    # - 仍有缺失 → PARTIAL（新状态，表示"部分成功"）
    # ================================================================
    missing = _get_missing_fields(data, field_name)
    status = ExtractionStatus.SUCCESS if not missing else ExtractionStatus.PARTIAL
    if missing:
        logger.warning(f"[{doc_id}] {field_name}: 三轮后仍缺失 {missing}")
    else:
        logger.info(f"[{doc_id}] {field_name}: Round 3 全部字段已填充")

    return ExtractionResult(
        field_name=field_name,
        doc_id=doc_id,
        status=status,
        data=data,
        raw_response=str(raw_result),
        source_chunks=all_source_ids,
    )


def extract_facts(
    store: VectorStore,
    doc_id: str,
    company_name: str = "",
    stock_code: str = "",
) -> FactRecord:
    """抽取一个文档的全部事实

    Args:
        store: 向量存储
        doc_id: 文档标识
        company_name: 公司名
        stock_code: 股票代码

    Returns:
        完整的FactRecord
    """
    logger.info(f"开始抽取事实: {doc_id}")

    record = FactRecord(
        doc_id=doc_id,
        company_name=company_name,
        stock_code=stock_code,
    )
    raw_responses = {}
    overall_status = ExtractionStatus.SUCCESS

    for field_name in FIELD_CONFIGS:
        result = extract_field(store, doc_id, field_name)
        raw_responses[field_name] = result.raw_response

        if result.status == ExtractionStatus.FAILED:
            overall_status = ExtractionStatus.PARTIAL
            logger.warning(f"[{doc_id}] {field_name} 抽取失败: {result.error}")
            continue

        # PARTIAL是多轮fallback新增的状态，表示部分字段三轮后仍缺失
        if result.status == ExtractionStatus.PARTIAL:
            overall_status = ExtractionStatus.PARTIAL

        # 将抽取结果赋值到record
        if field_name == "deal_summary" and result.data:
            record.deal_summary = DealSummary.model_validate(result.data)
        elif field_name == "acquisition_purpose" and result.data:
            record.acquisition_purpose = AcquisitionPurpose.model_validate(result.data)

    # 如果所有字段都失败，整体标记为FAILED
    if all(
        raw_responses.get(f) == "" or raw_responses.get(f) is None
        for f in FIELD_CONFIGS
    ):
        overall_status = ExtractionStatus.FAILED

    record.status = overall_status
    record.raw_responses = raw_responses
    logger.info(f"事实抽取完成: {doc_id}, status={overall_status.value}")
    return record
