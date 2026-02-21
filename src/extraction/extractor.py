"""LLM事实抽取

检索相关chunks → 拼装prompt → 调LLM → Pydantic校验输出。
支持多轮fallback检索：Round 1子字段组检索 → Round 2 HyDE → Round 3章节路由。
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


# --- 多轮fallback helper函数 ---


def _get_missing_fields(data: dict, field_name: str) -> list[str]:
    """检测结果中值为空字符串的主字段（不含_quote后缀）

    Args:
        data: LLM返回的字段字典
        field_name: 字段组名（deal_summary / acquisition_purpose）

    Returns:
        空值的主字段名列表
    """
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
    """找到覆盖缺失字段的SubFieldGroup"""
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
    """合并结果，只填空字段不覆盖已有值

    Args:
        existing: 已有数据
        new_data: 新抽取的数据
        missing_fields: 需要填充的主字段名列表

    Returns:
        合并后的数据字典
    """
    merged = dict(existing)
    for field in missing_fields:
        new_val = new_data.get(field, "")
        if new_val and not merged.get(field, ""):
            merged[field] = new_val
            # 同时填充对应的_quote字段
            quote_key = f"{field}_quote"
            if quote_key in new_data and not merged.get(quote_key, ""):
                merged[quote_key] = new_data[quote_key]
    return merged


def _deduplicate_chunks(
    chunk_lists: list[list[tuple[Chunk, float]]],
) -> list[tuple[Chunk, float]]:
    """多组检索结果去重，保留最高分

    Args:
        chunk_lists: 多组 (Chunk, score) 结果列表

    Returns:
        去重后的结果列表，按分数降序
    """
    best: dict[int, tuple[Chunk, float]] = {}
    for results in chunk_lists:
        for chunk, score in results:
            key = chunk.chunk_id
            # 用(doc_id, chunk_id)作为唯一标识
            composite_key = hash((chunk.doc_id, chunk.chunk_id))
            if composite_key not in best or score > best[composite_key][1]:
                best[composite_key] = (chunk, score)

    sorted_results = sorted(best.values(), key=lambda x: x[1], reverse=True)
    return sorted_results


def _format_chunks_text(results: list[tuple[Chunk, float]]) -> str:
    """将检索结果格式化为prompt文本"""
    return "\n\n---\n\n".join(
        f"[页码:{c.page}, 章节:{c.section}]\n{c.text}" for c, _ in results
    )


def _format_source_ids(results: list[tuple[Chunk, float]]) -> list[str]:
    """提取来源标识"""
    return [f"{c.doc_id}:chunk{c.chunk_id}:p{c.page}" for c, _ in results]


def _round1_retrieve(
    store: VectorStore, doc_id: str, field_name: str, top_k: int | None = None
) -> list[tuple[Chunk, float]]:
    """Round 1: 按子字段组分别检索 + 合并去重"""
    subgroups = FIELD_SUBGROUPS.get(field_name, [])
    if not subgroups:
        # fallback到原始query
        config = FIELD_CONFIGS[field_name]
        return search_chunks(store, config["query"], top_k=top_k, doc_id=doc_id)

    per_group_k = max(4, (top_k or settings.top_k) // len(subgroups))
    chunk_lists = []
    for sg in subgroups:
        results = search_chunks(store, sg.query, top_k=per_group_k, doc_id=doc_id)
        chunk_lists.append(results)

    deduped = _deduplicate_chunks(chunk_lists)
    final_k = top_k or settings.top_k
    return deduped[:final_k]


def _round2_hyde(
    store: VectorStore, doc_id: str, subgroup: SubFieldGroup, top_k: int | None = None
) -> list[tuple[Chunk, float]]:
    """Round 2: HyDE — LLM生成假设答案 → embed → 检索"""
    # 生成假设文本
    hyde_text = chat_raw(
        HYDE_SYSTEM,
        HYDE_USER.format(hyde_query=subgroup.hyde_query),
    )
    if not hyde_text:
        logger.warning(f"HyDE生成失败: {subgroup.name}")
        return []

    logger.debug(f"HyDE生成文本({subgroup.name}): {hyde_text[:80]}...")

    # 用假设文本做embed检索
    hyde_embedding = embed_text(hyde_text)
    results = store.search(hyde_embedding, top_k=top_k, doc_id=doc_id)
    return results


def _round3_section(
    store: VectorStore, doc_id: str, subgroup: SubFieldGroup, top_k: int | None = None
) -> list[tuple[Chunk, float]]:
    """Round 3: 章节路由 — section关键词过滤chunks → 子集内向量检索"""
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
    """定向LLM抽取缺失字段

    Args:
        chunks: 检索到的chunks
        field_name: 字段组名
        missing_fields: 需要抽取的字段列表
        doc_id: 文档ID

    Returns:
        抽取结果字典，失败返回None
    """
    if not chunks:
        return None

    # 构建字段描述
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
    """抽取单个事实字段（支持多轮fallback）

    Args:
        store: 向量存储
        doc_id: 文档标识
        field_name: 字段名（如 deal_summary）
        top_k: 检索chunks数量

    Returns:
        ExtractionResult
    """
    config = FIELD_CONFIGS[field_name]
    model_cls = FIELD_MODELS[field_name]

    # --- Round 1: 子字段组分别检索 → 全字段LLM抽取 ---
    if settings.enable_multi_round and field_name in FIELD_SUBGROUPS:
        results = _round1_retrieve(store, doc_id, field_name, top_k=top_k)
    else:
        results = search_chunks(store, config["query"], top_k=top_k, doc_id=doc_id)

    if not results:
        logger.warning(f"[{doc_id}] {field_name}: 未检索到相关chunks")
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.FAILED,
            error="未检索到相关chunks",
        )

    # 拼装prompt，调LLM
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

    # 如果未启用多轮fallback，直接返回
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

    # 检测缺失字段
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

    # --- Round 2: HyDE补全 ---
    missing_subgroups = _subgroups_for_missing(missing, field_name)
    for sg in missing_subgroups:
        sg_missing = [f for f in sg.target_fields if f in missing]
        if not sg_missing:
            continue

        hyde_results = _round2_hyde(store, doc_id, sg, top_k=top_k)
        if not hyde_results:
            continue

        all_source_ids.extend(_format_source_ids(hyde_results))
        new_data = _targeted_extract(hyde_results, field_name, sg_missing, doc_id)
        if new_data:
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

    # --- Round 3: 章节路由补全 ---
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

    # 最终检测
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
