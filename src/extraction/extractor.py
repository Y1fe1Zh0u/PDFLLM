"""LLM事实抽取

检索相关chunks → 拼装prompt → 调LLM → Pydantic校验输出。
"""

from pydantic import ValidationError

from src.extraction.prompts import FIELD_CONFIGS
from src.indexing.search import search_chunks
from src.indexing.store import VectorStore
from src.infra.llm_client import chat_json
from src.infra.logger import get_logger
from src.infra.models import (
    AcquisitionPurpose,
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


def extract_field(
    store: VectorStore,
    doc_id: str,
    field_name: str,
    top_k: int | None = None,
) -> ExtractionResult:
    """抽取单个事实字段

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

    # 1. 检索相关chunks
    results = search_chunks(store, config["query"], top_k=top_k, doc_id=doc_id)
    if not results:
        logger.warning(f"[{doc_id}] {field_name}: 未检索到相关chunks")
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.FAILED,
            error="未检索到相关chunks",
        )

    # 2. 拼装prompt
    chunks_text = "\n\n---\n\n".join(
        f"[页码:{c.page}, 章节:{c.section}]\n{c.text}" for c, _ in results
    )
    source_ids = [f"{c.doc_id}:chunk{c.chunk_id}:p{c.page}" for c, _ in results]

    user_prompt = config["user_prompt_template"].format(chunks_text=chunks_text)

    logger.info(f"[{doc_id}] {field_name}: 使用 {len(results)} chunks 调用LLM")

    # 3. 调LLM
    raw_result = chat_json(config["system_prompt"], user_prompt)

    if raw_result is None:
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.FAILED,
            source_chunks=source_ids,
            error="LLM调用失败",
        )

    # 4. Pydantic校验
    try:
        validated = model_cls.model_validate(raw_result)
        logger.info(f"[{doc_id}] {field_name}: 抽取成功")
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.SUCCESS,
            data=validated.model_dump(),
            raw_response=str(raw_result),
            source_chunks=source_ids,
        )
    except ValidationError as e:
        logger.warning(f"[{doc_id}] {field_name}: Pydantic校验失败: {e}")
        return ExtractionResult(
            field_name=field_name,
            doc_id=doc_id,
            status=ExtractionStatus.FAILED,
            raw_response=str(raw_result),
            source_chunks=source_ids,
            error=f"校验失败: {e}",
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
