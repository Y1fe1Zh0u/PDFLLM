"""端到端编排

串联 ingestion → indexing → extraction → storage，支持断点续跑。
"""

from pathlib import Path

from tqdm import tqdm

from src.extraction.extractor import extract_facts
from src.indexing.store import VectorStore
from src.infra.config import settings
from src.infra.db import FactDB
from src.infra.logger import get_logger, setup_logger
from src.infra.models import ExtractionStatus
from src.ingestion.chunker import chunk_pages
from src.ingestion.extractor import extract_doc_id, extract_metadata_from_filename, extract_pages

logger = get_logger(__name__)


def process_single_pdf(
    pdf_path: Path,
    store: VectorStore,
    db: FactDB,
) -> bool:
    """处理单个PDF文件

    Returns:
        是否成功
    """
    doc_id = extract_doc_id(pdf_path)
    logger.info(f"=== 处理文档: {doc_id} ===")

    try:
        # 1. 提取文本
        pages = extract_pages(pdf_path)
        if not pages:
            logger.warning(f"[{doc_id}] PDF提取无内容，跳过")
            return False

        # 2. 切片
        chunks = chunk_pages(pages, doc_id)
        logger.info(f"[{doc_id}] 生成 {len(chunks)} 个chunks")

        # 2.5 保存chunks到JSONL（方便调试）
        chunks_dir = Path(settings.output_dir) / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = chunks_dir / f"{doc_id}.jsonl"
        with open(chunks_file, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
        logger.info(f"[{doc_id}] chunks已保存: {chunks_file}")

        # 3. 索引
        store.add(chunks)

        # 4. 提取元数据
        meta = extract_metadata_from_filename(pdf_path.name)

        # 5. LLM抽取事实
        record = extract_facts(
            store,
            doc_id,
            company_name=meta["company_name"],
            stock_code=meta["stock_code"],
        )

        # 6. 存储
        db.save_fact(record)

        if record.status == ExtractionStatus.FAILED:
            logger.warning(f"[{doc_id}] 抽取失败")
            return False

        logger.info(f"[{doc_id}] 处理完成, status={record.status.value}")
        return True

    except Exception as e:
        logger.error(f"[{doc_id}] 处理异常: {e}", exc_info=True)
        return False


def run_pipeline(
    input_path: str,
    resume: bool = True,
) -> dict:
    """运行完整pipeline

    Args:
        input_path: PDF文件路径或目录路径
        resume: 是否跳过已处理的文档（断点续跑）

    Returns:
        统计结果 {"total": N, "success": N, "failed": N, "skipped": N}
    """
    # 初始化日志文件
    settings.ensure_dirs()
    setup_logger("pdfllm", log_file=f"{settings.log_dir}/pipeline.log")

    input_path = Path(input_path)
    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
    else:
        raise FileNotFoundError(f"路径不存在: {input_path}")

    if not pdf_files:
        logger.warning(f"未找到PDF文件: {input_path}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    logger.info(f"找到 {len(pdf_files)} 个PDF文件")

    # 初始化组件
    store = VectorStore()
    store.load()  # 尝试加载已有索引

    db = FactDB()
    processed = db.get_processed_doc_ids() if resume else set()

    stats = {"total": len(pdf_files), "success": 0, "failed": 0, "skipped": 0}

    for pdf_path in tqdm(pdf_files, desc="处理PDF"):
        doc_id = extract_doc_id(pdf_path)

        # 断点续跑：跳过已处理的
        if doc_id in processed:
            logger.info(f"[{doc_id}] 已处理，跳过")
            stats["skipped"] += 1
            continue

        ok = process_single_pdf(pdf_path, store, db)
        if ok:
            stats["success"] += 1
        else:
            stats["failed"] += 1

    # 保存索引
    store.save()

    logger.info(
        f"Pipeline完成: 总计{stats['total']}, "
        f"成功{stats['success']}, 失败{stats['failed']}, 跳过{stats['skipped']}"
    )
    return stats
