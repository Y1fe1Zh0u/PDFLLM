"""向量检索接口

封装VectorStore的检索功能，提供面向业务的查询接口。
"""

from src.indexing.store import VectorStore
from src.infra.embedder import embed_text
from src.infra.logger import get_logger
from src.infra.models import Chunk

logger = get_logger(__name__)


def search_chunks(
    store: VectorStore,
    query: str,
    top_k: int | None = None,
    doc_id: str | None = None,
) -> list[tuple[Chunk, float]]:
    """用自然语言查询检索相关chunks

    Args:
        store: 向量存储实例
        query: 查询文本
        top_k: 返回结果数量
        doc_id: 过滤特定文档

    Returns:
        (Chunk, score) 元组列表，按相似度降序
    """
    query_embedding = embed_text(query)
    results = store.search(query_embedding, top_k=top_k, doc_id=doc_id)
    logger.debug(f"查询'{query[:30]}...' 返回 {len(results)} 结果")
    return results
