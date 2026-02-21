"""FAISS向量索引存储

管理FAISS索引的创建、保存、加载，以及chunk元数据的持久化。
"""

from pathlib import Path

import faiss
import numpy as np

from src.infra.config import settings
from src.infra.embedder import embed_texts
from src.infra.logger import get_logger
from src.infra.models import Chunk

logger = get_logger(__name__)


class VectorStore:
    """FAISS向量索引管理器"""

    def __init__(self, index_dir: str | None = None):
        self.index_dir = Path(index_dir or settings.index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []

    def _index_path(self) -> Path:
        return self.index_dir / "faiss.bin"

    def _meta_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    def build(self, chunks: list[Chunk]) -> None:
        """从chunks构建索引

        Args:
            chunks: 要索引的文档切片列表
        """
        if not chunks:
            logger.warning("空chunks列表，跳过索引构建")
            return

        logger.info(f"开始构建索引: {len(chunks)} chunks")
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # 内积（配合normalize_embeddings使用）
        self.index.add(embeddings)
        self.chunks = chunks

        logger.info(f"索引构建完成: dim={dim}, size={self.index.ntotal}")

    def add(self, chunks: list[Chunk]) -> None:
        """向现有索引追加chunks"""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"追加 {len(chunks)} chunks，索引总量: {self.index.ntotal}")

    def save(self) -> None:
        """保存索引和元数据到磁盘"""
        if self.index is None:
            logger.warning("索引为空，跳过保存")
            return

        faiss.write_index(self.index, str(self._index_path()))

        with open(self._meta_path(), "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json() + "\n")

        logger.info(f"索引已保存: {self._index_path()}")

    def load(self) -> bool:
        """从磁盘加载索引和元数据

        Returns:
            是否成功加载
        """
        index_path = self._index_path()
        meta_path = self._meta_path()

        if not index_path.exists() or not meta_path.exists():
            logger.info("索引文件不存在，需要重新构建")
            return False

        self.index = faiss.read_index(str(index_path))

        self.chunks = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.chunks.append(Chunk.model_validate_json(line))

        logger.info(f"索引已加载: {self.index.ntotal} vectors, {len(self.chunks)} chunks")
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """向量检索

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            doc_id: 过滤特定文档

        Returns:
            (Chunk, score) 元组列表，按相似度降序
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = top_k or settings.top_k

        # 如果需要过滤doc_id，多检索一些再过滤
        search_k = top_k * 3 if doc_id else top_k

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, min(search_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if doc_id and chunk.doc_id != doc_id:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results

    def get_indexed_doc_ids(self) -> set[str]:
        """获取已索引的所有doc_id"""
        return {c.doc_id for c in self.chunks}
