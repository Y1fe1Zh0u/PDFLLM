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

    def search_by_section(
        self,
        query_embedding: np.ndarray,
        section_keywords: list[str],
        top_k: int | None = None,
        doc_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """在匹配section关键词的chunks子集内做向量检索（Round 3章节路由专用）

        与search()的区别：
        - search() 在全部chunks上做FAISS检索（全局搜索）
        - search_by_section() 先按section关键词过滤出候选子集，再在子集内计算相似度

        适用场景：
        - 并购报告的章节结构高度标准化，"估值方法"信息集中在"评估/估值/定价"章节
        - 全局搜索可能被其他高频章节的chunks"淹没"，section过滤能精确定位

        实现策略：
        - 不构建临时FAISS索引（候选集通常只有20-100个chunks，暴力计算足够快）
        - 用index.reconstruct(i)从已有FAISS索引中取回原始向量
        - 用numpy dot product计算相似度（因为embedding已normalize，dot = cosine）

        Args:
            query_embedding: 查询向量（已normalize）
            section_keywords: section过滤关键词，chunk.section包含任一关键词即匹配
            top_k: 返回结果数量
            doc_id: 过滤特定文档

        Returns:
            (Chunk, score) 元组列表，按相似度降序
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = top_k or settings.top_k

        # Step 1: 按section关键词 + doc_id 过滤出候选chunk的索引
        # 使用子串匹配：keyword="评估" 能匹配 section="交易标的的评估与定价"
        candidate_indices = []
        for i, chunk in enumerate(self.chunks):
            if doc_id and chunk.doc_id != doc_id:
                continue
            if any(kw in chunk.section for kw in section_keywords):
                candidate_indices.append(i)

        if not candidate_indices:
            return []

        # Step 2: 从FAISS索引中reconstruct候选向量，计算与query的内积（= cosine相似度）
        # IndexFlatIP原生支持reconstruct()，不需要额外存储向量
        query = query_embedding.reshape(-1).astype(np.float32)
        scored = []
        for i in candidate_indices:
            vec = self.index.reconstruct(i)
            score = float(np.dot(query, vec))
            scored.append((i, score))

        # Step 3: 按相似度降序取top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, score in scored[:top_k]:
            results.append((self.chunks[i], score))

        return results

    def get_indexed_doc_ids(self) -> set[str]:
        """获取已索引的所有doc_id"""
        return {c.doc_id for c in self.chunks}
