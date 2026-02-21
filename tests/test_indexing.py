"""indexing模块测试（mock embedder避免加载模型）"""

import tempfile
from unittest.mock import patch

import numpy as np

from src.infra.models import Chunk
from src.indexing.store import VectorStore


EMBED_DIM = 16


def _fake_embed_texts(texts: list[str]) -> np.ndarray:
    """生成假的embedding（固定维度，基于文本hash保证可复现）"""
    result = []
    for text in texts:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(EMBED_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        result.append(vec)
    return np.array(result, dtype=np.float32)


def _fake_embed_text(text: str) -> np.ndarray:
    return _fake_embed_texts([text])[0]


def _make_chunks(n: int = 5, doc_id: str = "test_doc") -> list[Chunk]:
    return [
        Chunk(
            doc_id=doc_id,
            chunk_id=i,
            text=f"这是第{i}个测试文本片段，包含一些内容。" * 3,
            page=i + 1,
            section=f"第{i}节",
        )
        for i in range(n)
    ]


@patch("src.indexing.store.embed_texts", _fake_embed_texts)
class TestVectorStore:
    def test_build_and_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            chunks = _make_chunks()
            store.build(chunks)

            assert store.index is not None
            assert store.index.ntotal == 5

            query_vec = _fake_embed_text("测试查询")
            results = store.search(query_vec, top_k=3)
            assert len(results) == 3
            assert all(isinstance(r[0], Chunk) for r in results)
            assert all(isinstance(r[1], float) for r in results)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 构建并保存
            store = VectorStore(index_dir=tmpdir)
            chunks = _make_chunks()
            store.build(chunks)
            store.save()

            # 加载
            store2 = VectorStore(index_dir=tmpdir)
            assert store2.load() is True
            assert len(store2.chunks) == 5
            assert store2.index.ntotal == 5

    def test_add_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            store.build(_make_chunks(3, doc_id="doc1"))
            store.add(_make_chunks(2, doc_id="doc2"))
            assert store.index.ntotal == 5
            assert store.get_indexed_doc_ids() == {"doc1", "doc2"}

    def test_search_filter_doc_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            store.build(_make_chunks(3, doc_id="doc1"))
            store.add(_make_chunks(3, doc_id="doc2"))

            query_vec = _fake_embed_text("测试")
            results = store.search(query_vec, top_k=5, doc_id="doc1")
            assert all(r[0].doc_id == "doc1" for r in results)

    def test_search_empty_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            query_vec = _fake_embed_text("测试")
            results = store.search(query_vec)
            assert results == []

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            assert store.load() is False

    def test_get_indexed_doc_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            store.build(_make_chunks(2, doc_id="a") + _make_chunks(3, doc_id="b"))
            assert store.get_indexed_doc_ids() == {"a", "b"}


@patch("src.indexing.store.embed_texts", _fake_embed_texts)
class TestVectorStoreBySection:
    """search_by_section方法测试"""

    def _make_section_chunks(self):
        """创建包含不同section的chunks"""
        return [
            Chunk(doc_id="doc1", chunk_id=0, text="收购方为深赤湾", page=1, section="交易概述"),
            Chunk(doc_id="doc1", chunk_id=1, text="交易金额50亿", page=2, section="交易方案"),
            Chunk(doc_id="doc1", chunk_id=2, text="估值采用收益法", page=3, section="评估方法"),
            Chunk(doc_id="doc1", chunk_id=3, text="业绩承诺三年", page=4, section="承诺与补偿"),
            Chunk(doc_id="doc1", chunk_id=4, text="行业发展趋势", page=5, section="行业分析"),
        ]

    def test_section_filter(self):
        """匹配section关键词的chunks应被正确过滤"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            store.build(self._make_section_chunks())

            query_vec = _fake_embed_text("估值方法")
            results = store.search_by_section(
                query_vec, section_keywords=["评估"], top_k=3
            )
            assert len(results) >= 1
            # 所有结果的section都应包含关键词
            for chunk, score in results:
                assert any(kw in chunk.section for kw in ["评估"])

    def test_multiple_keywords(self):
        """多个关键词应匹配任一即可"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            store.build(self._make_section_chunks())

            query_vec = _fake_embed_text("交易信息")
            results = store.search_by_section(
                query_vec, section_keywords=["交易", "方案"], top_k=5
            )
            assert len(results) >= 2
            sections = [c.section for c, _ in results]
            assert "交易概述" in sections
            assert "交易方案" in sections

    def test_no_matching_section(self):
        """无匹配section应返回空列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            store.build(self._make_section_chunks())

            query_vec = _fake_embed_text("不存在的内容")
            results = store.search_by_section(
                query_vec, section_keywords=["不存在的章节"], top_k=3
            )
            assert results == []

    def test_doc_id_filter(self):
        """doc_id过滤应生效"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            chunks = self._make_section_chunks() + [
                Chunk(doc_id="doc2", chunk_id=0, text="另一个文档的交易", page=1, section="交易概述"),
            ]
            store.build(chunks)

            query_vec = _fake_embed_text("交易")
            results = store.search_by_section(
                query_vec, section_keywords=["交易"], top_k=5, doc_id="doc1"
            )
            assert all(c.doc_id == "doc1" for c, _ in results)

    def test_empty_store(self):
        """空store应返回空列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(index_dir=tmpdir)
            query_vec = _fake_embed_text("测试")
            results = store.search_by_section(
                query_vec, section_keywords=["测试"], top_k=3
            )
            assert results == []
