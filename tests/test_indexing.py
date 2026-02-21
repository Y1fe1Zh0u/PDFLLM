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
