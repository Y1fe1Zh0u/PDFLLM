"""extraction模块测试（mock LLM和embedder）"""

import json
import tempfile
from unittest.mock import patch

import numpy as np

from src.extraction.extractor import extract_facts, extract_field
from src.indexing.store import VectorStore
from src.infra.db import FactDB
from src.infra.models import (
    Chunk,
    DealSummary,
    ExtractionStatus,
    FactRecord,
)

EMBED_DIM = 16


def _fake_embed_texts(texts):
    result = []
    for text in texts:
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(EMBED_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        result.append(vec)
    return np.array(result, dtype=np.float32)


def _fake_embed_text(text):
    return _fake_embed_texts([text])[0]


def _make_store(tmpdir) -> VectorStore:
    """创建一个包含测试数据的store"""
    store = VectorStore(index_dir=tmpdir)
    chunks = [
        Chunk(doc_id="test_doc", chunk_id=0, text="收购方为深赤湾，标的方为招商港口", page=1, section="交易概述"),
        Chunk(doc_id="test_doc", chunk_id=1, text="交易金额50亿元，发行价格15元/股", page=2, section="交易方案"),
        Chunk(doc_id="test_doc", chunk_id=2, text="本次并购旨在整合港口资源，提升协同效应", page=3, section="并购目的"),
    ]
    store.build(chunks)
    return store


MOCK_DEAL_RESPONSE = {
    "acquirer": "深赤湾",
    "target": "招商港口",
    "deal_type": "发行股份购买资产",
    "deal_amount": "50亿元",
    "share_price": "15元/股",
    "payment_method": "发行股份",
    "target_valuation": "",
    "valuation_method": "",
    "performance_commitment": "",
}

MOCK_PURPOSE_RESPONSE = {
    "strategic_purpose": "整合港口资源",
    "synergy": "提升运营效率",
    "industry_logic": "港口行业整合趋势",
    "summary": "通过并购整合港口资源以提升协同效应",
}


@patch("src.indexing.store.embed_texts", _fake_embed_texts)
@patch("src.indexing.search.embed_text", _fake_embed_text)
class TestExtractField:
    @patch("src.extraction.extractor.chat_json", return_value=MOCK_DEAL_RESPONSE)
    def test_extract_deal_summary(self, mock_chat):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_store(tmpdir)
            result = extract_field(store, "test_doc", "deal_summary")

            assert result.status == ExtractionStatus.SUCCESS
            assert result.data["acquirer"] == "深赤湾"
            assert result.data["target"] == "招商港口"
            assert len(result.source_chunks) > 0

    @patch("src.extraction.extractor.chat_json", return_value=None)
    def test_extract_field_llm_failure(self, mock_chat):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_store(tmpdir)
            result = extract_field(store, "test_doc", "deal_summary")

            assert result.status == ExtractionStatus.FAILED
            assert "LLM" in result.error


@patch("src.indexing.store.embed_texts", _fake_embed_texts)
@patch("src.indexing.search.embed_text", _fake_embed_text)
class TestExtractFacts:
    def _mock_chat_json(self, system_prompt, user_prompt, **kwargs):
        """根据prompt内容返回不同的mock结果"""
        if "交易概要" in system_prompt:
            return MOCK_DEAL_RESPONSE
        elif "并购目的" in system_prompt:
            return MOCK_PURPOSE_RESPONSE
        return None

    @patch("src.extraction.extractor.chat_json")
    def test_extract_facts_success(self, mock_chat):
        mock_chat.side_effect = self._mock_chat_json
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_store(tmpdir)
            record = extract_facts(store, "test_doc", company_name="深赤湾", stock_code="000022")

            assert record.doc_id == "test_doc"
            assert record.company_name == "深赤湾"
            assert record.deal_summary.acquirer == "深赤湾"
            assert record.acquisition_purpose.strategic_purpose == "整合港口资源"

    @patch("src.extraction.extractor.chat_json", return_value=None)
    def test_extract_facts_all_failed(self, mock_chat):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_store(tmpdir)
            record = extract_facts(store, "test_doc")

            # 所有字段抽取失败
            assert record.deal_summary.acquirer == ""


class TestFactDB:
    def test_save_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FactDB(db_path=f"{tmpdir}/test.db")
            record = FactRecord(
                doc_id="test_doc",
                company_name="深赤湾",
                stock_code="000022",
                deal_summary=DealSummary(acquirer="深赤湾", target="招商港口"),
            )
            db.save_fact(record)

            loaded = db.get_fact("test_doc")
            assert loaded is not None
            assert loaded.company_name == "深赤湾"
            assert loaded.deal_summary.acquirer == "深赤湾"

    def test_get_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FactDB(db_path=f"{tmpdir}/test.db")
            assert db.get_fact("nonexistent") is None

    def test_list_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FactDB(db_path=f"{tmpdir}/test.db")
            for i in range(3):
                db.save_fact(FactRecord(doc_id=f"doc_{i}", company_name=f"公司{i}"))
            docs = db.list_documents()
            assert len(docs) == 3

    def test_get_processed_doc_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FactDB(db_path=f"{tmpdir}/test.db")
            db.save_fact(FactRecord(doc_id="a", status=ExtractionStatus.SUCCESS))
            db.save_fact(FactRecord(doc_id="b", status=ExtractionStatus.FAILED))
            db.save_fact(FactRecord(doc_id="c", status=ExtractionStatus.PARTIAL))

            processed = db.get_processed_doc_ids()
            assert processed == {"a", "c"}  # FAILED不算已处理

    def test_upsert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FactDB(db_path=f"{tmpdir}/test.db")
            db.save_fact(FactRecord(doc_id="test", company_name="旧名"))
            db.save_fact(FactRecord(doc_id="test", company_name="新名"))

            loaded = db.get_fact("test")
            assert loaded.company_name == "新名"
            assert len(db.list_documents()) == 1
