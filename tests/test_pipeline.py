"""pipeline模块测试（集成测试，mock外部依赖）"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.infra.db import FactDB
from src.infra.models import Chunk, ExtractionStatus, FactRecord
from src.pipeline import process_single_pdf, run_pipeline


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


MOCK_PAGES = [
    {"page_number": 1, "text": "## 第一节 交易概述\n收购方为测试公司A，标的方为测试公司B"},
    {"page_number": 2, "text": "交易金额10亿元，发行价格10元/股"},
    {"page_number": 3, "text": "## 第三节 并购目的\n本次并购旨在拓展业务，实现协同效应"},
]

MOCK_DEAL = {
    "acquirer": "测试公司A",
    "target": "测试公司B",
    "deal_type": "发行股份购买资产",
    "deal_amount": "10亿元",
    "share_price": "10元/股",
    "payment_method": "发行股份",
    "target_valuation": "",
    "valuation_method": "",
    "performance_commitment": "",
}

MOCK_PURPOSE = {
    "strategic_purpose": "拓展业务",
    "synergy": "协同效应",
    "industry_logic": "",
    "summary": "拓展业务实现协同",
}


@patch("src.indexing.store.embed_texts", _fake_embed_texts)
@patch("src.indexing.search.embed_text", _fake_embed_text)
@patch("src.extraction.extractor.chat_json")
@patch("src.pipeline.extract_pages", return_value=MOCK_PAGES)
class TestProcessSinglePdf:
    def test_success(self, mock_extract, mock_chat, tmp_path):
        def _chat_side_effect(system, user, **kw):
            if "交易概要" in system:
                return MOCK_DEAL
            return MOCK_PURPOSE

        mock_chat.side_effect = _chat_side_effect

        from src.indexing.store import VectorStore

        store = VectorStore(index_dir=str(tmp_path / "index"))
        db = FactDB(db_path=str(tmp_path / "test.db"))

        # 创建一个假PDF路径
        pdf_path = tmp_path / "000001测试公司.pdf"
        pdf_path.touch()

        ok = process_single_pdf(pdf_path, store, db)
        assert ok is True

        # 验证数据库中的记录
        record = db.get_fact("000001测试公司")
        assert record is not None
        assert record.deal_summary.acquirer == "测试公司A"

    def test_empty_pdf(self, mock_extract, mock_chat, tmp_path):
        mock_extract.return_value = []

        from src.indexing.store import VectorStore

        store = VectorStore(index_dir=str(tmp_path / "index"))
        db = FactDB(db_path=str(tmp_path / "test.db"))

        pdf_path = tmp_path / "empty.pdf"
        pdf_path.touch()

        ok = process_single_pdf(pdf_path, store, db)
        assert ok is False


class TestRunPipeline:
    def test_no_pdf_files(self, tmp_path):
        stats = run_pipeline(str(tmp_path))
        assert stats["total"] == 0

    def test_nonexistent_path(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            run_pipeline("/nonexistent/path")

    @patch("src.indexing.store.embed_texts", _fake_embed_texts)
    @patch("src.indexing.search.embed_text", _fake_embed_text)
    @patch("src.extraction.extractor.chat_json")
    @patch("src.pipeline.extract_pages", return_value=MOCK_PAGES)
    def test_resume_skips_processed(self, mock_extract, mock_chat, tmp_path):
        mock_chat.return_value = MOCK_DEAL

        from src.indexing.store import VectorStore

        db = FactDB(db_path=str(tmp_path / "test.db"))

        # 预先插入一条已处理记录
        db.save_fact(FactRecord(doc_id="already_done", status=ExtractionStatus.SUCCESS))

        # 创建PDF文件
        (tmp_path / "already_done.pdf").touch()
        (tmp_path / "new_doc.pdf").touch()

        with patch("src.pipeline.VectorStore") as MockStore:
            mock_store_instance = MagicMock()
            mock_store_instance.load.return_value = False
            mock_store_instance.search.return_value = []
            MockStore.return_value = mock_store_instance

            with patch("src.pipeline.FactDB", return_value=db):
                stats = run_pipeline(str(tmp_path), resume=True)

        assert stats["skipped"] == 1
