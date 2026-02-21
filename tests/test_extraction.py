"""extraction模块测试（mock LLM和embedder）"""

import json
import tempfile
from unittest.mock import patch, call

import numpy as np

from src.extraction.extractor import (
    extract_facts,
    extract_field,
    _get_missing_fields,
    _merge_data,
    _deduplicate_chunks,
    _subgroups_for_missing,
)
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
@patch("src.extraction.extractor.embed_text", _fake_embed_text)
class TestExtractField:
    @patch("src.extraction.extractor.chat_raw", return_value="")
    @patch("src.extraction.extractor.chat_json", return_value=MOCK_DEAL_RESPONSE)
    def test_extract_deal_summary(self, mock_chat, mock_raw):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_store(tmpdir)
            result = extract_field(store, "test_doc", "deal_summary")

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
@patch("src.extraction.extractor.embed_text", _fake_embed_text)
class TestExtractFacts:
    def _mock_chat_json(self, system_prompt, user_prompt, **kwargs):
        """根据prompt内容返回不同的mock结果"""
        if "交易概要" in system_prompt:
            return MOCK_DEAL_RESPONSE
        elif "并购目的" in system_prompt:
            return MOCK_PURPOSE_RESPONSE
        return None

    @patch("src.extraction.extractor.chat_raw", return_value="")
    @patch("src.extraction.extractor.chat_json")
    def test_extract_facts_success(self, mock_chat, mock_raw):
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


# --- 多轮fallback测试 ---


class TestHelperFunctions:
    """helper函数单元测试"""

    def test_get_missing_fields_all_filled(self):
        data = {
            "acquirer": "深赤湾",
            "target": "招商港口",
            "deal_type": "发行股份购买资产",
            "deal_amount": "50亿",
            "share_price": "15元",
            "payment_method": "发行股份",
            "target_valuation": "100亿",
            "valuation_method": "收益法",
            "performance_commitment": "三年业绩承诺",
        }
        missing = _get_missing_fields(data, "deal_summary")
        assert missing == []

    def test_get_missing_fields_some_empty(self):
        data = {
            "acquirer": "深赤湾",
            "target": "招商港口",
            "deal_type": "发行股份购买资产",
            "deal_amount": "50亿",
            "share_price": "",
            "payment_method": "",
            "target_valuation": "",
            "valuation_method": "",
            "performance_commitment": "",
        }
        missing = _get_missing_fields(data, "deal_summary")
        assert set(missing) == {
            "share_price", "payment_method",
            "target_valuation", "valuation_method",
            "performance_commitment",
        }

    def test_get_missing_fields_unknown_field_name(self):
        """未知字段名应返回空列表"""
        missing = _get_missing_fields({"foo": "bar"}, "unknown_field")
        assert missing == []

    def test_merge_data_fills_empty(self):
        existing = {"acquirer": "深赤湾", "target_valuation": "", "valuation_method": ""}
        new_data = {"target_valuation": "100亿", "target_valuation_quote": "评估值为100亿", "valuation_method": "收益法"}
        merged = _merge_data(existing, new_data, ["target_valuation", "valuation_method"])
        assert merged["acquirer"] == "深赤湾"  # 保持不变
        assert merged["target_valuation"] == "100亿"
        assert merged["target_valuation_quote"] == "评估值为100亿"
        assert merged["valuation_method"] == "收益法"

    def test_merge_data_no_overwrite(self):
        existing = {"acquirer": "深赤湾", "target_valuation": "已有值"}
        new_data = {"acquirer": "新名称", "target_valuation": "新值"}
        merged = _merge_data(existing, new_data, ["acquirer", "target_valuation"])
        assert merged["acquirer"] == "深赤湾"  # 已有值不被覆盖
        assert merged["target_valuation"] == "已有值"

    def test_merge_data_empty_new_value_ignored(self):
        existing = {"target_valuation": ""}
        new_data = {"target_valuation": ""}
        merged = _merge_data(existing, new_data, ["target_valuation"])
        assert merged["target_valuation"] == ""

    def test_deduplicate_chunks_keeps_best_score(self):
        c1 = Chunk(doc_id="d", chunk_id=0, text="a", page=1)
        c2 = Chunk(doc_id="d", chunk_id=1, text="b", page=2)

        list1 = [(c1, 0.9), (c2, 0.7)]
        list2 = [(c1, 0.8), (c2, 0.95)]

        deduped = _deduplicate_chunks([list1, list2])
        scores = {c.chunk_id: s for c, s in deduped}
        assert scores[0] == 0.9   # c1保留list1的更高分
        assert scores[1] == 0.95  # c2保留list2的更高分

    def test_deduplicate_chunks_sorted_by_score(self):
        c1 = Chunk(doc_id="d", chunk_id=0, text="a", page=1)
        c2 = Chunk(doc_id="d", chunk_id=1, text="b", page=2)

        deduped = _deduplicate_chunks([[(c1, 0.5)], [(c2, 0.9)]])
        assert deduped[0][0].chunk_id == 1  # 高分在前
        assert deduped[1][0].chunk_id == 0

    def test_deduplicate_chunks_empty(self):
        assert _deduplicate_chunks([]) == []
        assert _deduplicate_chunks([[], []]) == []

    def test_subgroups_for_missing(self):
        groups = _subgroups_for_missing(["target_valuation", "valuation_method"], "deal_summary")
        assert len(groups) == 1
        assert groups[0].name == "valuation"

    def test_subgroups_for_missing_multiple(self):
        groups = _subgroups_for_missing(
            ["target_valuation", "performance_commitment"], "deal_summary"
        )
        names = {g.name for g in groups}
        assert names == {"valuation", "commitment"}


def _make_rich_store(tmpdir) -> VectorStore:
    """创建包含多个section的测试store"""
    store = VectorStore(index_dir=tmpdir)
    chunks = [
        Chunk(doc_id="doc1", chunk_id=0, text="收购方为深赤湾A，标的方为招商港口B", page=1, section="交易概述"),
        Chunk(doc_id="doc1", chunk_id=1, text="交易金额50亿元，发行价格15元/股", page=2, section="交易方案"),
        Chunk(doc_id="doc1", chunk_id=2, text="标的资产评估值为100亿元，采用收益法评估", page=3, section="评估方法"),
        Chunk(doc_id="doc1", chunk_id=3, text="交易对方承诺未来三年净利润不低于5亿", page=4, section="承诺与补偿"),
        Chunk(doc_id="doc1", chunk_id=4, text="本次并购旨在整合港口资源", page=5, section="并购目的"),
    ]
    store.build(chunks)
    return store


# Round 1全部填充的mock response
MOCK_DEAL_FULL = {
    "acquirer": "深赤湾",
    "target": "招商港口",
    "deal_type": "发行股份购买资产",
    "deal_amount": "50亿",
    "share_price": "15元",
    "payment_method": "发行股份",
    "target_valuation": "100亿",
    "valuation_method": "收益法",
    "performance_commitment": "三年5亿",
}

# Round 1部分缺失的mock response
MOCK_DEAL_PARTIAL = {
    "acquirer": "深赤湾",
    "target": "招商港口",
    "deal_type": "发行股份购买资产",
    "deal_amount": "50亿",
    "share_price": "15元",
    "payment_method": "发行股份",
    "target_valuation": "",
    "valuation_method": "",
    "performance_commitment": "",
}

# HyDE定向抽取补全的mock response
MOCK_TARGETED_VALUATION = {
    "target_valuation": "100亿元",
    "target_valuation_quote": "标的资产评估值为100亿元",
    "valuation_method": "收益法",
}

MOCK_TARGETED_COMMITMENT = {
    "performance_commitment": "三年净利润不低于5亿",
    "performance_commitment_quote": "交易对方承诺未来三年净利润不低于5亿",
}


@patch("src.indexing.store.embed_texts", _fake_embed_texts)
@patch("src.indexing.search.embed_text", _fake_embed_text)
@patch("src.extraction.extractor.embed_text", _fake_embed_text)
class TestMultiRoundExtraction:
    """多轮fallback抽取测试"""

    @patch("src.extraction.extractor.chat_raw", return_value="")
    @patch("src.extraction.extractor.chat_json", return_value=MOCK_DEAL_FULL)
    def test_round1_all_filled_no_further_rounds(self, mock_chat_json, mock_chat_raw):
        """Round 1全部填充时不应触发Round 2/3"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_rich_store(tmpdir)
            result = extract_field(store, "doc1", "deal_summary")

            assert result.status == ExtractionStatus.SUCCESS
            assert result.data["valuation_method"] == "收益法"
            # chat_json只被调用一次（Round 1），不触发定向抽取
            assert mock_chat_json.call_count == 1
            # chat_raw不应被调用（无HyDE）
            mock_chat_raw.assert_not_called()

    @patch("src.extraction.extractor.chat_raw", return_value="假设标的估值100亿采用收益法")
    @patch("src.extraction.extractor.chat_json")
    def test_round2_hyde_fills_missing(self, mock_chat_json, mock_chat_raw):
        """Round 2 HyDE应补全缺失字段"""
        call_count = [0]

        def mock_chat_side_effect(system_prompt, user_prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Round 1返回部分缺失
                return MOCK_DEAL_PARTIAL
            elif "估值" in system_prompt or "target_valuation" in system_prompt:
                return MOCK_TARGETED_VALUATION
            elif "承诺" in system_prompt or "performance_commitment" in system_prompt:
                return MOCK_TARGETED_COMMITMENT
            return {}

        mock_chat_json.side_effect = mock_chat_side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_rich_store(tmpdir)
            result = extract_field(store, "doc1", "deal_summary")

            # 缺失字段应被补全
            assert result.data["acquirer"] == "深赤湾"  # Round 1已有
            assert result.data["target_valuation"] == "100亿元"  # Round 2补全
            assert result.data["valuation_method"] == "收益法"

    @patch("src.extraction.extractor.chat_raw", return_value="")
    @patch("src.extraction.extractor.chat_json")
    def test_no_overwrite_existing_values(self, mock_chat_json, mock_chat_raw):
        """后续轮次不应覆盖Round 1已有的值"""
        call_count = [0]

        def mock_chat_side_effect(system_prompt, user_prompt, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return MOCK_DEAL_PARTIAL
            # 定向抽取返回的数据包含已有字段的不同值
            return {
                "acquirer": "不同的收购方",
                "target_valuation": "200亿",
                "valuation_method": "市场法",
                "performance_commitment": "五年承诺",
            }

        mock_chat_json.side_effect = mock_chat_side_effect

        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_rich_store(tmpdir)
            result = extract_field(store, "doc1", "deal_summary")

            # Round 1已有的acquirer不应被覆盖
            assert result.data["acquirer"] == "深赤湾"

    @patch("src.extraction.extractor.settings")
    @patch("src.extraction.extractor.chat_json", return_value=MOCK_DEAL_PARTIAL)
    def test_multi_round_disabled(self, mock_chat_json, mock_settings):
        """enable_multi_round=False应恢复原始行为"""
        mock_settings.enable_multi_round = False
        mock_settings.top_k = 8

        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_rich_store(tmpdir)
            result = extract_field(store, "doc1", "deal_summary")

            assert result.status == ExtractionStatus.SUCCESS
            # 只调用一次LLM
            assert mock_chat_json.call_count == 1
            # 缺失字段仍为空（因为没有多轮fallback）
            assert result.data["target_valuation"] == ""
