"""ingestion模块测试"""

from src.ingestion.extractor import extract_doc_id, extract_metadata_from_filename
from src.ingestion.chunker import chunk_pages, _find_sections, _split_text_and_tables


class TestExtractor:
    def test_extract_doc_id(self):
        assert extract_doc_id("/path/to/000022深赤湾.pdf") == "000022深赤湾"
        assert extract_doc_id("report.pdf") == "report"

    def test_extract_metadata_from_filename(self):
        meta = extract_metadata_from_filename("000022深赤湾Ａ_报告.pdf")
        assert meta["stock_code"] == "000022"
        assert "深赤湾" in meta["company_name"]

    def test_extract_metadata_no_code(self):
        meta = extract_metadata_from_filename("report.pdf")
        assert meta["stock_code"] == ""
        assert meta["company_name"] == ""

    def test_extract_metadata_truncates_at_verb(self):
        meta = extract_metadata_from_filename("000005世纪星源向特定对象发行股份.pdf")
        assert meta["company_name"] == "世纪星源"


class TestSectionDetection:
    def test_find_sections_numbered(self):
        text = "前言内容\n## 第一节 交易概述\n概述内容\n## 第二节 标的资产\n资产内容"
        sections = _find_sections(text)
        assert len(sections) == 2
        assert "第一节" in sections[0][1]
        assert "第二节" in sections[1][1]

    def test_find_sections_chinese_numbering(self):
        text = "一、交易概述\n内容\n二、标的资产\n内容"
        sections = _find_sections(text)
        assert len(sections) == 2

    def test_find_sections_empty(self):
        text = "这是一段普通文本，没有章节标题。"
        sections = _find_sections(text)
        assert len(sections) == 0


class TestTextTableSplit:
    def test_split_text_only(self):
        text = "这是纯文本内容\n\n第二段内容"
        blocks = _split_text_and_tables(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"

    def test_split_table_only(self):
        text = "|列1|列2|\n|---|---|\n|值1|值2|"
        blocks = _split_text_and_tables(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "table"

    def test_split_mixed(self):
        text = "前面的文本\n|列1|列2|\n|---|---|\n|值1|值2|\n后面的文本"
        blocks = _split_text_and_tables(text)
        assert len(blocks) == 3
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "table"
        assert blocks[2]["type"] == "text"


class TestChunker:
    def test_chunk_pages_basic(self):
        pages = [
            {"page_number": 1, "text": "这是第一页的内容。" * 50},
            {"page_number": 2, "text": "这是第二页的内容。" * 50},
        ]
        chunks = chunk_pages(pages, doc_id="test_doc", chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 0
        assert all(c.doc_id == "test_doc" for c in chunks)
        assert chunks[0].page == 1

    def test_chunk_pages_empty(self):
        chunks = chunk_pages([], doc_id="empty")
        assert chunks == []

    def test_chunk_pages_section_tracking(self):
        pages = [
            {"page_number": 1, "text": "## 第一节 交易概述\n" + "交易内容。" * 100},
            {"page_number": 2, "text": "## 第二节 标的资产\n" + "资产描述。" * 100},
        ]
        chunks = chunk_pages(pages, doc_id="test", chunk_size=100, chunk_overlap=20)
        assert chunks[0].section == "第一节 交易概述"
        last = chunks[-1]
        assert "第二节" in last.section

    def test_chunk_id_sequential(self):
        pages = [{"page_number": 1, "text": "内容" * 200}]
        chunks = chunk_pages(pages, doc_id="test", chunk_size=100, chunk_overlap=20)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == i

    def test_table_separate_chunk(self):
        pages = [
            {"page_number": 1, "text": "文本内容\n|列1|列2|\n|---|---|\n|值1|值2|\n更多文本"},
        ]
        chunks = chunk_pages(pages, doc_id="test", chunk_size=500, chunk_overlap=50)
        types = [c.metadata.get("chunk_type") for c in chunks]
        assert "table" in types
        assert "text" in types

    def test_table_not_split(self):
        """表格不应该被切断，即使超过chunk_size"""
        big_table = "|列1|列2|\n|---|---|\n" + "\n".join(f"|行{i}|值{i}|" for i in range(50))
        pages = [{"page_number": 1, "text": big_table}]
        chunks = chunk_pages(pages, doc_id="test", chunk_size=100, chunk_overlap=20)
        # 表格应该是完整的一个chunk
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        assert len(table_chunks) == 1
        assert "|行49|" in table_chunks[0].text

    def test_no_cross_section_chunks(self):
        """chunk不应该跨章节"""
        pages = [
            {"page_number": 1, "text": "## 第一节 A\n内容A。\n\n## 第二节 B\n内容B。"},
        ]
        chunks = chunk_pages(pages, doc_id="test", chunk_size=500, chunk_overlap=50)
        for c in chunks:
            # 每个chunk只属于一个章节
            assert not ("内容A" in c.text and "内容B" in c.text)
