"""测试 PDF 抽取功能"""
import pytest
from pathlib import Path


def test_imports():
    """测试模块导入"""
    from src.ingestion.pdf_extractor import PDFExtractor
    from src.utils.config import settings
    assert PDFExtractor is not None
    assert settings is not None
