"""PDF 文本和表格提取器"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pdfplumber
import camelot
import pandas as pd

logger = logging.getLogger(__name__)


class PDFExtractor:
    """PDF 文档抽取器"""

    def __init__(self, pdf_path: str):
        """初始化

        Args:
            pdf_path: PDF 文件路径
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    def extract_text(self) -> List[Dict[str, Any]]:
        """提取文本内容

        Returns:
            文本块列表，每个块包含 page, text, bbox 等信息
        """
        text_chunks = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    text_chunks.append({
                        "page": page_num,
                        "text": text.strip(),
                        "type": "text"
                    })

        logger.info(f"从 {self.pdf_path.name} 提取了 {len(text_chunks)} 个文本块")
        return text_chunks

    def extract_tables(self, flavor: str = "lattice") -> List[Dict[str, Any]]:
        """提取表格

        Args:
            flavor: 表格提取模式，'lattice' 或 'stream'

        Returns:
            表格列表，每个表格包含 page, dataframe, title 等信息
        """
        tables_data = []

        try:
            # 使用 camelot 提取表格
            tables = camelot.read_pdf(
                str(self.pdf_path),
                pages="all",
                flavor=flavor
            )

            for i, table in enumerate(tables):
                df = table.df
                tables_data.append({
                    "table_id": f"table_{i+1}",
                    "page": table.page,
                    "dataframe": df,
                    "accuracy": table.accuracy,
                    "type": "table"
                })

            logger.info(f"从 {self.pdf_path.name} 提取了 {len(tables_data)} 个表格")

        except Exception as e:
            logger.error(f"表格提取失败: {e}")

        return tables_data

    def extract_all(self) -> Dict[str, Any]:
        """提取所有内容

        Returns:
            包含文本和表格的字典
        """
        return {
            "document_id": self.pdf_path.stem,
            "source_path": str(self.pdf_path),
            "text_chunks": self.extract_text(),
            "tables": self.extract_tables(),
        }


def extract_pdf(pdf_path: str) -> Dict[str, Any]:
    """便捷函数：提取 PDF 内容

    Args:
        pdf_path: PDF 文件路径

    Returns:
        提取的结构化数据
    """
    extractor = PDFExtractor(pdf_path)
    return extractor.extract_all()
