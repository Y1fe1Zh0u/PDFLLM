"""PDF文本和表格提取

使用pymupdf4llm将PDF转为Markdown格式，保留表格结构。
"""

import re
from pathlib import Path

import pymupdf4llm

from src.infra.logger import get_logger

logger = get_logger(__name__)


def extract_doc_id(pdf_path: str | Path) -> str:
    """从文件路径生成doc_id（文件名去后缀）"""
    return Path(pdf_path).stem


def extract_metadata_from_filename(filename: str) -> dict:
    """从文件名中提取股票代码和公司名

    常见格式：000022深赤湾Ａ_xxx.pdf, 000035_中国天楹_xxx.pdf
    """
    stem = Path(filename).stem
    # 匹配6位数字（股票代码）
    code_match = re.search(r"(\d{6})", stem)
    stock_code = code_match.group(1) if code_match else ""

    # 提取公司名：代码后面的中文字符，遇到报告相关动词截断
    name_match = re.search(r"\d{6}[_\-]?([\u4e00-\u9fffＡ-Ｚａ-ｚ]+?(?=发行|向|资产|关于|重大|收购|吸收|合并|出售))", stem)
    if not name_match:
        # fallback：取前2-6个中文字符
        name_match = re.search(r"\d{6}[_\-]?([\u4e00-\u9fffＡ-Ｚａ-ｚ]{2,6})", stem)
    company_name = name_match.group(1) if name_match else ""

    return {"stock_code": stock_code, "company_name": company_name}


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """从PDF提取按页分组的Markdown文本

    Args:
        pdf_path: PDF文件路径

    Returns:
        按页分组的字典列表，每项包含 page_number 和 text
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    logger.info(f"开始提取PDF: {pdf_path.name}")

    # pymupdf4llm按页提取，自动将表格转为Markdown
    page_chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
    )

    pages = []
    for chunk in page_chunks:
        page_num = chunk["metadata"]["page"]  # 从0开始
        text = chunk["text"].strip()
        if text:
            pages.append({
                "page_number": page_num + 1,  # 转为从1开始
                "text": text,
            })

    logger.info(f"提取完成: {pdf_path.name}, 共{len(pages)}页有效内容")
    return pages
