"""章节识别 + 语义切片

策略：
1. 先按章节标题切分，保持章节完整性
2. 章节内部，将表格单独成chunk，文本按自然段合并
3. 长段落用滑动窗口切片，但不跨章节
4. 每个chunk携带完整元数据：section、page、chunk_type（text/table）
"""

import re

from src.infra.config import settings
from src.infra.logger import get_logger
from src.infra.models import Chunk

logger = get_logger(__name__)

# 章节标题模式（按优先级排列）
SECTION_PATTERNS = [
    # "# 第一节 交易概述" / "第一章 xxx"
    re.compile(r"^#{0,3}\s*第[一二三四五六七八九十百\d]+[节章]\s+.+", re.MULTILINE),
    # "一、交易概述"
    re.compile(r"^#{0,3}\s*[一二三四五六七八九十]+[、．.]\s*.+", re.MULTILINE),
    # Markdown标题 "## 重大事项提示"
    re.compile(r"^#{1,3}\s+.{2,30}\s*$", re.MULTILINE),
]

# 表格行模式：以|开头的行
TABLE_LINE_RE = re.compile(r"^\|.*\|", re.MULTILINE)
# 表格分隔行：|---|---|
TABLE_SEP_RE = re.compile(r"^\|[-:\s|]+\|$", re.MULTILINE)


def _find_sections(text: str) -> list[tuple[int, str]]:
    """找到所有章节标题及其位置

    Returns:
        [(position, title), ...] 按位置排序
    """
    hits: list[tuple[int, str]] = []
    for pattern in SECTION_PATTERNS:
        for m in pattern.finditer(text):
            title = m.group().strip().lstrip("#").strip()
            if 2 <= len(title) <= 50:
                hits.append((m.start(), title))

    # 去重：同一位置只保留一个
    seen_pos = set()
    unique = []
    for pos, title in sorted(hits):
        # 允许位置相近（5字符内）的合并
        if any(abs(pos - p) < 5 for p in seen_pos):
            continue
        seen_pos.add(pos)
        unique.append((pos, title))

    return unique


def _split_by_sections(full_text: str) -> list[dict]:
    """按章节标题切分文本

    Returns:
        [{"title": "第一节 交易概述", "text": "...", "start": 0}, ...]
    """
    sections = _find_sections(full_text)

    if not sections:
        return [{"title": "", "text": full_text, "start": 0}]

    result = []

    # 标题之前的内容（前言/封面）
    if sections[0][0] > 0:
        pre_text = full_text[:sections[0][0]].strip()
        if pre_text:
            result.append({"title": "", "text": pre_text, "start": 0})

    # 每个章节
    for i, (pos, title) in enumerate(sections):
        end = sections[i + 1][0] if i + 1 < len(sections) else len(full_text)
        text = full_text[pos:end].strip()
        if text:
            result.append({"title": title, "text": text, "start": pos})

    return result


def _is_table_block(block: str) -> bool:
    """判断一个文本块是否是表格"""
    lines = block.strip().split("\n")
    if len(lines) < 2:
        return False
    table_lines = sum(1 for l in lines if l.strip().startswith("|"))
    return table_lines >= len(lines) * 0.6


def _split_text_and_tables(section_text: str) -> list[dict]:
    """将章节内容分离为文本块和表格块

    Returns:
        [{"type": "text"|"table", "content": "..."}, ...]
    """
    lines = section_text.split("\n")
    blocks: list[dict] = []
    current_type = "text"
    current_lines: list[str] = []

    for line in lines:
        is_table_line = line.strip().startswith("|")

        if is_table_line and current_type == "text":
            # 切换到表格：先保存之前的文本
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    blocks.append({"type": "text", "content": text})
            current_lines = [line]
            current_type = "table"
        elif not is_table_line and current_type == "table":
            # 切换回文本：先保存之前的表格
            if current_lines:
                table = "\n".join(current_lines).strip()
                if table:
                    blocks.append({"type": "table", "content": table})
            current_lines = [line]
            current_type = "text"
        else:
            current_lines.append(line)

    # 保存最后一块
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            blocks.append({"type": current_type, "content": content})

    return blocks


def _chunk_text_block(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """将文本块按自然段合并，超长则滑动窗口切片

    优先按段落边界切分，避免切断句子。
    """
    if len(text) <= chunk_size:
        return [text]

    # 先按段落切分
    paragraphs = re.split(r"\n\s*\n", text)

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = current + "\n\n" + para if current else para
        else:
            # 当前段落放不下了
            if current:
                chunks.append(current.strip())

            if len(para) <= chunk_size:
                current = para
            else:
                # 超长段落：滑动窗口
                step = chunk_size - chunk_overlap
                pos = 0
                while pos < len(para):
                    end = min(pos + chunk_size, len(para))
                    chunks.append(para[pos:end].strip())
                    pos += step
                current = ""

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _get_page_for_position(
    position: int,
    page_boundaries: list[tuple[int, int, int]],
) -> int:
    """根据字符位置确定页码"""
    for start, end, page_num in page_boundaries:
        if position < end:
            return page_num
    return page_boundaries[-1][2] if page_boundaries else 1


def chunk_pages(
    pages: list[dict],
    doc_id: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """将按页文本切分为带元数据的chunks

    策略：
    1. 拼接全文，按章节标题切分
    2. 每个章节内部，分离文本和表格
    3. 表格单独成chunk（不切断）
    4. 文本按自然段合并，超长则滑动窗口

    Args:
        pages: extract_pages()的输出
        doc_id: 文档标识
        chunk_size: 切片大小（字符数），默认从config读取
        chunk_overlap: 切片重叠（字符数），默认从config读取

    Returns:
        Chunk列表
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not pages:
        return []

    # 构建全文和页码映射
    full_text = ""
    page_boundaries: list[tuple[int, int, int]] = []

    for page in pages:
        start = len(full_text)
        full_text += page["text"] + "\n\n"
        end = len(full_text)
        page_boundaries.append((start, end, page["page_number"]))

    # 1. 按章节切分
    sections = _split_by_sections(full_text)
    logger.debug(f"[{doc_id}] 识别到 {len(sections)} 个章节")

    chunks: list[Chunk] = []
    chunk_id = 0

    for section in sections:
        section_title = section["title"]
        section_start = section["start"]

        # 2. 分离文本和表格
        blocks = _split_text_and_tables(section["text"])

        # 累计偏移量，用于定位页码
        offset = section_start

        for block in blocks:
            block_type = block["type"]
            content = block["content"]

            if not content.strip():
                continue

            page_num = _get_page_for_position(offset, page_boundaries)

            if block_type == "table":
                # 表格单独成chunk，不切断
                # 超长表格也保持完整（表格切断后无意义）
                chunks.append(Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=content,
                    page=page_num,
                    section=section_title,
                    metadata={"chunk_type": "table"},
                ))
                chunk_id += 1
            else:
                # 文本按段落合并+滑动窗口
                text_chunks = _chunk_text_block(content, chunk_size, chunk_overlap)
                for tc in text_chunks:
                    if tc.strip():
                        chunks.append(Chunk(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            text=tc,
                            page=page_num,
                            section=section_title,
                            metadata={"chunk_type": "text"},
                        ))
                        chunk_id += 1

            # 更新偏移量
            offset += len(content) + 2  # +2 for \n\n

    logger.info(
        f"文档 {doc_id} 切片完成: {len(chunks)} chunks "
        f"(text: {sum(1 for c in chunks if c.metadata.get('chunk_type')=='text')}, "
        f"table: {sum(1 for c in chunks if c.metadata.get('chunk_type')=='table')})"
    )
    return chunks
