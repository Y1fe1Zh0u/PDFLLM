# PDFLLM - A股并购报告结构化事实抽取系统

## 项目目标

读取A股并购报告PDF → 提取结构化事实（交易概要+并购目的） → 输出可查看的事实档案

## 架构

```
PDF → 文本/表格提取(pymupdf4llm) → 规则提取元数据 → 语义切片
  → Embedding索引(bge-large-zh + FAISS) → 按事实字段检索
  → LLM结构化抽取(DeepSeek) → 事实档案(SQLite)
```

## 模块职责

- `src/infra/` — 基础设施：配置、数据模型、日志、数据库、embedding、LLM客户端
- `src/ingestion/` — PDF提取+切片：pymupdf4llm提取文本表格，章节识别+语义切片
- `src/indexing/` — 向量索引：FAISS索引创建/查询
- `src/extraction/` — LLM事实抽取：检索相关chunks→拼prompt→调LLM→结构化输出
- `src/pipeline.py` — 端到端编排，含断点续跑
- `scripts/run.py` — CLI入口

## 技术选型

- Python 3.13, pip + pyproject.toml
- PDF: pymupdf4llm + pymupdf
- Embedding: sentence-transformers + bge-large-zh-v1.5
- 向量库: faiss-cpu
- LLM: openai SDK (兼容DeepSeek/OpenRouter)
- 数据校验: pydantic + pydantic-settings
- 数据库: sqlite3 (标准库)

## 编码规范

- 中文注释和docstring
- 所有数据结构用Pydantic模型定义
- 用logger替代print
- import路径从项目根开始：`from src.infra.config import settings`
- 测试用pytest，mock外部依赖（LLM、embedding模型）
