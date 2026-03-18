# PDFLLM

A股并购报告结构化事实抽取系统。从上市公司并购重组报告PDF中自动提取交易概要、并购目的等关键事实，输出结构化数据档案。

## 系统架构

```
PDF文件
  │
  ▼
文本提取 (pymupdf4llm)          ← 提取Markdown格式文本+表格
  │
  ▼
章节识别 + 语义切片              ← 按标题层级拆分，固定窗口+重叠切片
  │
  ▼
向量索引 (bge-large-zh + FAISS) ← 中文语义embedding，内积相似度检索
  │
  ▼
多轮fallback检索与LLM抽取       ← 子组检索 → HyDE → 章节路由，逐轮补全
  │
  ▼
事实档案 (SQLite)               ← Pydantic校验后持久化存储
```

## 核心特性

- **多轮fallback检索**：分3轮逐步精确定位信息
  - Round 1：将查询拆为子字段组（交易方/对价/估值/承诺）分别检索，合并去重
  - Round 2：对缺失字段用HyDE（假设文档嵌入）生成语义丰富的查询向量
  - Round 3：按章节关键词路由，在目标章节内精确检索
- **断点续跑**：已处理的文档自动跳过，支持批量处理中断后恢复
- **结果溯源**：每个抽取字段附带原文引用（`_quote`字段），可追溯至具体chunk和页码

## 抽取字段

| 字段组 | 抽取内容 |
|--------|----------|
| 交易概要 | 收购方、标的方、交易类型、交易金额、发行价格、支付方式、标的估值、估值方法、业绩承诺 |
| 并购目的 | 战略目的、协同效应、行业逻辑、一句话总结 |

## 项目结构

```
src/
├── infra/           # 基础设施：配置、数据模型、日志、数据库、embedding、LLM客户端
├── ingestion/       # PDF提取 + 语义切片
├── indexing/        # FAISS向量索引创建与检索
├── extraction/      # LLM事实抽取（多轮fallback）
└── pipeline.py      # 端到端编排
scripts/
└── run.py           # CLI入口
tests/               # pytest测试（mock外部依赖）
```

## 快速开始

### 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 配置

创建 `.env` 文件：

```env
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

### 使用

```bash
# 处理单个PDF
python scripts/run.py data/uploads/report.pdf

# 批量处理目录下所有PDF
python scripts/run.py data/uploads/

# 不跳过已处理文档（重新处理全部）
python scripts/run.py data/uploads/ --no-resume

# 列出已处理的文档
python scripts/run.py --list

# 查看指定文档的事实档案
python scripts/run.py --show <doc_id>
```

### 输入文件命名规范

PDF文件名需包含股票代码和公司名，格式示例：

```
000022深赤湾Ａ发行股份购买资产并募集配套资金暨关联交易报告书（修订稿）.pdf
```

系统从文件名中自动提取股票代码（前6位数字）和公司名称。

## 技术栈

| 组件 | 选型 |
|------|------|
| PDF解析 | pymupdf4llm + pymupdf |
| Embedding | sentence-transformers + BAAI/bge-large-zh-v1.5 |
| 向量检索 | faiss-cpu (IndexFlatIP) |
| LLM | OpenAI SDK（兼容DeepSeek/OpenRouter） |
| 数据校验 | Pydantic v2 |
| 数据库 | SQLite（标准库） |
| 配置管理 | pydantic-settings + .env |

## 测试

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
