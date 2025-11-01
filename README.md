# 财务报表 RAG 系统

基于 RAG（检索增强生成）技术的财务报表智能分析系统。

## 项目架构

```
PDFLLM/
├── src/                    # 源代码
│   ├── ingestion/         # PDF 抽取模块
│   ├── indexing/          # 向量索引模块
│   ├── retrieval/         # 检索模块
│   ├── llm/               # LLM 分析模块
│   ├── api/               # API 服务
│   └── utils/             # 工具函数
├── data/                   # 数据目录
│   ├── uploads/           # 上传的 PDF
│   ├── outputs/           # 输出结果
│   └── faiss_index/       # 向量索引
├── tests/                  # 测试
├── notebooks/              # Jupyter notebooks
├── scripts/                # 脚本工具
├── docs/                   # 文档
└── config/                 # 配置文件
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API keys
```

### 3. 运行示例

```python
from src.ingestion.pdf_extractor import extract_pdf

# 提取 PDF
result = extract_pdf("path/to/financial_report.pdf")
print(result)
```

## 功能模块

- **Ingestion 层**: PDF 文本和表格提取
- **Indexing 层**: 文本切片和向量化
- **Retrieval 层**: 语义检索和重排序
- **LLM 层**: 智能分析和问答

## 技术栈

- **PDF 处理**: pdfplumber, camelot-py
- **向量化**: sentence-transformers, FAISS
- **LLM**: OpenAI API, LangChain
- **Web 框架**: FastAPI
- **数据处理**: pandas, numpy

## 开发进度

详见 [docs/architecture.md](docs/architecture.md)

## License

MIT
