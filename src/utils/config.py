"""配置管理模块"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """系统配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # OpenAI 配置
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"

    # 模型配置
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    max_tokens: int = 4096

    # 向量数据库
    vector_db_type: str = "faiss"
    faiss_index_path: str = "./data/faiss_index"

    # 文件路径
    upload_dir: str = "./data/uploads"
    output_dir: str = "./data/outputs"

    # 切片配置
    chunk_size: int = 400
    chunk_overlap: int = 80
    top_k: int = 5

    # API 配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # PDF提取配置
    camelot_flavor: str = "lattice"  # 表格提取模式：lattice 或 stream
    camelot_pages: str = "all"  # 提取的页码范围
    max_workers: int = 4  # 并发提取的最大worker数

    # 表格合并配置
    table_merge_enabled: bool = True  # 是否启用跨页表格合并
    table_merge_similarity_threshold: float = 0.7  # 表头相似度阈值
    table_merge_dry_run: bool = False  # 试运行模式（不实际合并）

    # 表格导出配置
    export_format: str = "csv"  # 导出格式：csv 或 excel
    export_clean_newlines: bool = True  # 是否清理单元格中的换行符
    export_with_title: bool = False  # 是否使用表格标题命名（实验性功能）

    @property
    def base_dir(self) -> Path:
        """项目根目录"""
        return Path(__file__).parent.parent.parent

    def ensure_dirs(self):
        """确保必要的目录存在"""
        dirs = [
            self.upload_dir,
            self.output_dir,
            self.faiss_index_path,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()
