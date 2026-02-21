"""配置管理模块

基于Pydantic Settings，从.env文件和环境变量加载配置。
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """系统配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM API配置
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_max_retries: int = 3

    # Embedding模型（本地）
    embedding_model: str = "BAAI/bge-large-zh-v1.5"

    # 路径配置
    upload_dir: str = "./data/uploads"
    output_dir: str = "./data/outputs"
    index_dir: str = "./data/index"
    db_path: str = "./data/outputs/facts.db"

    # 切片配置
    chunk_size: int = 512
    chunk_overlap: int = 64

    # 检索配置
    top_k: int = 8

    # 日志
    log_level: str = "INFO"
    log_dir: str = "./logs"

    @property
    def base_dir(self) -> Path:
        """项目根目录"""
        return Path(__file__).parent.parent.parent

    def ensure_dirs(self):
        """确保必要的目录存在"""
        for d in [self.upload_dir, self.output_dir, self.index_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


# 全局配置单例
settings = Settings()
