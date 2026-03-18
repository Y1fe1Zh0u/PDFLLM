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

    # 多轮fallback检索配置
    # 整体开关：关闭后extract_field()退回到原始的"单query全局搜索+单次LLM抽取"行为
    enable_multi_round: bool = True
    # HyDE（Hypothetical Document Embeddings）生成假设文本的温度
    # 设为0.7比抽取的0.1高很多，是为了让LLM生成更多样化的假设段落以提升检索召回率
    hyde_temperature: float = 0.7
    # HyDE假设答案的最大token数，100-200字中文约对应150-250 token
    hyde_max_tokens: int = 256

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
