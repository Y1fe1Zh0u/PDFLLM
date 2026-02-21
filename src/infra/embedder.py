"""Embedding模块

加载bge-large-zh模型，提供文本向量化接口。
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.infra.config import settings
from src.infra.logger import get_logger

logger = get_logger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """获取embedding模型（懒加载单例）"""
    global _model
    if _model is None:
        logger.info(f"加载embedding模型: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding模型加载完成")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """批量向量化文本

    Args:
        texts: 文本列表

    Returns:
        numpy数组，shape = (len(texts), embedding_dim)
    """
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 100, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    """单条文本向量化"""
    return embed_texts([text])[0]


def get_embedding_dim() -> int:
    """获取embedding维度"""
    return get_model().get_sentence_embedding_dimension()
