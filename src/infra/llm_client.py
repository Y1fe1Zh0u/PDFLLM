"""LLM客户端

封装OpenAI SDK，兼容DeepSeek/OpenRouter API。
"""

import json
import time

from openai import OpenAI

from src.infra.config import settings
from src.infra.logger import get_logger

logger = get_logger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """获取OpenAI客户端（懒加载单例）"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    return _client


def chat_json(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> dict | None:
    """调用LLM并返回JSON对象

    内置重试机制。返回None表示所有重试均失败。

    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        model: 模型名，默认从config读取
        temperature: 温度参数
        max_tokens: 最大token数

    Returns:
        解析后的JSON字典，失败返回None
    """
    client = get_client()
    model = model or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature
    max_tokens = max_tokens or settings.llm_max_tokens

    for attempt in range(settings.llm_max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = json.loads(content)
            logger.debug(f"LLM返回成功 (attempt {attempt + 1})")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"LLM返回非法JSON (attempt {attempt + 1}): {e}")
        except Exception as e:
            logger.warning(f"LLM调用失败 (attempt {attempt + 1}): {e}")

        if attempt < settings.llm_max_retries - 1:
            wait = 2 ** attempt
            logger.info(f"等待 {wait}s 后重试...")
            time.sleep(wait)

    logger.error("LLM调用全部重试失败")
    return None


def chat_raw(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """调用LLM返回原始文本（不要求JSON格式）"""
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        return ""
