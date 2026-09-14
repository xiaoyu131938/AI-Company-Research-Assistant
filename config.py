"""Application configuration for the OpenAI-compatible LLM client."""

import os


API_URL = os.getenv(
    "LLM_API_URL",
    "https://open.bigmodel.cn/api/paas/v4/",
).strip()
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "glm-4-flash").strip()
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 2_500
LLM_TIMEOUT_SECONDS = 120
LLM_MAX_SOURCE_CHARS = 30_000
MIN_SOURCE_CONTENT_CHARS = 200

PROHIBITED_WORDS = tuple(
    word.strip()
    for word in os.getenv("PROHIBITED_WORDS", "").split(",")
    if word.strip()
)


def get_api_key() -> str:
    """Read the API key at call time so local environment changes are respected."""
    return os.getenv("ZHIPU_API_KEY", "").strip()
