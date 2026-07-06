"""Shared OpenRouter configuration for annotation tools."""

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = httpx.Timeout(10.0, read=120.0)
CALL_DELAY = 1.0
MAX_RESPONSE_LEN = 50_000
MAX_RANGES = 500

MODEL_DEFAULT = "deepseek/deepseek-v4-flash"
MODELS_BULK = [
    "google/gemini-2.5-flash",
    "deepseek/deepseek-v4-flash",
    "openai/gpt-4o-mini",
]
