import json
import os
import re

import httpx

from tools.auto_annotation.config import (
    OPENROUTER_URL,
    REQUEST_TIMEOUT,
    CALL_DELAY,
    MAX_RESPONSE_LEN,
    MAX_RANGES,
    MODEL_DEFAULT,
)
from tools.auto_annotation.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def _get_max_line(structured_markdown: str) -> int:
    """Extract the highest line number from structured markdown."""
    max_num = 0
    for match in re.finditer(r"^\s*(\d+)\s*\|", structured_markdown, re.MULTILINE):
        num = int(match.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _extract_json(text: str) -> str:
    """Strip markdown code fences if present, return the inner text."""
    # Match ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _validate_ranges(raw: list, max_line: int) -> list[dict]:
    """Validate and clean parsed ranges."""
    if not isinstance(raw, list):
        raise ValueError("Response is not a JSON array")
    if len(raw) > MAX_RANGES:
        raise ValueError(f"Too many ranges ({len(raw)})")

    ranges = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"Range item is not an object: {item!r}")
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"start/end must be integers: {item!r}")
        if start < 1 or end < 1:
            raise ValueError(f"Line numbers must be positive: {item!r}")
        if start > end:
            raise ValueError(f"start > end: {item!r}")
        # Clamp to actual document range
        if start > max_line:
            continue
        end = min(end, max_line)
        ranges.append({"start": start, "end": end})

    return ranges


def annotate_page(structured_markdown: str) -> list[dict] | None:
    """Send structured markdown to OpenRouter and return content ranges.

    Returns None on transient/API failure. Raises ValueError if
    OPENROUTER_API_KEY is not configured.
    """
    if not structured_markdown:
        return []

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    model = os.environ.get("OPENROUTER_MODEL", MODEL_DEFAULT).strip()
    max_line = _get_max_line(structured_markdown)
    if max_line == 0:
        return []

    user_content = USER_PROMPT_TEMPLATE.format(structured_markdown=structured_markdown)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            OPENROUTER_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(f"  [ERROR] OpenRouter API returned {e.response.status_code}")
        return None
    except httpx.RequestError as e:
        print(f"  [ERROR] OpenRouter request failed: {type(e).__name__}")
        return None

    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        print("  [ERROR] Response body is not valid JSON")
        return None
    choices = body.get("choices", [])
    if not choices:
        print("  [ERROR] No choices in response")
        return None

    content = choices[0].get("message", {}).get("content")
    if not content:
        print("  [ERROR] Empty content in response")
        return None

    if len(content) > MAX_RESPONSE_LEN:
        print("  [ERROR] Response too large")
        return None

    json_text = _extract_json(content)
    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] Invalid JSON: {e}")
        return None

    try:
        return _validate_ranges(raw, max_line)
    except ValueError as e:
        print(f"  [ERROR] Validation: {e}")
        return None
