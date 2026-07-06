"""Compare annotation accuracy: trafilatura vs Qwen bare vs Qwen fine-tuned.

Runs all three approaches on the held-out test set and prints a comparison.
Reads test data from data/test.jsonl (produced by scripts/export_training_data.py).

Run: python tests/test_comparison.py
     python tests/test_comparison.py --methods trafilatura,qwen-bare
     python tests/test_comparison.py --methods qwen-finetuned
"""

import argparse
import json
import os
import re
import time

import httpx
import trafilatura
from dotenv import load_dotenv

from tools.auto_annotation.prompt import (
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_FINETUNED, USER_PROMPT_TEMPLATE_FINETUNED,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REQUEST_TIMEOUT = httpx.Timeout(10.0, read=120.0)
CALL_DELAY = 2.0

ALL_METHODS = ["gemini-flash", "trafilatura", "qwen-bare", "qwen-finetuned"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_markdown_lines(markdown: str) -> dict[int, str]:
    lines = {}
    for raw_line in markdown.split("\n"):
        m = re.match(r"^\s*(\d+)\s*\|\s*(.*)", raw_line)
        if m:
            num = int(m.group(1))
            text = m.group(2).strip()
            if text:
                lines[num] = text
    return lines


def _extract_keywords(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9\u00c0-\u024f]{3,}", text)}


def _ranges_to_line_set(ranges: list[dict]) -> set[int]:
    lines = set()
    for r in ranges:
        for n in range(r["start"], r["end"] + 1):
            lines.add(n)
    return lines


def _compute_metrics(predicted: set[int], truth: set[int]) -> dict:
    if not predicted and not truth:
        return {"precision": 1.0, "recall": 1.0, "iou": 1.0}
    intersection = len(predicted & truth)
    union = len(predicted | truth)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(truth) if truth else 0.0
    iou = intersection / union if union else 0.0
    return {"precision": precision, "recall": recall, "iou": iou}


def _extract_json(text: str) -> str:
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _get_max_line(markdown: str) -> int:
    max_num = 0
    for match in re.finditer(r"^\s*(\d+)\s*\|", markdown, re.MULTILINE):
        num = int(match.group(1))
        if num > max_num:
            max_num = num
    return max_num


# ---------------------------------------------------------------------------
# Trafilatura
# ---------------------------------------------------------------------------

def predict_trafilatura(html: str, markdown: str) -> set[int] | None:
    extracted = trafilatura.extract(
        html,
        favor_recall=True,
        include_tables=True,
        include_formatting=True,
        include_comments=False,
    )
    if extracted is None:
        return None
    md_lines = _parse_markdown_lines(markdown)
    extracted_kw = _extract_keywords(extracted)
    predicted = set()
    for num, text in md_lines.items():
        line_kw = _extract_keywords(text)
        if not line_kw:
            continue
        overlap = len(line_kw & extracted_kw) / len(line_kw)
        if overlap >= 0.5:
            predicted.add(num)
    return predicted


# ---------------------------------------------------------------------------
# Qwen via OpenRouter (bare-bones prompt)
# ---------------------------------------------------------------------------

def predict_gemini_flash(markdown: str, api_key: str) -> set[int] | None:
    model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash")
    return _call_openrouter(markdown, model, api_key, SYSTEM_PROMPT)


def predict_qwen_bare(markdown: str, api_key: str) -> set[int] | None:
    model = os.environ.get("QWEN_MODEL", "qwen/qwen3-4b")
    return _call_openrouter(markdown, model, api_key, SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Qwen fine-tuned (separate endpoint or model ID)
# ---------------------------------------------------------------------------

def predict_qwen_finetuned(markdown: str, api_key: str) -> set[int] | None:
    model = os.environ.get("QWEN_FINETUNED_MODEL", "")
    if not model:
        print("  [SKIP] QWEN_FINETUNED_MODEL not set")
        return None
    endpoint = os.environ.get("QWEN_FINETUNED_ENDPOINT", OPENROUTER_URL)
    finetuned_api_key = os.environ.get("QWEN_FINETUNED_API_KEY", api_key)
    return _call_openrouter(
        markdown, model, finetuned_api_key,
        SYSTEM_PROMPT_FINETUNED, endpoint,
        user_template=USER_PROMPT_TEMPLATE_FINETUNED,
    )


def _call_openrouter(
    markdown: str,
    model: str,
    api_key: str,
    system_prompt: str,
    endpoint: str = OPENROUTER_URL,
    user_template: str = USER_PROMPT_TEMPLATE,
) -> set[int] | None:
    max_line = _get_max_line(markdown)
    if max_line == 0:
        return set()

    user_content = user_template.format(structured_markdown=markdown)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(endpoint, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"  [ERROR] API: {e}")
        return None

    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        print("  [ERROR] Invalid JSON response")
        return None

    choices = body.get("choices", [])
    if not choices:
        print("  [ERROR] No choices")
        return None

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        print("  [ERROR] Empty content")
        return None

    json_text = _extract_json(content)
    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON parse: {e}")
        return None

    if not isinstance(raw, list):
        print(f"  [ERROR] Not a list")
        return None

    ranges = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        s, e = item.get("start"), item.get("end")
        if isinstance(s, int) and isinstance(e, int) and 1 <= s <= e <= max_line:
            ranges.append({"start": s, "end": e})

    return _ranges_to_line_set(ranges)


# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------

def load_test_data() -> list[dict]:
    test_path = os.path.join(DATA_DIR, "test.jsonl")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found. Run: python -m tools.data_export.export_grpo")
    rows = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assistant_msg = row["messages"][2]["content"]
            ground_truth = json.loads(assistant_msg)
            markdown = row["messages"][1]["content"]
            rows.append({
                "page_id": row.get("page_id"),
                "url": row.get("url", ""),
                "html": row.get("html", ""),
                "markdown": markdown,
                "truth_ranges": ground_truth,
            })
    return rows


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation(methods: list[str]):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    test_data = load_test_data()
    print(f"\nEvaluating {len(test_data)} test pages\n")
    print(f"Methods: {', '.join(methods)}\n")

    # results[method] = list of per-page dicts
    results: dict[str, list[dict]] = {m: [] for m in methods}

    for i, page in enumerate(test_data):
        page_id = page["page_id"]
        url = page["url"][:40]
        md_lines = _parse_markdown_lines(page["markdown"])
        truth_lines = _ranges_to_line_set(page["truth_ranges"])
        truth_lines &= set(md_lines.keys())

        print(f"  [{i+1}/{len(test_data)}] Page {page_id}: {url}")

        for method in methods:
            if method == "gemini-flash":
                pred = predict_gemini_flash(page["markdown"], api_key)
            elif method == "trafilatura":
                pred = predict_trafilatura(page["html"], page["markdown"])
            elif method == "qwen-bare":
                pred = predict_qwen_bare(page["markdown"], api_key)
            elif method == "qwen-finetuned":
                pred = predict_qwen_finetuned(page["markdown"], api_key)
            else:
                continue

            if pred is None:
                results[method].append({"page_id": page_id, "metrics": None})
            else:
                metrics = _compute_metrics(pred, truth_lines)
                results[method].append({"page_id": page_id, "metrics": metrics})

        # Rate limit for API methods
        if any(m.startswith("qwen") or m == "gemini-flash" for m in methods) and i < len(test_data) - 1:
            time.sleep(CALL_DELAY)

    _print_comparison(results, test_data)


def _print_comparison(results: dict[str, list[dict]], test_data: list[dict]):
    methods = list(results.keys())

    # Header
    print("\n" + "=" * (30 + 22 * len(methods)))
    header = f"{'ID':>5}  {'URL':<20}"
    for m in methods:
        header += f"  {'IoU':>6} {'P':>5} {'R':>5}"
    print(header)
    print("-" * (30 + 22 * len(methods)))

    # Per-page rows
    for i, page in enumerate(test_data):
        page_id = page["page_id"]
        url = page["url"][:20]
        row = f"{page_id:>5}  {url:<20}"
        for m in methods:
            r = results[m][i]
            if r["metrics"] is None:
                row += f"  {'FAIL':>6} {'':>5} {'':>5}"
            else:
                met = r["metrics"]
                row += f"  {met['iou']:>6.3f} {met['precision']:>5.3f} {met['recall']:>5.3f}"
        print(row)

    print("-" * (30 + 22 * len(methods)))

    # Means
    mean_row = f"{'MEAN':>5}  {'':20}"
    for m in methods:
        successful = [r for r in results[m] if r["metrics"] is not None]
        if successful:
            n = len(successful)
            mi = sum(r["metrics"]["iou"] for r in successful) / n
            mp = sum(r["metrics"]["precision"] for r in successful) / n
            mr = sum(r["metrics"]["recall"] for r in successful) / n
            mean_row += f"  {mi:>6.3f} {mp:>5.3f} {mr:>5.3f}"
        else:
            mean_row += f"  {'N/A':>6} {'':>5} {'':>5}"
    print(mean_row)

    # Count row
    count_row = f"{'n':>5}  {'':20}"
    for m in methods:
        successful = [r for r in results[m] if r["metrics"] is not None]
        count_row += f"  {len(successful):>6} {'':>5} {'':>5}"
    print(count_row)

    print("=" * (30 + 22 * len(methods)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(ALL_METHODS),
                        help="Comma-separated methods to evaluate")
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]
    run_evaluation(methods)
