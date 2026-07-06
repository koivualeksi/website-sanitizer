"""Triple-LLM bulk annotation: auto-validate on 2/3 majority vote.

Runs three models in parallel on each page.
If any two produce identical ranges, the annotation is auto-validated.
Full disagreement is saved as unvalidated for manual review.

Usage:
    python -m tools.auto_annotation.bulk_annotate [--limit N] [--dry-run] [--urgent] [--watch [SECONDS]]
    python -m tools.auto_annotation.bulk_annotate --test [--limit N]

Test mode runs against validated pages to measure agreement rate and
per-model IoU vs ground truth.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import load_dotenv
from psycopg.rows import dict_row

from db.pool import create_pool
from tools.auto_annotation.config import (
    OPENROUTER_URL,
    REQUEST_TIMEOUT,
    CALL_DELAY,
    MAX_RESPONSE_LEN,
    MAX_RANGES,
    MODELS_BULK as MODELS,
)
from tools.auto_annotation.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from scraper.converter import html_to_markdown_light


def _fetch_unannotated(pool, limit, urgent_only=False):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            urgent_clause = "AND p.urgent = TRUE" if urgent_only else ""
            cur.execute(
                f"""SELECT p.id, p.url, p.html
                   FROM pages p
                   LEFT JOIN annotations a ON a.page_id = p.id
                   WHERE p.status = %s
                     AND a.id IS NULL
                     {urgent_clause}
                   ORDER BY p.id
                   LIMIT %s""",
                ("success", limit),
            )
            return cur.fetchall()


def _fetch_validated(pool, limit):
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT p.id, p.url, p.html, a.ranges
                   FROM pages p
                   JOIN annotations a ON a.page_id = p.id
                   WHERE p.status = %s
                     AND a.validated = %s
                   ORDER BY p.id
                   LIMIT %s""",
                ("success", True, limit),
            )
            return cur.fetchall()


def _get_max_line(markdown: str) -> int:
    max_num = 0
    for match in re.finditer(r"^\s*(\d+)\s*\|", markdown, re.MULTILINE):
        num = int(match.group(1))
        if num > max_num:
            max_num = num
    return max_num


def _extract_json(text: str) -> str:
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _call_model(markdown: str, model: str, api_key: str) -> list[dict] | None:
    max_line = _get_max_line(markdown)
    if max_line == 0:
        return []

    user_content = USER_PROMPT_TEMPLATE.format(structured_markdown=markdown)
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
            OPENROUTER_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"[{model}] API error: {e}")
        return None

    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        print(f"[{model}] Invalid JSON response")
        return None

    choices = body.get("choices", [])
    if not choices:
        print(f"[{model}] No choices")
        return None

    content = choices[0].get("message", {}).get("content", "")
    if not content or len(content) > MAX_RESPONSE_LEN:
        print(f"[{model}] Empty or too large")
        return None

    json_text = _extract_json(content)
    try:
        raw = json.loads(json_text)
    except json.JSONDecodeError:
        print(f"[{model}] JSON parse failed")
        return None

    if not isinstance(raw, list):
        print(f"[{model}] Not a list")
        return None

    ranges = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        s, e = item.get("start"), item.get("end")
        if isinstance(s, int) and isinstance(e, int) and 1 <= s <= e <= max_line:
            ranges.append({"start": s, "end": e})

    return ranges


def _ranges_equal(a: list[dict], b: list[dict]) -> bool:
    """Full agreement: identical ranges after sorting."""
    def normalize(ranges):
        return sorted((r["start"], r["end"]) for r in ranges)
    return normalize(a) == normalize(b)


def _ranges_to_lines(ranges: list[dict]) -> set[int]:
    lines = set()
    for r in ranges:
        for n in range(r["start"], r["end"] + 1):
            lines.add(n)
    return lines


def _compute_iou(pred: list[dict], truth: list[dict]) -> float:
    p = _ranges_to_lines(pred)
    t = _ranges_to_lines(truth)
    if not p and not t:
        return 1.0
    inter = len(p & t)
    union = len(p | t)
    return inter / union if union else 0.0


def _call_all(markdown: str, api_key: str) -> list[list[dict] | None]:
    """Call all models in parallel."""
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = [pool.submit(_call_model, markdown, m, api_key) for m in MODELS]
        return [f.result() for f in futures]


def _find_majority(results: list[list[dict] | None]) -> tuple[list[dict] | None, str | None]:
    """Find 2/3 majority. Returns (agreed_ranges, description) or (None, None)."""
    for i in range(len(results)):
        if results[i] is None:
            continue
        for j in range(i + 1, len(results)):
            if results[j] is None:
                continue
            if _ranges_equal(results[i], results[j]):
                labels = [chr(ord("A") + i), chr(ord("A") + j)]
                return results[i], "+".join(labels)
    return None, None


def _lines_to_ranges(lines: set[int]) -> list[dict]:
    """Convert a set of line numbers back to contiguous ranges."""
    if not lines:
        return []
    sorted_lines = sorted(lines)
    ranges = []
    start = sorted_lines[0]
    end = start
    for n in sorted_lines[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append({"start": start, "end": end})
            start = n
            end = n
    ranges.append({"start": start, "end": end})
    return ranges


def _find_closest_pair(results: list[list[dict] | None]) -> tuple[list[dict] | None, str | None]:
    """Find the two most similar results by IoU, return their averaged (union) ranges."""
    valid = [(i, r) for i, r in enumerate(results) if r is not None]
    if len(valid) < 2:
        return None, None
    best_iou = -1.0
    best_i, best_j = 0, 1
    for a in range(len(valid)):
        for b in range(a + 1, len(valid)):
            iou = _compute_iou(valid[a][1], valid[b][1])
            if iou > best_iou:
                best_iou = iou
                best_i, best_j = a, b
    idx_a, res_a = valid[best_i]
    idx_b, res_b = valid[best_j]
    merged = _ranges_to_lines(res_a) | _ranges_to_lines(res_b)
    labels = f"{chr(65+idx_a)}|{chr(65+idx_b)}"
    return _lines_to_ranges(merged), labels


def _save_annotation(pool, page_id, ranges, validated: bool):
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO annotations (page_id, ranges, source, validated, skipped, updated_at)
               VALUES (%s, %s, 'llm', %s, FALSE, now())
               ON CONFLICT (page_id) DO UPDATE SET
                   ranges = EXCLUDED.ranges,
                   source = 'llm',
                   validated = EXCLUDED.validated,
                   skipped = FALSE,
                   updated_at = now()""",
            (page_id, json.dumps(ranges), validated),
        )


def _run_annotate(pool, pages, api_key, dry_run):
    """Annotate unannotated pages with 2/3 majority vote."""
    print(f"Found {len(pages)} unannotated pages")
    print(f"Models: {' + '.join(MODELS)}")
    if dry_run:
        print("--dry-run: not saving")
    print()

    agreed = 0
    disagreed = 0
    failed = 0

    for i, page in enumerate(pages):
        page_id = page["id"]
        url = page["url"][:55]
        markdown = html_to_markdown_light(page["html"], page["url"])

        print(f"[{i+1}/{len(pages)}] Page {page_id}: {url} ... ", end="", flush=True)

        results = _call_all(markdown, api_key)
        ok_count = sum(1 for r in results if r is not None)

        if ok_count < 2:
            which = [chr(ord("A") + j) for j, r in enumerate(results) if r is None]
            print(f"FAILED ({'+'.join(which)})")
            failed += 1
        else:
            majority, who = _find_majority(results)
            if majority is not None:
                if not dry_run:
                    _save_annotation(pool, page_id, majority, validated=True)
                print(f"AGREE {who} ({len(majority)} ranges) -> auto-validated")
                agreed += 1
            else:
                merged, pair = _find_closest_pair(results)
                counts = " ".join(f"{chr(65+j)}={len(r)}" for j, r in enumerate(results) if r is not None)
                if not dry_run:
                    _save_annotation(pool, page_id, merged, validated=False)
                print(f"DISAGREE ({counts}) -> merged {pair} ({len(merged)} ranges) -> manual review")
                disagreed += 1

        if i < len(pages) - 1:
            time.sleep(CALL_DELAY)

    print(f"\nDone: {agreed} agreed, {disagreed} disagreed, {failed} failed")
    print(f"Total: {agreed + disagreed + failed}")


def _run_test(pool, pages, api_key):
    """Test mode: run against validated pages, measure agreement + correctness."""
    print(f"TEST MODE: {len(pages)} validated pages")
    labels = ", ".join(f"{chr(65+i)}={m}" for i, m in enumerate(MODELS))
    print(f"Models: {labels}")
    print()

    agreed = 0
    disagreed = 0
    failed = 0
    ious_per_model = [[] for _ in MODELS]
    ious_agree = []

    for i, page in enumerate(pages):
        page_id = page["id"]
        url = page["url"][:45]
        truth = page["ranges"] or []
        markdown = html_to_markdown_light(page["html"], page["url"])

        print(f"[{i+1}/{len(pages)}] Page {page_id}: {url} ... ", end="", flush=True)

        results = _call_all(markdown, api_key)
        ok_count = sum(1 for r in results if r is not None)

        if ok_count < 2:
            which = [chr(ord("A") + j) for j, r in enumerate(results) if r is None]
            print(f"FAILED ({'+'.join(which)})")
            failed += 1
        else:
            ious = []
            for j, r in enumerate(results):
                if r is not None:
                    iou = _compute_iou(r, truth)
                    ious_per_model[j].append(iou)
                    ious.append((chr(ord("A") + j), iou))
                else:
                    ious.append((chr(ord("A") + j), None))

            majority, who = _find_majority(results)
            iou_strs = " ".join(f"IoU({l})={v:.3f}" for l, v in ious if v is not None)

            if majority is not None:
                agreed += 1
                ious_agree.append(_compute_iou(majority, truth))
                print(f"AGREE({who})  {iou_strs}")
            else:
                disagreed += 1
                counts = " ".join(f"{chr(ord('A')+j)}={len(r)}" for j, r in enumerate(results) if r is not None)
                print(f"DISAGR  {iou_strs}  ({counts})")

        if i < len(pages) - 1:
            time.sleep(CALL_DELAY)

    n = agreed + disagreed
    print(f"\n{'=' * 60}")
    print(f"Results: {agreed} agreed, {disagreed} disagreed, {failed} failed")
    print(f"Agreement rate: {agreed}/{n} = {agreed/n:.1%}" if n else "No results")
    for j, model in enumerate(MODELS):
        if ious_per_model[j]:
            avg = sum(ious_per_model[j]) / len(ious_per_model[j])
            print(f"Mean IoU  {chr(ord('A')+j)} ({model}):  {avg:.3f}")
    if ious_agree:
        print(f"Mean IoU  majority only:  {sum(ious_agree)/len(ious_agree):.3f}")
    print(f"{'=' * 60}")


def main():
    limit = 50
    dry_run = "--dry-run" in sys.argv
    test_mode = "--test" in sys.argv
    urgent_only = "--urgent" in sys.argv
    watch = "--watch" in sys.argv
    watch_interval = 30
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])
    if watch and "--watch" in sys.argv:
        wi = sys.argv.index("--watch")
        if wi + 1 < len(sys.argv) and not sys.argv[wi + 1].startswith("--"):
            watch_interval = int(sys.argv[wi + 1])

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("OPENROUTER_API_KEY not set")
        return

    pool = create_pool()
    try:
        if test_mode:
            pages = _fetch_validated(pool, limit)
            _run_test(pool, pages, api_key)
        elif watch:
            print(f"Watch mode: polling every {watch_interval}s for new pages (limit={limit})\n")
            while True:
                pages = _fetch_unannotated(pool, limit, urgent_only)
                if pages:
                    _run_annotate(pool, pages, api_key, dry_run)
                else:
                    print(f"No unannotated pages found. Waiting {watch_interval}s...")
                print()
                time.sleep(watch_interval)
        else:
            pages = _fetch_unannotated(pool, limit, urgent_only)
            _run_annotate(pool, pages, api_key, dry_run)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
