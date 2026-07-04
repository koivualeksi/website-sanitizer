"""Export annotated pages as train/test JSONL.

Each row is a chat completion example:
  {"messages": [system, user, assistant], "page_id": ..., "url": ...}

Uses a fixed split manifest for reproducibility. Generates a fresh
80/20 split if no manifest exists.

Run: python -m tools.export_data
"""

import json
import os
import random

from dotenv import load_dotenv
from psycopg.rows import dict_row

from db.pool import create_pool
from tools.auto_annotation.prompt import USER_PROMPT_TEMPLATE_FINETUNED
from scraper.converter import html_to_markdown_light


SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SYSTEM_PROMPT = "Return content line ranges as start:end pairs."


def _fetch_annotated(pool) -> list[dict]:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT p.id, p.url, p.html, p.markdown, a.ranges
                   FROM pages p
                   JOIN annotations a ON a.page_id = p.id
                   WHERE p.status = %s
                     AND a.validated = %s
                     AND COALESCE(p.dataset, 'original') = 'original'
                   ORDER BY p.id""",
                ("success", True),
            )
            return cur.fetchall()


def _ranges_to_compact(ranges: list[dict]) -> str:
    """Convert [{"start":10,"end":20},...] to '10:20,...' format."""
    if not ranges:
        return "1:1"  # fallback for empty ranges
    parts = [f"{r['start']}:{r['end']}" for r in ranges]
    return ",".join(parts)


def _to_chat_row(page: dict) -> dict:
    light_md = html_to_markdown_light(page["html"], page["url"])
    user_content = USER_PROMPT_TEMPLATE_FINETUNED.format(structured_markdown=light_md)
    assistant_content = _ranges_to_compact(page["ranges"] or [])
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "page_id": page["id"],
        "url": page["url"],
    }


def _write_split(path, pages, indices):
    with open(path, "w", encoding="utf-8") as f:
        for i in indices:
            row = _to_chat_row(pages[i])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {path} ({len(indices)} rows)")


def main():
    load_dotenv()
    pool = create_pool()

    try:
        pages = _fetch_annotated(pool)
    finally:
        pool.close()

    print(f"Found {len(pages)} annotated pages")

    manifest_path = os.path.join(OUTPUT_DIR, "split_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            old_manifest = json.load(f)
        test_ids = set(old_manifest.get("test_page_ids", []))
        print(f"Loaded existing manifest: {len(test_ids)} test pages")
    else:
        all_ids = [p["id"] for p in pages]
        rng = random.Random(SEED)
        rng.shuffle(all_ids)
        split_idx = int(len(all_ids) * 0.8)
        test_ids = set(all_ids[split_idx:])
        print(f"Generated fresh 80/20 split: {len(test_ids)} test pages")

    test_idx = sorted(i for i, p in enumerate(pages) if p["id"] in test_ids)
    train_idx = sorted(i for i, p in enumerate(pages) if p["id"] not in test_ids)

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _write_split(os.path.join(OUTPUT_DIR, "train.jsonl"), pages, train_idx)
    _write_split(os.path.join(OUTPUT_DIR, "test.jsonl"), pages, test_idx)

    manifest = {
        "seed": SEED,
        "total": len(pages),
        "train_page_ids": [pages[i]["id"] for i in train_idx],
        "test_page_ids": [pages[i]["id"] for i in test_idx],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
