"""Export silver-tier annotations as train JSONL for SFT.

All silver pages (single-opinion, unvalidated). Pages sharing a domain
with the test set are excluded (template leakage). No test split —
evaluation uses the gold+diamond test set from export_grpo.

Run: python -m tools.data_export.export_sft   (after export_grpo, which
writes the split manifest this script needs for the domain guard)
"""

import json
import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv
from psycopg.rows import dict_row

from db.pool import create_pool
from tools.auto_annotation.prompt import USER_PROMPT_TEMPLATE_FINETUNED
from scraper.converter import html_to_markdown_light


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SYSTEM_PROMPT = "Return content line ranges as start:end pairs."


def _fetch_silver(pool) -> list[dict]:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT p.id, p.url, p.html, p.markdown, a.ranges, a.tier
                   FROM pages p
                   JOIN annotations a ON a.page_id = p.id
                   WHERE p.status = %s
                     AND a.tier = %s
                   ORDER BY p.id""",
                ("success", "silver"),
            )
            return cur.fetchall()


def _norm_host(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _test_set_domains(pool) -> set[str]:
    manifest_path = os.path.join(OUTPUT_DIR, "split_manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit("split_manifest.json not found — run export_grpo first "
                 "so the test split exists for the domain guard")
    with open(manifest_path) as f:
        test_ids = json.load(f).get("test_page_ids", [])
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT url FROM pages WHERE id = ANY(%s)", (test_ids,))
            return {_norm_host(url) for (url,) in cur.fetchall()}


def _ranges_to_compact(ranges: list[dict]) -> str:
    if not ranges:
        return "1:1"
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
        "tier": "silver",
    }


def main():
    load_dotenv()
    pool = create_pool()

    try:
        pages = _fetch_silver(pool)
        test_domains = _test_set_domains(pool)
    finally:
        pool.close()

    print(f"Found {len(pages)} silver pages")

    kept = [p for p in pages if _norm_host(p["url"]) not in test_domains]
    if len(kept) < len(pages):
        print(f"Dropped {len(pages) - len(kept)} pages sharing a domain "
              f"with the test set")
    pages = kept

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "train_sft.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for page in pages:
            row = _to_chat_row(page)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(pages)} rows)")


if __name__ == "__main__":
    main()
