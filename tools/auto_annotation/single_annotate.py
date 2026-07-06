"""Single-LLM bulk annotation: one model, no validation (silver tier).

Fast path for scaling up SFT training data. Each page gets annotated by
a single model and saved as source='llm', validated=False -> silver tier.

Usage:
    python -m tools.auto_annotation.single_annotate [--limit N] [--model MODEL] [--dry-run] [--watch [SECONDS]]
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
from psycopg.rows import dict_row

from db.pool import create_pool
from scraper.converter import html_to_markdown_light
from tools.auto_annotation.bulk_annotate import _call_model, _fetch_unannotated
from tools.auto_annotation.config import CALL_DELAY, MODEL_DEFAULT


def _save_annotation(pool, page_id, ranges):
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO annotations (page_id, ranges, source, validated, skipped, updated_at)
               VALUES (%s, %s, 'llm', FALSE, FALSE, now())
               ON CONFLICT (page_id) DO UPDATE SET
                   ranges = EXCLUDED.ranges,
                   source = 'llm',
                   validated = FALSE,
                   skipped = FALSE,
                   updated_at = now()""",
            (page_id, json.dumps(ranges)),
        )


def _run(pool, pages, api_key, model, dry_run):
    print(f"Found {len(pages)} unannotated pages")
    print(f"Model: {model}")
    if dry_run:
        print("--dry-run: not saving")
    print()

    ok = 0
    failed = 0

    for i, page in enumerate(pages):
        page_id = page["id"]
        url = page["url"][:55]
        markdown = html_to_markdown_light(page["html"], page["url"])

        print(f"[{i+1}/{len(pages)}] Page {page_id}: {url} ... ", end="", flush=True)

        ranges = _call_model(markdown, model, api_key)

        if ranges is None:
            print("FAILED")
            failed += 1
        else:
            if not dry_run:
                _save_annotation(pool, page_id, ranges)
            print(f"OK ({len(ranges)} ranges) -> silver")
            ok += 1

        if i < len(pages) - 1:
            time.sleep(CALL_DELAY)

    print(f"\nDone: {ok} ok, {failed} failed")


def main():
    limit = 50
    dry_run = "--dry-run" in sys.argv
    watch = "--watch" in sys.argv
    watch_interval = 30
    model = MODEL_DEFAULT

    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        model = sys.argv[idx + 1]
    if watch:
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
        if watch:
            print(f"Watch mode: polling every {watch_interval}s for new pages (limit={limit})\n")
            while True:
                pages = _fetch_unannotated(pool, limit)
                if pages:
                    _run(pool, pages, api_key, model, dry_run)
                else:
                    print(f"No unannotated pages found. Waiting {watch_interval}s...")
                print()
                time.sleep(watch_interval)
        else:
            pages = _fetch_unannotated(pool, limit)
            _run(pool, pages, api_key, model, dry_run)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
