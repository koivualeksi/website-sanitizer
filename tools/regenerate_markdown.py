"""Regenerate markdown for all pages using structure-first generation.

Replaces stored markdown with output from html_to_markdown(),
clears source_map, and deletes all annotations.

Run:
    python -m tools.regenerate_markdown
    python -m tools.regenerate_markdown --dry-run
"""
import argparse

from dotenv import load_dotenv

load_dotenv()

from psycopg.rows import dict_row

from db.pool import create_pool
from scraper.converter import html_to_markdown


def main():
    parser = argparse.ArgumentParser(description="Regenerate markdown for all pages")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without modifying data")
    parser.add_argument(
        "--confirm", action="store_true",
        help="Required to actually run. This deletes all annotations.",
    )
    parser.add_argument(
        "--keep-annotations", action="store_true",
        help="Regenerate markdown but keep annotations intact.",
    )
    args = parser.parse_args()

    pool = create_pool()

    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT id, url, html FROM pages "
                "WHERE status = 'success' AND html IS NOT NULL"
            )
            rows = cur.fetchall()

            cur.execute("SELECT COUNT(*) AS n FROM annotations")
            ann_count = cur.fetchone()["n"]

    keep = args.keep_annotations
    print(f"Pages to regenerate: {len(rows)}")
    if keep:
        print("Annotations: KEEPING (--keep-annotations)")
    else:
        print(f"Annotations to delete: {ann_count}")

    if args.dry_run:
        print("\n--dry-run: no changes made.")
        pool.close()
        return

    if not args.confirm:
        print("\nPass --confirm to proceed.")
        pool.close()
        return

    with pool.connection() as conn:
        with conn.cursor() as cur:
            for i, row in enumerate(rows, 1):
                markdown = html_to_markdown(row["html"], base_url=row["url"])
                cur.execute(
                    "UPDATE pages SET markdown = %s, source_map = NULL WHERE id = %s",
                    (markdown, row["id"]),
                )
                if i % 500 == 0:
                    print(f"  [{i}/{len(rows)}]")

            deleted = 0
            if not keep:
                cur.execute("DELETE FROM annotations")
                deleted = cur.rowcount

        conn.commit()

    if keep:
        print(f"\nDone: {len(rows)} pages updated, annotations kept.")
    else:
        print(f"\nDone: {len(rows)} pages updated, {deleted} annotations deleted.")
    pool.close()


if __name__ == "__main__":
    main()
