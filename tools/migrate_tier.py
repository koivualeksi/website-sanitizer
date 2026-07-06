"""Add tier generated column to annotations table.

Idempotent — safe to run multiple times.

Run: python -m tools.migrate_tier
"""

from dotenv import load_dotenv
from psycopg.rows import dict_row

from db.pool import create_pool


def main():
    load_dotenv()
    pool = create_pool()

    with pool.connection() as conn:
        # Pre-migration check: validated-but-empty rows that will become bronze
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT count(*) AS n FROM annotations WHERE validated AND ranges = '[]'::jsonb"
            )
            n = cur.fetchone()["n"]
            if n:
                print(f"Note: {n} validated-but-empty rows will become bronze (not exported)")

        conn.execute("""
            ALTER TABLE annotations ADD COLUMN IF NOT EXISTS tier TEXT GENERATED ALWAYS AS (
                CASE
                    WHEN skipped OR ranges = '[]'::jsonb THEN 'bronze'
                    WHEN source = 'manual' AND validated  THEN 'diamond'
                    WHEN source = 'llm'    AND validated  THEN 'gold'
                    ELSE 'silver'
                END
            ) STORED
        """)

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT tier, count(*) AS n FROM annotations GROUP BY tier ORDER BY tier")
            rows = cur.fetchall()

    print("Tier counts:")
    for row in rows:
        print(f"  {row['tier']:8s} {row['n']}")

    pool.close()


if __name__ == "__main__":
    main()
