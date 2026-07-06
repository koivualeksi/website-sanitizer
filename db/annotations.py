import asyncio
import json

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SORTABLE_COLUMNS = {"id", "url", "source", "validated", "tier"}
SORT_DIRECTIONS = {"asc", "desc"}


def _list_pages_sync(
    pool: ConnectionPool,
    tab: str,
    page: int,
    per_page: int,
    sort_col: str,
    sort_dir: str,
    search: str,
) -> tuple[list[dict], int]:
    if sort_col not in SORTABLE_COLUMNS:
        sort_col = "id"
    if sort_dir not in SORT_DIRECTIONS:
        sort_dir = "asc"

    conditions = ["p.status = 'success'"]
    params: list = []

    if tab == "unvalidated":
        conditions.append("(a.validated IS NULL OR a.validated = FALSE)")
        conditions.append("(a.skipped IS NULL OR a.skipped = FALSE)")
    elif tab == "validated":
        conditions.append("a.validated = TRUE")

    if search:
        if search.strip().isdigit():
            conditions.append("p.id = %s")
            params.append(int(search.strip()))
        else:
            search_escaped = (
                search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            conditions.append("p.url ILIKE %s ESCAPE '\\'")
            params.append(f"%{search_escaped}%")

    where = " AND ".join(conditions)

    # Column mapping for sort (prefix with table alias)
    col_map = {
        "id": "p.id",
        "url": "p.url",
        "source": "a.source",
        "validated": "a.validated",
        "tier": "CASE a.tier WHEN 'diamond' THEN 3 WHEN 'gold' THEN 2 WHEN 'silver' THEN 1 ELSE 0 END",
    }
    order_col = col_map[sort_col]
    # Nulls last for annotation columns so unannotated pages sort to end
    nulls = " NULLS LAST" if sort_col in ("source", "validated", "tier") else ""
    # Urgent pages always first
    order_clause = f"ORDER BY p.urgent DESC, {order_col} {sort_dir}{nulls}"

    offset = (page - 1) * per_page

    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT COUNT(*) AS total FROM pages p LEFT JOIN annotations a ON a.page_id = p.id WHERE {where}",
                params,
            )
            total = cur.fetchone()["total"]

            cur.execute(
                f"""
                SELECT p.id, p.url, p.status, p.urgent,
                       a.ranges, a.source, a.validated, a.skipped, a.tier
                FROM pages p
                LEFT JOIN annotations a ON a.page_id = p.id
                WHERE {where}
                {order_clause}
                LIMIT %s OFFSET %s
                """,
                params + [per_page, offset],
            )
            rows = cur.fetchall()

    return rows, total


def _get_annotation_sync(pool: ConnectionPool, page_id: int) -> dict | None:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT p.id, p.url, p.html, p.markdown,
                       a.ranges, a.source, a.validated, a.skipped,
                       COALESCE(a.has_cookies, FALSE) AS has_cookies,
                       a.tier
                FROM pages p
                LEFT JOIN annotations a ON a.page_id = p.id
                WHERE p.id = %s AND p.status = 'success'
                """,
                (page_id,),
            )
            return cur.fetchone()



def _save_annotation_sync(
    pool: ConnectionPool,
    page_id: int,
    ranges: list[dict],
    source: str,
) -> None:
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO annotations (page_id, ranges, source, validated, skipped, updated_at)
            VALUES (%s, %s, %s, TRUE, FALSE, now())
            ON CONFLICT (page_id) DO UPDATE SET
                ranges = EXCLUDED.ranges,
                source = EXCLUDED.source,
                validated = TRUE,
                skipped = FALSE,
                updated_at = now()
            """,
            (page_id, json.dumps(ranges), source),
        )


def _accept_annotation_sync(pool: ConnectionPool, page_id: int) -> None:
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE annotations
            SET validated = TRUE, skipped = FALSE, source = 'manual', updated_at = now()
            WHERE page_id = %s
            """,
            (page_id,),
        )


def _mark_skipped_sync(pool: ConnectionPool, page_id: int) -> None:
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO annotations (page_id, skipped, updated_at)
            VALUES (%s, TRUE, now())
            ON CONFLICT (page_id) DO UPDATE SET
                skipped = TRUE,
                validated = FALSE,
                updated_at = now()
            """,
            (page_id,),
        )


def _toggle_cookies_sync(pool: ConnectionPool, page_id: int) -> bool:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO annotations (page_id, has_cookies, updated_at)
                VALUES (%s, TRUE, now())
                ON CONFLICT (page_id) DO UPDATE SET
                    has_cookies = NOT annotations.has_cookies,
                    updated_at = now()
                RETURNING has_cookies
                """,
                (page_id,),
            )
            return cur.fetchone()["has_cookies"]


def _get_next_unvalidated_sync(
    pool: ConnectionPool, current_page_id: int
) -> int | None:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT p.id FROM pages p
                LEFT JOIN annotations a ON a.page_id = p.id
                WHERE p.status = 'success'
                  AND (a.validated IS NULL OR a.validated = FALSE)
                  AND (a.skipped IS NULL OR a.skipped = FALSE)
                  AND p.id > %s
                ORDER BY p.urgent DESC, p.id ASC
                LIMIT 1
                """,
                (current_page_id,),
            )
            row = cur.fetchone()
            return row["id"] if row else None


def _get_tab_counts_sync(pool: ConnectionPool) -> dict:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (
                        WHERE (a.validated IS NULL OR a.validated = FALSE)
                          AND (a.skipped IS NULL OR a.skipped = FALSE)
                    ) AS unvalidated,
                    COUNT(*) FILTER (WHERE a.validated = TRUE) AS validated
                FROM pages p
                LEFT JOIN annotations a ON a.page_id = p.id
                WHERE p.status = 'success'
                """
            )
            return cur.fetchone()


# Async wrappers

async def list_pages(pool, tab, page, per_page, sort_col, sort_dir, search):
    return await asyncio.to_thread(
        _list_pages_sync, pool, tab, page, per_page, sort_col, sort_dir, search
    )


async def get_annotation(pool, page_id):
    return await asyncio.to_thread(_get_annotation_sync, pool, page_id)



async def save_annotation(pool, page_id, ranges, source):
    await asyncio.to_thread(_save_annotation_sync, pool, page_id, ranges, source)


async def accept_annotation(pool, page_id):
    await asyncio.to_thread(_accept_annotation_sync, pool, page_id)


async def mark_skipped(pool, page_id):
    await asyncio.to_thread(_mark_skipped_sync, pool, page_id)


async def toggle_cookies(pool, page_id):
    return await asyncio.to_thread(_toggle_cookies_sync, pool, page_id)


async def get_next_unvalidated(pool, current_page_id):
    return await asyncio.to_thread(
        _get_next_unvalidated_sync, pool, current_page_id
    )


async def get_tab_counts(pool):
    return await asyncio.to_thread(_get_tab_counts_sync, pool)
