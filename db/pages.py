import asyncio
from functools import partial

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def _fetch_pending_sync(pool: ConnectionPool, limit: int | None = None) -> list[dict]:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            if limit is not None:
                cur.execute(
                    "SELECT id, url FROM pages WHERE status IS NULL LIMIT %s",
                    (limit,),
                )
            else:
                cur.execute("SELECT id, url FROM pages WHERE status IS NULL")
            return cur.fetchall()


def _update_success_sync(
    pool: ConnectionPool, page_id: int, html: str, markdown: str,
):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE pages SET html = %s, markdown = %s, source_map = NULL, "
            "status = 'success', scraped_at = now() WHERE id = %s",
            (html, markdown, page_id),
        )


def _update_failure_sync(pool: ConnectionPool, page_id: int, error: str):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE pages SET status = 'failed', error = %s, scraped_at = now() WHERE id = %s",
            (error, page_id),
        )


def _update_skipped_sync(pool: ConnectionPool, page_id: int, reason: str):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE pages SET status = 'skipped', error = %s, scraped_at = now() WHERE id = %s",
            (reason, page_id),
        )


async def fetch_pending(pool: ConnectionPool, limit: int | None = None) -> list[dict]:
    return await asyncio.to_thread(_fetch_pending_sync, pool, limit)


async def update_success(
    pool: ConnectionPool, page_id: int, html: str, markdown: str,
):
    await asyncio.to_thread(_update_success_sync, pool, page_id, html, markdown)


async def update_failure(pool: ConnectionPool, page_id: int, error: str):
    await asyncio.to_thread(_update_failure_sync, pool, page_id, error)


async def update_skipped(pool: ConnectionPool, page_id: int, reason: str):
    await asyncio.to_thread(_update_skipped_sync, pool, page_id, reason)
