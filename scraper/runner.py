import asyncio

from tqdm import tqdm

from db import pages, pool
from scraper.browser import create_browser
from scraper.converter import html_to_markdown
from scraper.fetcher import fetch_page
from scraper.rate_limiter import DomainRateLimiter
from scraper.robots import RobotsChecker


async def run(runners: int, delay: float, timeout: int, retries: int, dry_run: bool, limit: int | None):
    db_pool = pool.create_pool()
    pending = await pages.fetch_pending(db_pool, limit)

    if not pending:
        print("No pending URLs.")
        db_pool.close()
        return

    if dry_run:
        for row in pending:
            print(row["url"])
        print(f"\n{len(pending)} pending URLs.")
        db_pool.close()
        return

    browsers = [await create_browser() for _ in range(runners)]
    rate_limiter = DomainRateLimiter(delay)
    robots = RobotsChecker()
    semaphore = asyncio.Semaphore(runners)
    timeout_ms = timeout * 1000

    progress = tqdm(total=len(pending), desc="Scraping", unit="page")
    counts = {"success": 0, "failed": 0, "skipped": 0}

    async def process(i: int, row: dict):
        page_id, url = row["id"], row["url"]
        async with semaphore:
            await rate_limiter.wait(url)

            if not await robots.is_allowed(url):
                await pages.update_skipped(db_pool, page_id, "robots.txt")
                counts["skipped"] += 1
                progress.update(1)
                return

            browser = browsers[i % runners]
            last_error = None
            for attempt in range(retries + 1):
                try:
                    html = await fetch_page(browser, url, timeout=timeout_ms)
                    markdown = html_to_markdown(html, base_url=url)
                    await pages.update_success(db_pool, page_id, html, markdown)
                    counts["success"] += 1
                    progress.update(1)
                    return
                except Exception as e:
                    last_error = str(e)

            await pages.update_failure(db_pool, page_id, last_error)
            counts["failed"] += 1
            progress.update(1)

    tasks = [asyncio.create_task(process(i, row)) for i, row in enumerate(pending)]
    await asyncio.gather(*tasks)

    progress.close()

    for b in browsers:
        await b.close()
    await robots.close()
    db_pool.close()

    print(f"\nDone. success={counts['success']} failed={counts['failed']} skipped={counts['skipped']}")
