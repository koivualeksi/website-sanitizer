import argparse
import asyncio

from dotenv import load_dotenv

from scraper import runner


def main():
    parser = argparse.ArgumentParser(description="Async web scraper using CloakBrowser")
    parser.add_argument("--runners", type=int, default=3, help="number of concurrent browser instances")
    parser.add_argument("--delay", type=float, default=10, help="per-domain minimum interval in seconds")
    parser.add_argument("--timeout", type=int, default=45, help="page load timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="retry count per URL")
    parser.add_argument("--dry-run", action="store_true", help="list pending URLs without scraping")
    parser.add_argument("--limit", type=int, default=None, help="only process first N pending URLs")
    args = parser.parse_args()

    load_dotenv()

    asyncio.run(runner.run(
        runners=args.runners,
        delay=args.delay,
        timeout=args.timeout,
        retries=args.retries,
        dry_run=args.dry_run,
        limit=args.limit,
    ))
