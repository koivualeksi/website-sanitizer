import asyncio
import time
from urllib.parse import urlparse


class DomainRateLimiter:
    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._last_request: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def wait(self, url: str):
        domain = urlparse(url).netloc
        if domain not in self._locks:
            self._locks[domain] = asyncio.Lock()
        async with self._locks[domain]:
            now = time.monotonic()
            last = self._last_request.get(domain, 0.0)
            wait_time = self._min_interval - (now - last)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_request[domain] = time.monotonic()
