from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


class RobotsChecker:
    def __init__(self):
        self._cache: dict[str, RobotFileParser | None] = {}
        self._client = httpx.AsyncClient(
            timeout=5.0,
            headers={"User-Agent": "websanitbot/1.0"},
        )

    async def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"

        if domain not in self._cache:
            try:
                resp = await self._client.get(f"{domain}/robots.txt")
                if resp.status_code == 200:
                    rp = RobotFileParser()
                    rp.parse(resp.text.splitlines())
                    self._cache[domain] = rp
                else:
                    self._cache[domain] = None
            except (httpx.HTTPError, Exception):
                self._cache[domain] = None

        rp = self._cache[domain]
        if rp is None:
            return True
        return rp.can_fetch(user_agent, url)

    async def close(self):
        await self._client.aclose()
