from playwright.async_api import Browser, TimeoutError as PlaywrightTimeout


async def fetch_page(browser: Browser, url: str, timeout: int = 45000) -> str:
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout)
        except PlaywrightTimeout:
            pass
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)
        return await page.content()
    finally:
        await page.close()
