from cloakbrowser import launch_async


async def create_browser(headless=True):
    return await launch_async(headless=headless)
