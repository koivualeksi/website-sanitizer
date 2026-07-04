"""Cookie/consent banner removal from HTML.

Removes known cookie consent containers before conversion to markdown.
Selectors are grouped by provider for auditability.
"""

from bs4 import BeautifulSoup, Tag

# Each entry: (provider name, list of CSS selectors targeting the top-level container)
# We only need the outermost container — removing it takes all children with it.
COOKIE_SELECTORS: list[tuple[str, list[str]]] = [
    ("CookieBot", [
        "#CybotCookiebotDialog",
        "#CybotCookiebotDialogBodyUnderlay",
        "#CybotCookiebotFader",
        "#Cookiebot",
        "#CookiebotWidget",
        "#CookieConsentStateDisplayStyles",
        "#CookiebotDialogStyle",
        "#CookieBanner",
        ".CookieDeclaration",
    ]),
    ("CookieYes", [
        "#cookieyes",
        "#cookieyes-banner",
        "[class*='cky-consent']",
        "[class*='cky-cookie']",
        "[class*='wcc-cookie']",
        "[role='dialog'].wcc-modal",
        ".cky-modal",
        "[class*='cky-accordion']",
    ]),
    ("CookieScript", [
        "#cookiescript_injected_wrapper",
    ]),
    ("Complianz", [
        "#cmplz-cookiebanner-container",
        "#cmplz-manage-consent",
        "[class*='cmplz-consent']",
        "[class*='cookie-statement']",
    ]),
    ("OneTrust", [
        "#onetrust-consent-sdk",
        "#onetrust-banner-sdk",
        ".onetrust-pc-dark-filter",
    ]),
    ("MooveGDPR", [
        "#moove_gdpr_cookie_info_bar",
        "#moove_gdpr_cookie_modal",
        "[id^='moove_gdpr']",
        "[id^='moove-gdpr']",
    ]),
    ("COI", [
        "[class*='coi-consent-banner']",
        "[class*='cookie-details']",
    ]),
    ("Cassie", [
        "#cassie-widget",
        "#cassieLoaderId",
    ]),
    ("AirCookie", [
        "[data-aircookie-remove-on]",
    ]),
    ("CookieFirst", [
        "[data-cookiefirst-widget]",
    ]),
    ("CookieLawInfo", [
        "#cookie-law-info-bar",
        "#cookie-law-info-again",
        "[class*='cookielawinfo']",
        "[class*='wt-cli-cookie']",
    ]),
    ("PrestoGDPR", [
        "[class*='pr-cookie']",
    ]),
    ("HubSpot", [
        "#hs-eu-cookie-confirmation",
    ]),
    ("CookieInformation", [
        "#cookie-information-template-wrapper",
    ]),
    ("Borlabs", [
        "#BorlabsCookieBox",
    ]),
    ("CookieHub", [
        ".ch2",
    ]),
    ("Teamtailor", [
        "dialog.company-links",
        "dialog.text-company-primary",
        "dialog[data-controller='common--cookies--alert']",
        "dialog[data-controller='common--cookies--preferences']",
    ]),
    ("CookieConsent", [
        "#cliSettingsPopup",
        "[role='dialog'].cc-window",
        "[role='dialog']#cm",
        "[role='dialog']#s-cnt",
        "dialog.cookie-popup",
    ]),
    ("Termly", [
        "#termly-code-snippet-support",
    ]),
    ("SimpleBanner", [
        ".simple-banner",
    ]),
    ("Generic", [
        "[data-cookieconsent]",
        "#hu-cookies",
        "#cookieDiv",
        "#cookie-container",
        "#cookie-notice",
        "#cookiebar",
        "#consentUI",
        "#dsh-cookie-modal",
        "#fyfconsent",
        "[id^='pd-cp-cookie']",
        ".tru_cookie-dialog_wrapper",
        ".cookieadmin_cookie_modal",
        ".consent-banner-root",
        ".cookies",
        ".ez-consent",
        ".outershell",
        ".outershellRiverty",
    ]),
]

# Script/style tags with cookie-related IDs are always safe to remove.
# These are JS/CSS for consent managers, never visible content.
COOKIE_SCRIPT_ID_PATTERNS = [
    "cookie", "consent", "gdpr", "cmplz", "moove_gdpr",
    "cookielaw", "pll_cookie", "wpml-cookie", "wc-js-cookie",
]


def find_cookie_elements(soup: BeautifulSoup) -> list[Tag]:
    """Find outermost cookie/consent container elements (visible content only).

    Script/style tags with cookie IDs are decomposed immediately (never visible).
    Returns only outermost matched elements — nested matches are excluded.
    """
    candidates = []
    candidate_ids: set[int] = set()

    for _provider, selectors in COOKIE_SELECTORS:
        for selector in selectors:
            for el in soup.select(selector):
                if not isinstance(el, Tag) or id(el) in candidate_ids:
                    continue
                candidates.append(el)
                candidate_ids.add(id(el))

    # Script/style with cookie IDs — decompose, they never produce lines
    for el in soup.find_all(["script", "style", "link"]):
        el_id = el.get("id", "")
        if not el_id:
            continue
        if any(pat in el_id.lower() for pat in COOKIE_SCRIPT_ID_PATTERNS):
            if id(el) not in candidate_ids:
                el.decompose()

    # Keep only outermost
    outermost: list[Tag] = []
    for el in candidates:
        if el.parent is None:
            continue
        has_ancestor = False
        for p in el.parents:
            if isinstance(p, Tag) and id(p) in candidate_ids:
                has_ancestor = True
                break
        if not has_ancestor:
            outermost.append(el)

    return outermost


def remove_cookie_banners(soup: BeautifulSoup) -> list[dict]:
    """Remove known cookie/consent containers from the soup (in-place).

    Returns a list of removals for auditing:
      [{"provider": str, "selector": str, "tag": str, "id": str|None, "classes": list}]
    """
    removals = []
    to_remove = []

    # 1. Named selectors (provider-specific containers)
    for provider, selectors in COOKIE_SELECTORS:
        for selector in selectors:
            for el in soup.select(selector):
                if not isinstance(el, Tag) or el in to_remove:
                    continue
                removals.append({
                    "provider": provider,
                    "selector": selector,
                    "tag": el.name,
                    "id": el.get("id"),
                    "classes": el.get("class", []),
                })
                to_remove.append(el)

    # 2. Script/style tags with cookie-related IDs (never visible content)
    for el in soup.find_all(["script", "style", "link"]):
        if el in to_remove:
            continue
        el_id = el.get("id", "")
        if not el_id:
            continue
        el_id_lower = el_id.lower()
        if any(pat in el_id_lower for pat in COOKIE_SCRIPT_ID_PATTERNS):
            removals.append({
                "provider": "ScriptCleanup",
                "selector": f"#{el_id}",
                "tag": el.name,
                "id": el_id,
                "classes": [],
            })
            to_remove.append(el)

    for el in to_remove:
        el.decompose()
    return removals


def clean_html(html: str) -> str:
    """Remove cookie/consent banners from an HTML string."""
    soup = BeautifulSoup(html, "html.parser")
    remove_cookie_banners(soup)
    return str(soup)
