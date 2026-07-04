import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag


_STRUCTURAL_TAGS = {"body", "header", "footer", "main", "nav", "section", "article", "aside", "form"}
_PROMOTABLE_ROLES = {"header", "footer", "main", "nav", "aside", "article"}
_SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "head", "meta", "link"}
_INLINE_MD = {"strong": "**", "b": "**", "em": "*", "i": "*"}
_INLINE_TAGS = {*_INLINE_MD, "a", "span"}
_SAFE_SCHEMES = {"http", "https", "mailto"}
_CONTENT_TAGS = {"ul", "ol", "dl", "table", "thead", "tbody", "tr", "td", "th",
                 "li", "dt", "dd", "p", "blockquote", "pre", "figure",
                 "figcaption", "details", "summary", "fieldset",
                 "h1", "h2", "h3", "h4", "h5", "h6"}

_LINK_RE = re.compile(r'\[.*?\]\([^)]*\)')


def _is_link_only(text: str) -> bool:
    """True if text contains only markdown links — no other content."""
    stripped = _LINK_RE.sub("", text).strip()
    return stripped == "" and _LINK_RE.search(text) is not None


def _safe_href(el: Tag) -> str:
    """Extract and sanitize a URL from an element's href or data-href."""
    raw = el.get("href") or el.get("data-href") or ""
    href = re.sub(r"[\x00-\x1f]", "", raw).strip()
    if not href:
        return ""
    scheme = href.split(":")[0].lower()
    if scheme in _SAFE_SCHEMES:
        return href.replace("(", "%28").replace(")", "%29")
    return ""


def _resolve_relative_urls(soup: BeautifulSoup, base_url: str) -> None:
    """Resolve relative URLs in href and data-href attributes to absolute."""
    for el in soup.find_all(["a", "button"]):
        for attr in ("href", "data-href"):
            val = el.get(attr)
            if val and not val.startswith(("#", "javascript:", "data:")):
                resolved = urljoin(base_url, val)
                el[attr] = resolved


def _get_text_with_formatting(el: Tag) -> str:
    """Extract text from an element, applying markdown markers for inline formatting."""
    parts: list[str] = []
    for child in el.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            text = " ".join(child.split())
            if text:
                parts.append(text)
        elif isinstance(child, Tag):
            if child.name in _SKIP_TAGS:
                continue
            if child.name == "a":
                inner = _get_text_with_formatting(child)
                if inner:
                    href = _safe_href(child)
                    if href:
                        parts.append(f"[{inner}]({href})")
                    else:
                        parts.append(inner)
                continue
            inner = _get_text_with_formatting(child)
            if inner:
                marker = _INLINE_MD.get(child.name, "")
                parts.append(f"{marker}{inner}{marker}")
    return " ".join(parts)


def _md_prefix(el: Tag) -> str:
    """Return markdown prefix for headings and list items."""
    tag = el.name
    if re.match(r"^h[1-6]$", tag):
        level = int(tag[1])
        return "#" * level + " "
    if tag == "li":
        parent = el.parent
        if parent is not None and isinstance(parent, Tag) and parent.name == "ol":
            index = 1
            for sibling in parent.children:
                if sibling is el:
                    break
                if isinstance(sibling, Tag) and sibling.name == "li":
                    index += 1
            return f"{index}. "
        return "- "
    return ""


def _is_text_unit(el: Tag) -> bool:
    """True if element is a leaf text node — has text and no block children with text."""
    has_text = False
    for child in el.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            if child.strip():
                has_text = True
            continue
        if isinstance(child, Tag):
            if child.name in _SKIP_TAGS:
                continue
            if child.name in _INLINE_TAGS:
                if child.get_text(strip=True):
                    has_text = True
                continue
            # Non-inline, non-skip tag — block child
            if child.get_text(strip=True):
                return False
    return has_text


def _get_inline_text(el: Tag) -> str:
    """Extract text from inline children only, skipping block-level children."""
    parts: list[str] = []
    for child in el.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            text = " ".join(child.split())
            if text:
                parts.append(text)
        elif isinstance(child, Tag):
            if child.name in _SKIP_TAGS:
                continue
            if child.name not in _INLINE_TAGS:
                continue
            if child.name == "a":
                inner = _get_text_with_formatting(child)
                if inner:
                    href = _safe_href(child)
                    if href:
                        parts.append(f"[{inner}]({href})")
                    else:
                        parts.append(inner)
            else:
                inner = _get_text_with_formatting(child)
                if inner:
                    marker = _INLINE_MD.get(child.name, "")
                    parts.append(f"{marker}{inner}{marker}")
    return " ".join(parts)


def _count_text_lines(el: Tag) -> int:
    """Count text lines this element would produce, simulating how _collect handles it."""
    if el.name in _SKIP_TAGS:
        return 0
    if el.name in _STRUCTURAL_TAGS or _semantic_role(el):
        if not el.get_text(strip=True):
            return 0
        sub: list[tuple[int, str, str, str]] = []
        _collect(el, 0, sub)
        return sum(1 for item in sub if item[1] == "text")
    if _is_text_unit(el):
        return 1 if _get_text_with_formatting(el) else 0
    if el.name in _INLINE_TAGS:
        return 1 if _get_text_with_formatting(el) else 0
    # Mixed content: inline text from element + recurse into children
    count = 0
    inline_text = _get_inline_text(el)
    if inline_text:
        count += 1
    sub = []
    _collect(el, 0, sub, skip_inline=bool(inline_text))
    count += sum(1 for item in sub if item[1] == "text")
    return count


def _collect_raw(
    html: str, base_url: str, collect_classes: bool = False,
) -> list[tuple[int, str, str, str]] | tuple[list[tuple[int, str, str, str]], list[list[str]], list[dict]]:
    """Parse HTML and collect raw items (Phase 1). Shared by both converters.

    When collect_classes=True, returns (raw_items, class_chains, element_metas) where
    class_chains[i] is the CSS class ancestor chain and element_metas[i] contains
    {has_image, has_input} for the i-th text/cookie item in raw_items.
    """
    from scraper.cookies import find_cookie_elements

    soup = BeautifulSoup(html, "html.parser")
    if base_url:
        _resolve_relative_urls(soup, base_url)

    # Replace cookie containers with placeholders preserving line count
    for el in find_cookie_elements(soup):
        line_count = _count_text_lines(el)
        if line_count > 0:
            placeholder = soup.new_tag("cookie-placeholder")
            placeholder["data-lines"] = str(line_count)
            placeholder.string = "cookies"  # non-empty so parent structural tags aren't skipped
            if el.name in _INLINE_TAGS:
                placeholder["data-inline"] = "1"
            el.replace_with(placeholder)
        else:
            el.decompose()

    root = soup.body or soup
    class_chains: list[list[str]] | None = [] if collect_classes else None
    element_metas: list[dict] | None = [] if collect_classes else None

    raw_items: list[tuple[int, str, str, str]] = []
    if isinstance(root, Tag) and root.name in _STRUCTURAL_TAGS:
        raw_items.append((0, "structural", root.name, ""))
        _collect(root, 1, raw_items, class_chains=class_chains,
                 element_metas=element_metas)
    else:
        _collect(root, 0, raw_items, class_chains=class_chains,
                 element_metas=element_metas)

    # Fallback: if body produced almost no text, unwrap <noscript> and retry
    text_count = sum(1 for item in raw_items if item[1] == "text")
    if text_count < 3:
        noscripts = root.find_all("noscript") if isinstance(root, Tag) else []
        if noscripts:
            for ns in noscripts:
                ns.unwrap()
            raw_items = []
            if class_chains is not None:
                class_chains.clear()
            if element_metas is not None:
                element_metas.clear()
            if isinstance(root, Tag) and root.name in _STRUCTURAL_TAGS:
                raw_items.append((0, "structural", root.name, ""))
                _collect(root, 1, raw_items, class_chains=class_chains,
                         element_metas=element_metas)
            else:
                _collect(root, 0, raw_items, class_chains=class_chains,
                         element_metas=element_metas)

    if collect_classes:
        return raw_items, class_chains, element_metas
    return raw_items


def _collapse_link_runs(
    items: list[tuple[int, str, str, str]],
    min_run: int = 3,
) -> list[tuple[int, str, str, str]]:
    """Replace contiguous runs of link-only text items with a single summary.

    Runs shorter than min_run are kept as-is.
    """
    result: list[tuple[int, str, str, str]] = []
    i = 0
    while i < len(items):
        item = items[i]
        if item[1] != "text" or not _is_link_only(item[2]):
            result.append(item)
            i += 1
            continue
        run_start = i
        while i < len(items) and items[i][1] == "text" and _is_link_only(items[i][2]):
            i += 1
        run = items[run_start:i]
        if len(run) < min_run:
            result.extend(run)
        else:
            link_count = sum(len(_LINK_RE.findall(it[2])) for it in run)
            line_count = len(run)
            result.append((run[0][0], "link_summary", str(link_count), str(line_count)))
    return result


def _render_items(
    raw_items: list[tuple[int, str, str, str]],
) -> str:
    """Phase 2 + 3: compress depths and render to string. Handles link_summary items."""
    stack: list[tuple[int, int]] = []
    items: list[tuple[int, str, str, str]] = []
    next_line_num = 1

    for raw_depth, item_type, text, md_prefix in raw_items:
        if item_type == "pop":
            while stack and stack[-1][0] > raw_depth:
                stack.pop()
            continue

        while stack and stack[-1][0] > raw_depth:
            stack.pop()

        if stack and stack[-1][0] == raw_depth:
            depth = stack[-1][1]
        elif stack:
            depth = stack[-1][1] + 1
            stack.append((raw_depth, depth))
        else:
            depth = 0
            stack.append((raw_depth, depth))

        if item_type == "structural":
            items.append((depth, "structural", text, ""))
        elif item_type == "link_summary":
            link_count = text
            line_count = int(md_prefix)
            items.append((depth, "link_summary", link_count, str(line_count)))
            next_line_num += line_count
        elif item_type == "cookie_summary":
            line_count = int(text)
            items.append((depth, "cookie_summary", str(line_count), ""))
            next_line_num += line_count
        else:
            items.append((depth, "text", text, md_prefix))
            next_line_num += 1

    max_line_num = next_line_num - 1
    col_width = max(2, len(str(max_line_num))) if max_line_num > 0 else 2

    lines: list[str] = []
    line_num = 1
    for depth, item_type, text, md_prefix in items:
        indent = "  " * depth
        if item_type == "structural":
            lines.append(f"{' ' * col_width} | {indent}[{text}]")
        elif item_type == "link_summary":
            link_count = text
            line_count = int(md_prefix)
            noun = "link" if int(link_count) == 1 else "links"
            lines.append(f"{str(line_num).rjust(col_width)} | {indent}- {link_count} {noun}")
            line_num += line_count
        elif item_type == "cookie_summary":
            line_count = int(text)
            noun = "line" if line_count == 1 else "lines"
            lines.append(f"{str(line_num).rjust(col_width)} | {indent}[cookies] - {line_count} {noun}")
            line_num += line_count
        else:
            lines.append(f"{str(line_num).rjust(col_width)} | {indent}{md_prefix}{text}")
            line_num += 1

    return "\n".join(lines)


def html_to_markdown(html: str, base_url: str = "") -> str:
    """Convert HTML to indented markdown with line numbers.

    Returns a single string with numbered lines. Structural tags (header,
    footer, nav, etc.) appear as indented labels. Text content appears as
    indented markdown with line numbers for LLM reference.
    """
    raw_items = _collect_raw(html, base_url)
    return _render_items(raw_items)


def _collapse_urls(text: str, page_url: str, page_origin: str, page_path: str) -> str:
    """Replace full URLs in markdown links with collapsed versions.

    - Same origin + starts with page path: ./remainder
    - Same origin + different path: /path
    - Different origin: unchanged
    """
    def _replace(m: re.Match) -> str:
        url = m.group(1)
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin != page_origin:
            return m.group(0)
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query
        if parsed.fragment:
            path += "#" + parsed.fragment
        if page_path and path.startswith(page_path.rstrip("/") + "/"):
            return f"(./{path[len(page_path.rstrip('/')) + 1:]})"
        if page_path and path.rstrip("/") == page_path.rstrip("/"):
            return "(./)"
        return f"({path})"
    return re.sub(r'\((https?://[^)]+)\)', _replace, text)


def html_to_markdown_light(html: str, base_url: str = "") -> str:
    """Like html_to_markdown but collapses runs of 3+ consecutive link-only lines
    and shortens same-domain URLs.

    Line numbers are preserved so annotations remain compatible across views.
    """
    raw_items = _collect_raw(html, base_url)
    raw_items = _collapse_link_runs(raw_items, min_run=3)
    rendered = _render_items(raw_items)
    if base_url:
        parsed = urlparse(base_url)
        page_origin = f"{parsed.scheme}://{parsed.netloc}"
        page_path = parsed.path
        rendered = _collapse_urls(rendered, base_url, page_origin, page_path)
        return f"[url: {base_url}]\n{rendered}"
    return rendered


def _semantic_role(el: Tag) -> str | None:
    """If a non-semantic element (div, etc.) has an id or class matching a
    structural role name, return that role so it can be promoted."""
    for attr in ("id", "role"):
        val = el.get(attr, "")
        if isinstance(val, str) and val.lower() in _PROMOTABLE_ROLES:
            return val.lower()
    classes = el.get("class")
    if classes:
        for cls in classes:
            if cls.lower() in _PROMOTABLE_ROLES:
                return cls.lower()
    return None


def _get_class_chain(el: Tag) -> list[str]:
    """Get ancestor class names for style grouping (element up to root)."""
    chain: list[str] = []
    node = el
    while node:
        if isinstance(node, Tag):
            classes = node.get("class")
            if classes:
                chain.append(" ".join(sorted(classes)))
        node = node.parent
    return chain


_HIDDEN_STYLE_RE = re.compile(
    r'display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?![.\d])', re.I)
_BOILERPLATE_ROLES = frozenset({
    "menu", "menubar", "complementary", "navigation",
    "alert", "dialog", "contentinfo", "banner",
})
_SCHEMA_CONTENT_PROPS = frozenset({
    "articlebody", "description", "reviewbody", "mainentity", "text",
})
_BYLINE_RE = re.compile(r'byline|author|dateline|writtenby|p-author', re.I)


def _get_element_meta(el: Tag) -> dict:
    """Check element-level flags for feature extraction."""
    has_image = el.name == "img" or bool(el.find("img"))
    has_input = el.name in ("input", "select", "textarea") or bool(
        el.find(["input", "select", "textarea"])
    )
    is_button = el.name == "button" or bool(el.find_parent("button"))
    has_aria_hidden = (
        el.get("aria-hidden") == "true"
        or bool(el.find_parent(attrs={"aria-hidden": "true"}))
    )
    in_table = el.name in ("table", "td", "th", "tr") or bool(el.find_parent("table"))
    in_details = el.name in ("details", "summary") or bool(el.find_parent("details"))

    # F1: is_hidden — inline style display:none/visibility:hidden/opacity:0
    is_hidden = False
    node = el
    while node and isinstance(node, Tag):
        style = node.get("style", "")
        if style and _HIDDEN_STYLE_RE.search(style):
            is_hidden = True
            break
        node = node.parent

    # F3: role_is_boilerplate — ARIA role on element/ancestors
    role_is_bp = False
    node = el
    while node and isinstance(node, Tag):
        role_val = node.get("role", "")
        if isinstance(role_val, str) and role_val.lower() in _BOILERPLATE_ROLES:
            role_is_bp = True
            break
        node = node.parent

    # F7: has_schema_content — itemprop on element/ancestors
    has_schema = False
    node = el
    while node and isinstance(node, Tag):
        itemprop = node.get("itemprop", "")
        if isinstance(itemprop, str) and itemprop.lower() in _SCHEMA_CONTENT_PROPS:
            has_schema = True
            break
        node = node.parent

    # F15: is_byline — class/id matches byline pattern
    is_byline = False
    el_classes = " ".join(el.get("class", []))
    el_id = el.get("id", "") or ""
    if _BYLINE_RE.search(el_classes) or _BYLINE_RE.search(el_id):
        is_byline = True

    # F16: in_figure — has <figure> ancestor
    in_figure = el.name == "figure" or bool(el.find_parent("figure"))

    # F19: is_last_child — no next sibling at block level
    is_last = True
    for sib in el.next_siblings:
        if isinstance(sib, Tag):
            is_last = False
            break

    # F4/F5: sibling_count and sibling_tag_uniformity
    # Find nearest block-level ancestor
    block_parent = el.parent
    while block_parent and isinstance(block_parent, Tag):
        if block_parent.name in _STRUCTURAL_TAGS or block_parent.name in _CONTENT_TAGS:
            break
        block_parent = block_parent.parent
    if block_parent and isinstance(block_parent, Tag):
        child_tags = [c.name for c in block_parent.children if isinstance(c, Tag)
                      and c.name not in _SKIP_TAGS]
        sibling_count = len(child_tags)
        if child_tags:
            most_common = max(set(child_tags), key=child_tags.count)
            sibling_tag_uniformity = child_tags.count(most_common) / len(child_tags)
        else:
            sibling_tag_uniformity = 0.0
    else:
        sibling_count = 0
        sibling_tag_uniformity = 0.0

    return {
        "has_image": int(has_image),
        "has_input": int(has_input),
        "is_button": int(is_button),
        "has_aria_hidden": int(has_aria_hidden),
        "in_table": int(in_table),
        "in_details": int(in_details),
        "is_hidden": int(is_hidden),
        "role_is_boilerplate": int(role_is_bp),
        "has_schema_content": int(has_schema),
        "is_byline": int(is_byline),
        "in_figure": int(in_figure),
        "is_last_child": int(is_last),
        "sibling_count": sibling_count,
        "sibling_tag_uniformity": round(sibling_tag_uniformity, 4),
    }


def _collect(
    el: Tag,
    depth: int,
    out: list[tuple[int, str, str, str]],
    skip_inline: bool = False,
    class_chains: list[list[str]] | None = None,
    element_metas: list[dict] | None = None,
) -> None:
    """Walk the DOM, emitting structural labels and text items with raw depth.

    Structural tags get labels. Content tags increment depth for children.
    Non-semantic wrappers (div, span, etc.) are transparent — no depth change.
    When skip_inline is True, inline tags are skipped (already extracted by caller).
    When class_chains is provided, appends the CSS class ancestor chain for each
    text/cookie item emitted (aligned 1:1 with non-structural items in out).
    When element_metas is provided, appends per-element metadata (has_image, has_input).
    """
    for child in el.children:
        if not isinstance(child, Tag):
            continue
        if child.name in _SKIP_TAGS:
            continue

        if child.name == "cookie-placeholder":
            if skip_inline and child.get("data-inline"):
                continue
            line_count = int(child.get("data-lines", "0"))
            if line_count > 0:
                out.append((depth, "cookie_summary", str(line_count), ""))
                if class_chains is not None:
                    class_chains.append([])
                if element_metas is not None:
                    element_metas.append({"has_image": 0, "has_input": 0,
                                          "is_button": 0, "has_aria_hidden": 0,
                                          "in_table": 0, "in_details": 0,
                                          "is_hidden": 0, "role_is_boilerplate": 0,
                                          "has_schema_content": 0, "is_byline": 0,
                                          "in_figure": 0, "is_last_child": 0,
                                          "sibling_count": 0, "sibling_tag_uniformity": 0.0})
            continue

        # Promote divs with semantic id/class/role to structural tags
        role = None if child.name in _STRUCTURAL_TAGS else _semantic_role(child)

        if child.name in _STRUCTURAL_TAGS or role:
            tag_label = role or child.name
            if child.get_text(strip=True):
                out.append((depth, "structural", tag_label, ""))
                _collect(child, depth + 1, out, class_chains=class_chains,
                         element_metas=element_metas)
                out.append((depth, "pop", "", ""))

        elif _is_text_unit(child):
            if skip_inline and child.name in _INLINE_TAGS:
                continue
            text = _get_text_with_formatting(child)
            if text:
                if child.name in ("a", "button"):
                    href = _safe_href(child)
                    if href:
                        text = f"[{text}]({href})"
                out.append((depth, "text", text, _md_prefix(child)))
                if class_chains is not None:
                    class_chains.append(_get_class_chain(child))
                if element_metas is not None:
                    element_metas.append(_get_element_meta(child))

        elif child.name in _INLINE_TAGS:
            if skip_inline:
                continue
            text = _get_text_with_formatting(child)
            if text:
                if child.name in ("a", "button"):
                    href = _safe_href(child)
                    if href:
                        text = f"[{text}]({href})"
                out.append((depth, "text", text, _md_prefix(child)))
                if class_chains is not None:
                    class_chains.append(_get_class_chain(child))
                if element_metas is not None:
                    element_metas.append(_get_element_meta(child))

        else:
            inline_text = _get_inline_text(child)
            has_inline = bool(inline_text)
            if has_inline:
                out.append((depth, "text", inline_text, _md_prefix(child)))
                if class_chains is not None:
                    class_chains.append(_get_class_chain(child))
                if element_metas is not None:
                    element_metas.append(_get_element_meta(child))
            next_depth = depth + 1 if child.name in _CONTENT_TAGS else depth
            _collect(child, next_depth, out, skip_inline=has_inline,
                     class_chains=class_chains, element_metas=element_metas)
