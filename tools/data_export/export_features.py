"""Export annotated pages as per-line feature rows for BiGRU/CNN training.

Each content line becomes one row with hand-crafted features and a binary
label (1 = main content, 0 = boilerplate).

Output: data/features_train.csv, data/features_test.csv, data/idf_table.json

Run: python -m tools.export_features
"""

import json
import math
import os
import re
from collections import Counter
from urllib.parse import urlparse

from dotenv import load_dotenv
from psycopg.rows import dict_row

from db.pool import create_pool
from scraper.converter import _collect_raw, _LINK_RE, _is_link_only

# Readability-style class/id name patterns (matched against full ancestor class chain)
_POSITIVE_CLASS_RE = re.compile(
    r'article|body|content|entry|hentry|h-entry|main|page|pagination|post|text|blog|story', re.I)
_NEGATIVE_CLASS_RE = re.compile(
    r'-ad-|hidden|banner|combx|comment|com-|contact|footer|gdpr|masthead|media|meta|outbrain'
    r'|promo|related|scroll|share|shoutbox|sidebar|skyscraper|sponsor|shopping|tags|widget', re.I)
_UNLIKELY_CLASS_RE = re.compile(
    r'-ad-|ai2html|banner|breadcrumbs|combx|comment|community|cover-wrap|disqus|extra|footer'
    r'|gdpr|header|legends|menu|related|remark|replies|rss|shoutbox|sidebar|skyscraper|social'
    r'|sponsor|supplemental|ad-break|agegate|pagination|pager|popup|yom-remote', re.I)

_BOILERPLATE_CLASS_RE = re.compile(
    r'byline|author|timestamp|outbrain|taboola|criteo|consent|modal|paywall'
    r'|obfuscated|blurred|overlay|embed|newsletter|subscribe|login|rating', re.I)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

_DEFAULT_META = {
    "has_image": 0, "has_input": 0, "is_button": 0,
    "has_aria_hidden": 0, "in_table": 0, "in_details": 0,
    "is_hidden": 0, "role_is_boilerplate": 0,
    "has_schema_content": 0, "is_byline": 0,
    "in_figure": 0, "is_last_child": 0,
    "sibling_count": 0, "sibling_tag_uniformity": 0.0,
}

_SENTENCE_END_RE = re.compile(r'[.!?](?:\s|$)')
_MD_LINK_URL_RE = re.compile(r'\[([^\]]*)\]\(([^)]*)\)')


def _external_link_count(text: str, base_url: str) -> int:
    """Count markdown links pointing to a different domain."""
    base_domain = urlparse(base_url).netloc.lower().lstrip("www.")
    count = 0
    for m in _MD_LINK_URL_RE.finditer(text):
        link_domain = urlparse(m.group(2)).netloc.lower().lstrip("www.")
        if link_domain and link_domain != base_domain:
            count += 1
    return count


def _text_features(text: str, md_prefix: str) -> dict:
    link_chars = sum(len(m.group(0)) for m in _LINK_RE.finditer(text))
    text_len = len(text)
    words = text.split()
    prefix_stripped = md_prefix.rstrip(" ")
    word_lengths = [len(w) for w in words] if words else [0]
    alpha_chars = sum(1 for c in text if c.isalpha())
    upper_chars = sum(1 for c in text if c.isupper())
    digit_chars = sum(1 for c in text if c.isdigit())
    punct_chars = sum(1 for c in text if c in '.,;:!?()[]{}"\'-')
    tokens = text.split()
    unique_tokens = set(w.lower() for w in tokens) if tokens else set()
    return {
        "text_length": text_len,
        "word_count": len(words),
        "link_ratio": link_chars / text_len if text_len > 0 else 0.0,
        "link_count": len(_LINK_RE.findall(text)),
        "is_link_only": int(_is_link_only(text)),
        "is_heading": int(prefix_stripped.startswith("#")),
        "heading_level": len(prefix_stripped) if prefix_stripped.startswith("#") else 0,
        "has_bold": int("**" in text),
        "is_list_item": int(md_prefix == "- " or (len(md_prefix) >= 3 and md_prefix[-2:] == ". " and md_prefix[:-2].isdigit())),
        "punctuation_ratio": round(punct_chars / text_len, 4) if text_len > 0 else 0.0,
        "punctuation_count": punct_chars,
        "sentence_count": len(_SENTENCE_END_RE.findall(text)),
        "avg_word_length": round(sum(word_lengths) / len(word_lengths), 2),
        "uppercase_ratio": round(upper_chars / alpha_chars, 4) if alpha_chars > 0 else 0.0,
        "uppercase_count": upper_chars,
        "number_ratio": round(digit_chars / text_len, 4) if text_len > 0 else 0.0,
        "number_count": digit_chars,
        "type_token_ratio": round(len(unique_tokens) / len(tokens), 4) if tokens else 0.0,
        "unique_word_count": len(unique_tokens),
        "short_link_count": sum(1 for m in _MD_LINK_URL_RE.finditer(text) if len(m.group(1)) < 10),
        "short_link_ratio": (sum(1 for m in _MD_LINK_URL_RE.finditer(text) if len(m.group(1)) < 10)
                             / max(len(_LINK_RE.findall(text)), 1)),
        "_raw_text": text,  # kept temporarily for IDF/frequency, stripped before CSV
    }


def _label_for_range(line_start: int, line_count: int, ranges: list[dict]) -> int:
    """Return 1 if any line in [line_start, line_start+line_count) falls in a content range."""
    for r in ranges:
        rs, re_ = r.get("start", 0), r.get("end", 0)
        if line_start + line_count - 1 >= rs and line_start <= re_:
            return 1
    return 0


def _class_features(class_chain: list[str]) -> dict:
    """Compute class name regex features from a CSS ancestor chain."""
    full_chain = " ".join(class_chain) if class_chain else ""
    own_classes = class_chain[0] if class_chain else ""
    # own_class_weight: Readability-style ±25 scoring on element's OWN class/id
    own_weight = 0
    if own_classes:
        if _POSITIVE_CLASS_RE.search(own_classes):
            own_weight += 25
        if _NEGATIVE_CLASS_RE.search(own_classes):
            own_weight -= 25
        if _UNLIKELY_CLASS_RE.search(own_classes):
            own_weight -= 25
    return {
        "has_positive_class": int(bool(_POSITIVE_CLASS_RE.search(full_chain))) if full_chain else 0,
        "has_negative_class": int(bool(_NEGATIVE_CLASS_RE.search(full_chain))) if full_chain else 0,
        "has_unlikely_class": int(bool(_UNLIKELY_CLASS_RE.search(full_chain))) if full_chain else 0,
        "has_boilerplate_class": int(bool(_BOILERPLATE_CLASS_RE.search(full_chain))) if full_chain else 0,
        "own_class_weight": own_weight,
    }


def extract_line_features(html: str, base_url: str) -> list[dict]:
    """Extract per-line feature dicts from HTML using the converter's DOM walk."""
    raw_items, class_chains, element_metas = _collect_raw(html, base_url, collect_classes=True)

    struct_stack: list[tuple[int, str]] = []  # (raw_depth, tag_name)
    rows: list[dict] = []
    line_num = 1
    class_idx = 0

    for raw_depth, item_type, text, md_prefix in raw_items:
        if item_type == "pop":
            while struct_stack and struct_stack[-1][0] >= raw_depth:
                struct_stack.pop()
            continue

        if item_type == "structural":
            while struct_stack and struct_stack[-1][0] >= raw_depth:
                struct_stack.pop()
            struct_stack.append((raw_depth, text))
            continue

        # text, link_summary, or cookie_summary — all get a row
        ancestors = [tag for _, tag in struct_stack]
        parent_tag = ancestors[-1] if ancestors else "body"

        semantic_depth = len(struct_stack)
        row = {
            "line_num": line_num,
            "depth": raw_depth,
            "ancestor_depth_ratio": round(semantic_depth / raw_depth, 4) if raw_depth > 0 else 0.0,
            "parent_tag": parent_tag,
            "in_header": int("header" in ancestors),
            "in_nav": int("nav" in ancestors),
            "in_main": int("main" in ancestors),
            "in_article": int("article" in ancestors),
            "in_footer": int("footer" in ancestors),
            "in_aside": int("aside" in ancestors),
            "in_form": int("form" in ancestors),
            "is_link_summary": 0,
            "is_cookie_summary": 0,
        }

        if item_type == "text":
            row.update(_text_features(text, md_prefix))
            row["external_link_count"] = _external_link_count(text, base_url)
            row["comma_count"] = text.count(",") + text.count("\uff0c")
            row["span_lines"] = 1
            chain = class_chains[class_idx] if class_idx < len(class_chains) else []
            meta = element_metas[class_idx] if class_idx < len(element_metas) else _DEFAULT_META
            row["_class_chain"] = chain
            row.update(_class_features(chain))
            row.update(meta)
            class_idx += 1
            line_num += 1
        elif item_type == "link_summary":
            lc = int(md_prefix)  # line_count stored in md_prefix slot
            row.update({
                "text_length": 0, "word_count": 0, "link_ratio": 1.0,
                "link_count": lc,
                "is_link_only": 1, "is_heading": 0, "heading_level": 0,
                "has_bold": 0, "is_list_item": 1, "is_link_summary": 1,
                "span_lines": lc,
                "punctuation_ratio": 0.0, "punctuation_count": 0,
                "sentence_count": 0, "avg_word_length": 0.0,
                "uppercase_ratio": 0.0, "uppercase_count": 0,
                "number_ratio": 0.0, "number_count": 0,
                "type_token_ratio": 0.0, "unique_word_count": 0,
                "short_link_count": 0, "short_link_ratio": 0.0,
                "comma_count": 0,
                "external_link_count": 0,
                "has_positive_class": 0, "has_negative_class": 0, "has_unlikely_class": 0,
                "has_boilerplate_class": 0, "own_class_weight": 0,
                **_DEFAULT_META,
                "_raw_text": "",
                "_class_chain": [],
            })
            line_num += lc
        elif item_type == "cookie_summary":
            lc = int(text)
            chain = class_chains[class_idx] if class_idx < len(class_chains) else []
            meta = element_metas[class_idx] if class_idx < len(element_metas) else _DEFAULT_META
            row.update({
                "text_length": 0, "word_count": 0, "link_ratio": 1.0,
                "link_count": 0,
                "is_link_only": 0, "is_heading": 0, "heading_level": 0,
                "has_bold": 0, "is_list_item": 0, "is_cookie_summary": 1,
                "span_lines": lc,
                "punctuation_ratio": 0.0, "punctuation_count": 0,
                "sentence_count": 0, "avg_word_length": 0.0,
                "uppercase_ratio": 0.0, "uppercase_count": 0,
                "number_ratio": 0.0, "number_count": 0,
                "type_token_ratio": 0.0, "unique_word_count": 0,
                "short_link_count": 0, "short_link_ratio": 0.0,
                "comma_count": 0,
                "external_link_count": 0,
                **_class_features(chain), **meta,
                "_raw_text": "",
                "_class_chain": chain,
            })
            class_idx += 1
            line_num += lc

        rows.append(row)

    total_lines = line_num - 1
    total_chars = sum(r["text_length"] for r in rows)
    cumulative = 0
    for r in rows:
        r["position_pct"] = round(r["line_num"] / total_lines, 4) if total_lines > 0 else 0.0
        r["total_lines"] = total_lines
        cumulative += r["text_length"]
        r["cumulative_text_pct"] = round(cumulative / total_chars, 4) if total_chars > 0 else 0.0

    _add_window_features(rows)
    _add_word_novelty(rows)
    _add_bigram_novelty(rows)
    _add_container_score(rows)
    _add_content_region(rows)
    _add_style_groups(rows)
    return rows


def _add_window_features(rows: list[dict], window: int = 5) -> None:
    """Add rolling-window averages for text_length and link_ratio."""
    n = len(rows)
    half = window // 2
    text_lengths = [r["text_length"] for r in rows]
    link_ratios = [r["link_ratio"] for r in rows]
    for i, row in enumerate(rows):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        span = end - start
        row["mean_text_length_w5"] = round(sum(text_lengths[start:end]) / span, 1)
        row["mean_link_ratio_w5"] = round(sum(link_ratios[start:end]) / span, 4)


def _add_word_novelty(rows: list[dict]) -> None:
    """Add page-level word novelty features.

    For each word on the page, its novelty = 1 / (total occurrences on page).
    Unique words score 1.0, repeated boilerplate words score low.
    Per line: mean novelty (normalized) and sum novelty (length-sensitive).
    """
    page_word_counts: Counter = Counter()
    row_tokens: list[list[str]] = []
    for row in rows:
        tokens = _tokenize(row.get("_raw_text", ""))
        row_tokens.append(tokens)
        page_word_counts.update(tokens)
    for row, tokens in zip(rows, row_tokens):
        if tokens:
            novelty_scores = [1.0 / page_word_counts[w] for w in tokens]
            row["word_novelty"] = round(sum(novelty_scores) / len(novelty_scores), 4)
            row["word_novelty_sum"] = round(sum(novelty_scores), 4)
        else:
            row["word_novelty"] = 0.0
            row["word_novelty_sum"] = 0.0


def _add_bigram_novelty(rows: list[dict]) -> None:
    """F17: mean 1/count(bigram) across word bigrams on the page."""
    page_bigram_counts: Counter = Counter()
    row_bigrams: list[list[tuple[str, str]]] = []
    for row in rows:
        tokens = _tokenize(row.get("_raw_text", ""))
        bigrams = list(zip(tokens, tokens[1:])) if len(tokens) > 1 else []
        row_bigrams.append(bigrams)
        page_bigram_counts.update(bigrams)
    for row, bigrams in zip(rows, row_bigrams):
        if bigrams:
            scores = [1.0 / page_bigram_counts[bg] for bg in bigrams]
            row["bigram_novelty"] = round(sum(scores) / len(scores), 4)
        else:
            row["bigram_novelty"] = 0.0


def _add_container_score(rows: list[dict]) -> None:
    """F6: Readability-style content score per structural block.

    Score = sum of (1 + commas + min(3, text_len/100)) for lines sharing
    the same (parent_tag, depth) block.
    """
    if not rows:
        return
    blocks: list[list[int]] = []
    current_block: list[int] = [0]
    for i in range(1, len(rows)):
        if (rows[i]["parent_tag"] == rows[i - 1]["parent_tag"]
                and rows[i]["depth"] == rows[i - 1]["depth"]):
            current_block.append(i)
        else:
            blocks.append(current_block)
            current_block = [i]
    if current_block:
        blocks.append(current_block)
    for block in blocks:
        score = sum(1 + rows[j].get("comma_count", 0) + min(3, rows[j].get("text_length", 0) / 100)
                    for j in block)
        for j in block:
            rows[j]["container_content_score"] = round(score, 2)


def _add_content_region(rows: list[dict]) -> None:
    """F9: binary — is line in the block containing >50% of page text?

    Finds the block with the most text. If it has >50% of total text,
    marks all lines in it.
    """
    if not rows:
        return
    total_text = sum(r.get("text_length", 0) for r in rows)
    if total_text == 0:
        for r in rows:
            r["content_region_ratio"] = 0
        return
    blocks: list[list[int]] = []
    current_block: list[int] = [0]
    for i in range(1, len(rows)):
        if (rows[i]["parent_tag"] == rows[i - 1]["parent_tag"]
                and rows[i]["depth"] == rows[i - 1]["depth"]):
            current_block.append(i)
        else:
            blocks.append(current_block)
            current_block = [i]
    if current_block:
        blocks.append(current_block)
    best_block = max(blocks, key=lambda b: sum(rows[j].get("text_length", 0) for j in b))
    best_text = sum(rows[j].get("text_length", 0) for j in best_block)
    best_set = set(best_block) if best_text > 0.5 * total_text else set()
    for i, r in enumerate(rows):
        r["content_region_ratio"] = int(i in best_set)


def _add_style_groups(rows: list[dict], min_size: int = 3) -> None:
    """Add style group features based on CSS class ancestry.

    Each row is assigned to its nearest styled ancestor's class. Groups with
    fewer than min_size lines are merged into their parent's class, iteratively.
    Then per-group aggregate features are computed.
    """
    if not rows:
        return

    from collections import defaultdict

    # Step 1: assign each row to nearest styled ancestor class
    assignments = []
    for row in rows:
        chain = row.get("_class_chain", [])
        assignments.append(chain[0] if chain else "_no_class")

    # Step 2: iteratively merge small groups into parent class
    for _ in range(8):
        counts = Counter(assignments)
        changed = False
        for i, row in enumerate(rows):
            if counts[assignments[i]] < min_size:
                chain = row.get("_class_chain", [])
                cur = assignments[i]
                for j, c in enumerate(chain):
                    if c == cur and j + 1 < len(chain):
                        assignments[i] = chain[j + 1]
                        changed = True
                        break
        if not changed:
            break

    # Step 3: compute per-group aggregate features
    group_indices: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(assignments):
        group_indices[g].append(i)

    for g, indices in group_indices.items():
        size = len(indices)
        link_ratios = [rows[i].get("link_ratio", 0.0) for i in indices]
        word_counts = [rows[i].get("word_count", 0) for i in indices]
        mean_lr = sum(link_ratios) / len(link_ratios)
        mean_wc = sum(word_counts) / len(word_counts)
        for i in indices:
            rows[i]["style_group_size"] = size
            rows[i]["style_group_link_density"] = round(mean_lr, 4)
            rows[i]["style_group_mean_words"] = round(mean_wc, 1)

    # Clean up temporary field
    for row in rows:
        row.pop("_class_chain", None)


def _add_block_features(rows: list[dict]) -> None:
    """Add features about the structural block each line belongs to."""
    if not rows:
        return
    # Group rows by parent_tag + depth to identify blocks
    # A block is a contiguous run of rows with the same (parent_tag, depth)
    blocks: list[list[int]] = []
    current_block: list[int] = [0]
    for i in range(1, len(rows)):
        if (rows[i]["parent_tag"] == rows[i - 1]["parent_tag"]
                and rows[i]["depth"] == rows[i - 1]["depth"]):
            current_block.append(i)
        else:
            blocks.append(current_block)
            current_block = [i]
    if current_block:
        blocks.append(current_block)

    for block in blocks:
        block_size = len(block)
        lengths = [rows[j]["text_length"] for j in block]
        total_text = sum(lengths)
        total_links = sum(rows[j]["text_length"] * rows[j]["link_ratio"] for j in block)
        block_link_density = total_links / total_text if total_text > 0 else 0.0
        mean_len = total_text / block_size if block_size > 0 else 0.0
        if block_size > 1 and mean_len > 0:
            variance = sum((l - mean_len) ** 2 for l in lengths) / block_size
            text_cv = round(variance ** 0.5 / mean_len, 4)
        else:
            text_cv = 0.0
        # F8: block_heading_density
        n_headings = sum(1 for j in block if rows[j].get("is_heading", 0) == 1)
        block_heading_density = n_headings / block_size if block_size > 0 else 0.0
        # F12: block_li_p_ratio (list items / paragraphs)
        n_li = sum(1 for j in block if rows[j].get("is_list_item", 0) == 1)
        n_p = sum(1 for j in block if rows[j].get("is_heading", 0) == 0
                  and rows[j].get("is_list_item", 0) == 0
                  and rows[j].get("is_link_summary", 0) == 0
                  and rows[j].get("is_cookie_summary", 0) == 0)
        block_li_p_ratio = n_li / max(n_p, 1)
        # F13: block_img_p_ratio
        n_img = sum(1 for j in block if rows[j].get("has_image", 0) == 1)
        block_img_p_ratio = n_img / max(n_p, 1)
        # F14: block_input_density
        n_input = sum(1 for j in block if rows[j].get("has_input", 0) == 1)
        block_input_density = n_input / max(block_size, 1)
        # F11: container_p_ratio (paragraph-like / total)
        container_p_ratio = n_p / max(block_size, 1)
        for rank, j in enumerate(block):
            rows[j]["block_size"] = block_size
            rows[j]["block_text_density"] = round(mean_len, 1)
            rows[j]["block_link_density"] = round(block_link_density, 4)
            rows[j]["relative_position_in_block"] = round(rank / (block_size - 1), 4) if block_size > 1 else 0.0
            rows[j]["sibling_text_variance"] = text_cv
            rows[j]["block_heading_density"] = round(block_heading_density, 4)
            rows[j]["block_li_p_ratio"] = round(block_li_p_ratio, 4)
            rows[j]["block_img_p_ratio"] = round(block_img_p_ratio, 4)
            rows[j]["block_input_density"] = round(block_input_density, 4)
            rows[j]["container_p_ratio"] = round(container_p_ratio, 4)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization, strip markdown link syntax."""
    clean = _LINK_RE.sub(lambda m: m.group(0).split("](")[0][1:], text)
    return [w.lower().strip(".,;:!?()[]{}\"'-") for w in clean.split() if len(w) > 1]


def build_idf_table(all_page_rows: list[list[dict]]) -> dict[str, float]:
    """Build IDF table: for each word, log(N / df) where df = pages containing the word."""
    n_pages = len(all_page_rows)
    doc_freq: Counter = Counter()
    for page_rows in all_page_rows:
        page_words = set()
        for row in page_rows:
            page_words.update(_tokenize(row.get("_raw_text", "")))
        for word in page_words:
            doc_freq[word] += 1
    return {word: round(math.log(n_pages / df), 4) for word, df in doc_freq.items()}


def build_line_frequency(all_page_rows: list[list[dict]]) -> Counter:
    """Count how many pages each exact line text appears on."""
    line_pages: dict[str, set[int]] = {}
    for page_idx, page_rows in enumerate(all_page_rows):
        for row in page_rows:
            text = row.get("_raw_text", "").strip()
            if text:
                if text not in line_pages:
                    line_pages[text] = set()
                line_pages[text].add(page_idx)
    return Counter({text: len(pages) for text, pages in line_pages.items()})


def _add_text_uniqueness(rows: list[dict], idf_table: dict[str, float],
                         line_freq: Counter, n_pages: int) -> None:
    """Add mean_idf, max_idf (per-page IDF), and line_frequency features."""
    # Per-page IDF: N = total lines on page, df = lines containing each word
    n_lines = len(rows)
    line_tokens = [_tokenize(row.get("_raw_text", "")) for row in rows]
    doc_freq: Counter = Counter()
    for tokens in line_tokens:
        doc_freq.update(set(tokens))
    page_idf = {w: math.log(n_lines / df) for w, df in doc_freq.items()} if n_lines > 0 else {}

    for row, tokens in zip(rows, line_tokens):
        if tokens:
            idfs = [page_idf.get(t, 0.0) for t in tokens]
            row["mean_idf"] = round(sum(idfs) / len(idfs), 4)
            row["max_idf"] = round(max(idfs), 4)
        else:
            row["mean_idf"] = 0.0
            row["max_idf"] = 0.0
        freq = line_freq.get(row.get("_raw_text", "").strip(), 0)
        row["line_frequency"] = round(freq / n_pages, 4) if n_pages > 0 else 0.0


COLUMNS = [
    "page_id", "dataset", "line_num", "span_lines",
    "position_pct", "total_lines", "depth",
    "ancestor_depth_ratio",
    "parent_tag",
    "in_header", "in_nav", "in_main", "in_article",
    "in_footer", "in_aside", "in_form",
    "text_length", "word_count", "link_ratio", "link_count",
    "is_link_only", "is_heading", "heading_level",
    "has_bold", "is_list_item",
    "is_link_summary", "is_cookie_summary",
    "punctuation_ratio", "punctuation_count",
    "sentence_count", "avg_word_length",
    "uppercase_ratio", "uppercase_count",
    "number_ratio", "number_count",
    "type_token_ratio", "unique_word_count",
    "short_link_count", "short_link_ratio",
    "comma_count",
    "external_link_count",
    "has_positive_class", "has_negative_class", "has_unlikely_class",
    "has_boilerplate_class", "own_class_weight",
    "has_image", "has_input",
    "is_button", "has_aria_hidden", "in_table", "in_details",
    "is_hidden", "role_is_boilerplate", "has_schema_content",
    "is_byline", "in_figure", "is_last_child",
    "sibling_count", "sibling_tag_uniformity",
    "mean_idf", "max_idf", "line_frequency",
    "word_novelty", "word_novelty_sum",
    "bigram_novelty",
    "cumulative_text_pct",
    "mean_text_length_w5", "mean_link_ratio_w5",
    "block_size", "block_text_density", "block_link_density",
    "relative_position_in_block", "sibling_text_variance",
    "block_heading_density", "block_li_p_ratio", "block_img_p_ratio",
    "block_input_density", "container_p_ratio",
    "container_content_score", "content_region_ratio",
    "style_group_size", "style_group_link_density", "style_group_mean_words",
    "label",
]


def _fetch_annotated(pool) -> list[dict]:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT p.id, p.url, p.html, a.ranges,
                          COALESCE(p.dataset, 'original') AS dataset
                   FROM pages p
                   JOIN annotations a ON a.page_id = p.id
                   WHERE p.status = %s
                     AND a.validated = %s
                     AND COALESCE(p.dataset, 'original') = 'original'
                   ORDER BY p.id""",
                ("success", True),
            )
            return cur.fetchall()


def _page_to_rows(page: dict) -> list[dict]:
    rows = extract_line_features(page["html"], page["url"])
    _add_block_features(rows)
    ranges = page["ranges"] or []
    dataset = page.get("dataset", "original")
    for row in rows:
        row["page_id"] = page["id"]
        row["dataset"] = dataset
        row["label"] = _label_for_range(row["line_num"], row["span_lines"], ranges)
    return rows


def _write_csv(path: str, all_rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in all_rows:
            values = []
            for col in COLUMNS:
                v = row[col]
                if isinstance(v, float):
                    values.append(str(v))
                else:
                    values.append(str(v))
            f.write(",".join(values) + "\n")


def _export_all(pages: list[dict], workers: int = 8) -> None:
    """Export all pages to a single features.csv with dataset column."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Extracting features ({workers} workers)...")
    all_page_rows, failed = [], 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_page_to_rows, p): p for p in pages}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                all_page_rows.append(fut.result())
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Failed: {e}")
            if i % 2000 == 0:
                print(f"  [{i}/{len(pages)}] ok={len(all_page_rows)} failed={failed}")
    print(f"  {len(all_page_rows)} pages ({failed} failed)")

    # IDF is per-page so no corpus table needed, but line_frequency still needs corpus
    line_freq = build_line_frequency(all_page_rows)
    n = len(all_page_rows)
    for page_rows in all_page_rows:
        _add_text_uniqueness(page_rows, {}, line_freq, n)

    all_rows = []
    for page_rows in all_page_rows:
        for row in page_rows:
            row.pop("_raw_text", None)
            all_rows.append(row)

    out_path = os.path.join(OUTPUT_DIR, "features.csv")
    _write_csv(out_path, all_rows)
    content = sum(1 for r in all_rows if r["label"] == 1)
    print(f"Wrote {out_path} ({len(all_rows)} rows, {content} content)")

    idf_path = os.path.join(OUTPUT_DIR, "idf_table.json")
    with open(idf_path, "w", encoding="utf-8") as f:
        json.dump({}, f)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Export all pages to single features.csv")
    args = parser.parse_args()

    load_dotenv()
    pool = create_pool()

    try:
        pages = _fetch_annotated(pool)
    finally:
        pool.close()

    print(f"Found {len(pages)} annotated pages")

    if args.all:
        _export_all(pages)
        return

    manifest_path = os.path.join(OUTPUT_DIR, "split_manifest.json")
    fixed_test_ids: set[int] = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            old_manifest = json.load(f)
        fixed_test_ids = set(old_manifest.get("test_page_ids", []))

    train_pages = [p for p in pages if p["id"] not in fixed_test_ids]
    test_pages = [p for p in pages if p["id"] in fixed_test_ids]

    print(f"Train: {len(train_pages)} pages, Test: {len(test_pages)} pages")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pass 1: extract features for all pages
    print("Pass 1: extracting features...")
    train_page_rows = [_page_to_rows(p) for p in train_pages]
    test_page_rows = [_page_to_rows(p) for p in test_pages]

    # Build IDF and line frequency from training pages only
    print("Building IDF table and line frequency...")
    idf_table = build_idf_table(train_page_rows)
    line_freq = build_line_frequency(train_page_rows)
    n_train = len(train_page_rows)
    print(f"  IDF vocabulary: {len(idf_table)} words")
    print(f"  Unique lines: {len(line_freq)}")

    # Pass 2: add text uniqueness features
    print("Pass 2: adding text uniqueness features...")
    for page_rows in train_page_rows:
        _add_text_uniqueness(page_rows, idf_table, line_freq, n_train)
    for page_rows in test_page_rows:
        _add_text_uniqueness(page_rows, idf_table, line_freq, n_train)

    # Flatten and strip _raw_text
    train_rows: list[dict] = []
    for page_rows in train_page_rows:
        for row in page_rows:
            row.pop("_raw_text", None)
            train_rows.append(row)

    test_rows: list[dict] = []
    for page_rows in test_page_rows:
        for row in page_rows:
            row.pop("_raw_text", None)
            test_rows.append(row)

    train_path = os.path.join(OUTPUT_DIR, "features_train.csv")
    _write_csv(train_path, train_rows)
    content_count = sum(1 for r in train_rows if r["label"] == 1)
    print(f"Wrote {train_path} ({len(train_rows)} rows, {content_count} content / {len(train_rows) - content_count} boilerplate)")

    test_path = os.path.join(OUTPUT_DIR, "features_test.csv")
    _write_csv(test_path, test_rows)
    content_count = sum(1 for r in test_rows if r["label"] == 1)
    print(f"Wrote {test_path} ({len(test_rows)} rows, {content_count} content / {len(test_rows) - content_count} boilerplate)")

    # Save IDF table for inference
    idf_path = os.path.join(OUTPUT_DIR, "idf_table.json")
    with open(idf_path, "w", encoding="utf-8") as f:
        json.dump(idf_table, f, ensure_ascii=False)
    print(f"Wrote {idf_path} ({len(idf_table)} words)")


if __name__ == "__main__":
    main()
