"""Evaluate trafilatura extraction accuracy against ground truth annotations.

Compares trafilatura's content extraction with manually validated annotations.
For each annotated page, runs the raw HTML through trafilatura, then maps the
extracted text back to markdown line numbers via keyword matching.

Run: python -m pytest tests/test_trafilatura.py -s -v
Or:  python tests/test_trafilatura.py
"""

import re
import os

from dotenv import load_dotenv
from psycopg.rows import dict_row

import trafilatura

from db.pool import create_pool


def _fetch_ground_truth(pool) -> list[dict]:
    """Fetch all validated manual annotations with their page HTML and markdown."""
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT p.id, p.url, p.html, p.markdown, a.ranges
                   FROM pages p
                   JOIN annotations a ON a.page_id = p.id
                   WHERE p.status = %s
                     AND a.tier = %s
                   ORDER BY p.id""",
                ("success", "diamond"),
            )
            return cur.fetchall()


def _parse_markdown_lines(markdown: str) -> dict[int, str]:
    """Parse structured markdown into {line_number: text_content}.

    Lines look like: '  3 |   Some text here'
    Structural labels like '   | [nav]' have no line number and are skipped.
    """
    lines = {}
    for raw_line in markdown.split("\n"):
        m = re.match(r"^\s*(\d+)\s*\|\s*(.*)", raw_line)
        if m:
            num = int(m.group(1))
            text = m.group(2).strip()
            if text:
                lines[num] = text
    return lines


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase alphanumeric words (3+ chars) from text."""
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9\u00c0-\u024f]{3,}", text)}


def _run_trafilatura(html: str) -> str | None:
    """Run trafilatura on raw HTML, return extracted text or None."""
    return trafilatura.extract(
        html,
        favor_recall=True,
        include_tables=True,
        include_formatting=True,
        include_comments=False,
    )


def _map_extraction_to_lines(
    extracted_text: str, md_lines: dict[int, str], threshold: float = 0.5
) -> set[int]:
    """Map trafilatura's extracted text to markdown line numbers.

    For each markdown line, check what fraction of its keywords appear in the
    extracted text. If >= threshold, consider that line "predicted as content".
    """
    extracted_kw = _extract_keywords(extracted_text)
    predicted = set()
    for num, text in md_lines.items():
        line_kw = _extract_keywords(text)
        if not line_kw:
            continue
        overlap = len(line_kw & extracted_kw) / len(line_kw)
        if overlap >= threshold:
            predicted.add(num)
    return predicted


def _ranges_to_line_set(ranges: list[dict]) -> set[int]:
    """Expand annotation ranges to a set of line numbers."""
    lines = set()
    for r in ranges:
        for n in range(r["start"], r["end"] + 1):
            lines.add(n)
    return lines


def _compute_metrics(predicted: set[int], truth: set[int]) -> dict:
    """Compute precision, recall, and IoU from two line sets."""
    if not predicted and not truth:
        return {"precision": 1.0, "recall": 1.0, "iou": 1.0}
    intersection = len(predicted & truth)
    union = len(predicted | truth)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(truth) if truth else 0.0
    iou = intersection / union if union else 0.0
    return {"precision": precision, "recall": recall, "iou": iou}


def _keyword_stats(
    md_lines: dict[int, str],
    truth_lines: set[int],
    predicted_lines: set[int],
) -> dict:
    """Count keywords by category: true positive, false positive, false negative lines."""
    tp_kw, fp_kw, fn_kw = 0, 0, 0
    for num, text in md_lines.items():
        count = len(_extract_keywords(text))
        if num in predicted_lines and num in truth_lines:
            tp_kw += count
        elif num in predicted_lines and num not in truth_lines:
            fp_kw += count
        elif num not in predicted_lines and num in truth_lines:
            fn_kw += count
    return {"tp_keywords": tp_kw, "fp_keywords": fp_kw, "fn_keywords": fn_kw}


def run_evaluation():
    """Run the full trafilatura evaluation and return results."""
    load_dotenv()
    pool = create_pool()

    try:
        pages = _fetch_ground_truth(pool)
    finally:
        pool.close()

    if not pages:
        print("No validated annotations found in DB.")
        return []

    print(f"\nEvaluating trafilatura on {len(pages)} pages...\n")
    results = []

    for page in pages:
        page_id = page["id"]
        url = page["url"]
        html = page["html"]
        markdown = page["markdown"]
        truth_ranges = page["ranges"] or []

        md_lines = _parse_markdown_lines(markdown)
        truth_lines = _ranges_to_line_set(truth_ranges)
        # Filter truth to lines that actually exist in markdown
        truth_lines &= set(md_lines.keys())

        extracted = _run_trafilatura(html)

        if extracted is None:
            print(f"  Page {page_id}: trafilatura returned nothing")
            results.append({
                "page_id": page_id,
                "url": url,
                "metrics": None,
                "keyword_stats": None,
                "truth_line_count": len(truth_lines),
                "pred_line_count": 0,
                "total_lines": len(md_lines),
            })
            continue

        pred_lines = _map_extraction_to_lines(extracted, md_lines)
        metrics = _compute_metrics(pred_lines, truth_lines)
        kw_stats = _keyword_stats(md_lines, truth_lines, pred_lines)

        results.append({
            "page_id": page_id,
            "url": url,
            "metrics": metrics,
            "keyword_stats": kw_stats,
            "truth_line_count": len(truth_lines),
            "pred_line_count": len(pred_lines),
            "total_lines": len(md_lines),
        })

    _print_summary(results)
    return results


def _print_summary(results: list[dict]):
    """Print a summary table of results."""
    print("\n" + "=" * 100)
    print(
        f"{'ID':>5}  {'IoU':>6}  {'Prec':>6}  {'Rec':>6}  "
        f"{'Pred':>5}  {'Truth':>5}  {'Total':>5}  "
        f"{'TP kw':>6}  {'FP kw':>6}  {'FN kw':>6}  URL"
    )
    print("-" * 100)

    successful = []
    for r in results:
        page_id = r["page_id"]
        url = r["url"][:35]
        truth = r["truth_line_count"]
        pred = r["pred_line_count"]
        total = r["total_lines"]

        if r["metrics"] is None:
            print(
                f"{page_id:>5}  {'FAIL':>6}  {'':>6}  {'':>6}  "
                f"{pred:>5}  {truth:>5}  {total:>5}  "
                f"{'':>6}  {'':>6}  {'':>6}  {url}"
            )
        else:
            m = r["metrics"]
            kw = r["keyword_stats"]
            print(
                f"{page_id:>5}  {m['iou']:>6.3f}  {m['precision']:>6.3f}  "
                f"{m['recall']:>6.3f}  {pred:>5}  {truth:>5}  {total:>5}  "
                f"{kw['tp_keywords']:>6}  {kw['fp_keywords']:>6}  {kw['fn_keywords']:>6}  {url}"
            )
            successful.append(r)

    print("-" * 100)

    if successful:
        n = len(successful)
        mean_iou = sum(r["metrics"]["iou"] for r in successful) / n
        mean_prec = sum(r["metrics"]["precision"] for r in successful) / n
        mean_rec = sum(r["metrics"]["recall"] for r in successful) / n
        total_tp = sum(r["keyword_stats"]["tp_keywords"] for r in successful)
        total_fp = sum(r["keyword_stats"]["fp_keywords"] for r in successful)
        total_fn = sum(r["keyword_stats"]["fn_keywords"] for r in successful)
        print(
            f"{'MEAN':>5}  {mean_iou:>6.3f}  {mean_prec:>6.3f}  "
            f"{mean_rec:>6.3f}  {'':>5}  {'':>5}  {'':>5}  "
            f"{total_tp:>6}  {total_fp:>6}  {total_fn:>6}  "
            f"({n}/{len(successful)} succeeded)"
        )
    else:
        print("No successful evaluations.")

    print("=" * 100)


# -- pytest entry point --

def test_trafilatura_accuracy():
    """Pytest-compatible entry point for the trafilatura evaluation."""
    load_dotenv()
    results = run_evaluation()
    successful = [r for r in results if r["metrics"] is not None]

    assert len(successful) > 0, "No pages were successfully evaluated"

    mean_iou = sum(r["metrics"]["iou"] for r in successful) / len(successful)
    print(f"\nTrafilatura mean IoU: {mean_iou:.3f}")


# -- standalone entry point --

if __name__ == "__main__":
    run_evaluation()
