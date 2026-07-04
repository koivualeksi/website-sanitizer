"""Evaluation suite: measure LLM annotation accuracy against ground truth.

Run: python -m pytest tests/test_llm_accuracy.py -s -v
Or:  python tests/test_llm_accuracy.py
"""

import os
import time

from dotenv import load_dotenv
from psycopg.rows import dict_row

from db.pool import create_pool
from tools.auto_annotation.annotator import annotate_page, CALL_DELAY


def _fetch_ground_truth(pool) -> list[dict]:
    """Fetch all validated manual annotations with their page data."""
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT p.id, p.url, p.markdown, a.ranges
                   FROM pages p
                   JOIN annotations a ON a.page_id = p.id
                   WHERE p.status = %s
                     AND a.validated = %s
                     AND a.source = %s
                   ORDER BY p.id""",
                ("success", True, "manual"),
            )
            return cur.fetchall()


def _ranges_to_line_set(ranges: list[dict]) -> set[int]:
    """Expand ranges to a set of line numbers."""
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


def run_evaluation():
    """Run the full evaluation and return results."""
    load_dotenv()
    pool = create_pool()

    try:
        pages = _fetch_ground_truth(pool)
    finally:
        pool.close()

    if not pages:
        print("No validated annotations found in DB.")
        return []

    print(f"\nEvaluating {len(pages)} pages...\n")
    results = []

    for i, page in enumerate(pages):
        page_id = page["id"]
        url = page["url"]
        markdown = page["markdown"]
        truth_ranges = page["ranges"] or []

        # Use stored markdown directly
        predicted_ranges = annotate_page(markdown)

        if predicted_ranges is None:
            print(f"  Page {page_id}: FAILED (API error)")
            results.append({
                "page_id": page_id,
                "url": url,
                "predicted": None,
                "expected": truth_ranges,
                "metrics": None,
            })
        else:
            truth_lines = _ranges_to_line_set(truth_ranges)
            pred_lines = _ranges_to_line_set(predicted_ranges)
            metrics = _compute_metrics(pred_lines, truth_lines)

            results.append({
                "page_id": page_id,
                "url": url,
                "predicted": predicted_ranges,
                "expected": truth_ranges,
                "metrics": metrics,
            })

        # Rate limiting delay (skip after last page)
        if i < len(pages) - 1:
            time.sleep(CALL_DELAY)

    _print_summary(results)
    return results


def _print_summary(results: list[dict]):
    """Print a summary table of results."""
    print("\n" + "=" * 90)
    print(f"{'ID':>5}  {'IoU':>6}  {'Prec':>6}  {'Rec':>6}  {'Pred':>15}  {'Truth':>15}  URL")
    print("-" * 90)

    successful = []
    for r in results:
        page_id = r["page_id"]
        url = r["url"][:40]
        expected = _fmt_ranges(r["expected"])

        if r["metrics"] is None:
            print(f"{page_id:>5}  {'FAIL':>6}  {'':>6}  {'':>6}  {'ERROR':>15}  {expected:>15}  {url}")
        else:
            m = r["metrics"]
            predicted = _fmt_ranges(r["predicted"])
            print(
                f"{page_id:>5}  {m['iou']:>6.3f}  {m['precision']:>6.3f}  "
                f"{m['recall']:>6.3f}  {predicted:>15}  {expected:>15}  {url}"
            )
            successful.append(m)

    print("-" * 90)

    if successful:
        mean_iou = sum(m["iou"] for m in successful) / len(successful)
        mean_prec = sum(m["precision"] for m in successful) / len(successful)
        mean_rec = sum(m["recall"] for m in successful) / len(successful)
        print(
            f"{'MEAN':>5}  {mean_iou:>6.3f}  {mean_prec:>6.3f}  "
            f"{mean_rec:>6.3f}  ({len(successful)}/{len(results)} succeeded)"
        )
    else:
        print("No successful evaluations.")

    print("=" * 90)


def _fmt_ranges(ranges: list[dict] | None) -> str:
    """Format ranges compactly for the summary table."""
    if ranges is None:
        return "N/A"
    if not ranges:
        return "[]"
    parts = [f"{r['start']}-{r['end']}" for r in ranges]
    s = ", ".join(parts)
    return s if len(s) <= 15 else s[:12] + "..."


# -- pytest entry point --

def test_llm_accuracy():
    """Pytest-compatible entry point for the evaluation."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        import pytest
        pytest.skip("OPENROUTER_API_KEY not set")

    results = run_evaluation()
    successful = [r for r in results if r["metrics"] is not None]

    assert len(successful) > 0, "No pages were successfully evaluated"

    mean_iou = sum(r["metrics"]["iou"] for r in successful) / len(successful)
    print(f"\nMean IoU: {mean_iou:.3f}")


# -- standalone entry point --

if __name__ == "__main__":
    run_evaluation()
