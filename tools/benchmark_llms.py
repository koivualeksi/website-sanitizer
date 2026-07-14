"""Benchmark API LLMs on the fine-tuning test set.

Sends every page of data/test.jsonl through the same annotation prompt the
auto-annotation pipeline uses and scores predictions against ground truth
with the same metrics as the fine-tuned models (per-page IoU/P/R; honest
mean counts unparseable responses as IoU 0). Establishes the big-model
upper bar for the model ladder.

Caveat when reading results: the bulk annotators (MODELS_BULK in
tools/auto_annotation/config.py) helped produce the labels, so those exact
models score partly against their own opinions.

Usage:
  python -m tools.benchmark_llms --models google/gemini-2.5-flash
  python -m tools.benchmark_llms --models a,b,c --concurrency 6
  python -m tools.benchmark_llms --models a --max-samples 5   # smoke

Results stream to output/benchmarks/bench_<model>.json (safe to re-run;
completed pages are skipped, failed ones retried).
"""

import argparse
import asyncio
import json
import os
import re
import time

import httpx
from dotenv import load_dotenv

from tools.auto_annotation.annotator import _extract_json, _get_max_line, _validate_ranges
from tools.auto_annotation.config import OPENROUTER_URL
from tools.auto_annotation.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark API LLMs on the test set")
    p.add_argument("--models", required=True,
                   help="Comma-separated OpenRouter model ids")
    p.add_argument("--data-file", default=os.path.join(REPO_ROOT, "data", "test.jsonl"))
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "output", "benchmarks"))
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=0,
                   help="Limit pages (0 = all); for smoke tests")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="Completion cap (reasoning models spend thinking tokens)")
    return p.parse_args()


def load_test_pages(path, limit=0):
    pages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            md = row["messages"][1]["content"]
            truth = set()
            for m in re.finditer(r"(\d+):(\d+)", row["messages"][2]["content"]):
                truth.update(range(int(m.group(1)), int(m.group(2)) + 1))
            pages.append({"page_id": row["page_id"], "md": md,
                          "truth": sorted(truth)})
    return pages[:limit] if limit > 0 else pages


def score(pred_lines, truth_lines):
    pred, truth = set(pred_lines), set(truth_lines)
    if not pred and not truth:
        return {"iou": 1.0, "precision": 1.0, "recall": 1.0}
    inter, union = len(pred & truth), len(pred | truth)
    return {"iou": inter / union if union else 0.0,
            "precision": inter / len(pred) if pred else 0.0,
            "recall": inter / len(truth) if truth else 0.0}


async def annotate(client, sem, model, page, api_key, retries, max_tokens):
    """One page -> result dict. metrics=None marks an unusable response."""
    max_line = _get_max_line(page["md"])
    if max_line == 0:  # empty scrape: no lines to send, predict nothing
        return {"page_id": page["page_id"], "response": "",
                "metrics": score([], page["truth"]), "usage": {}}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": USER_PROMPT_TEMPLATE.format(structured_markdown=page["md"])},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        # normalized param; ignored by non-reasoning models
        "reasoning": {"effort": "low"},
        "usage": {"include": True},  # cost accounting in the response
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    content, usage, error = None, {}, None
    for attempt in range(retries):
        try:
            async with sem:
                r = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            if r.status_code == 429 or r.status_code >= 500:
                error = f"http {r.status_code}"
                await asyncio.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            body = r.json()
            choices = body.get("choices") or []
            content = (choices[0].get("message", {}) or {}).get("content") if choices else None
            usage = body.get("usage") or {}
            if content:
                break
            error = body.get("error", {}).get("message") or "empty content"
            await asyncio.sleep(2 * (attempt + 1))
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            error = f"{type(e).__name__}: {e}"
            await asyncio.sleep(5 * (attempt + 1))

    if not content:
        return {"page_id": page["page_id"], "response": None,
                "metrics": None, "usage": usage, "error": error}

    try:
        ranges = _validate_ranges(json.loads(_extract_json(content)), max_line)
    except (json.JSONDecodeError, ValueError) as e:
        return {"page_id": page["page_id"], "response": content[:2000],
                "metrics": None, "usage": usage, "error": f"parse: {e}"}

    pred = set()
    for rg in ranges:
        pred.update(range(rg["start"], rg["end"] + 1))
    return {"page_id": page["page_id"], "response": content[:2000],
            "metrics": score(pred, page["truth"]), "usage": usage,
            "ranges": ranges}


def summarize(results, n_pages):
    ok = [r["metrics"] for r in results if r["metrics"] is not None]
    n_fail = len(results) - len(ok)
    honest = sum(m["iou"] for m in ok) / n_pages if n_pages else 0.0
    out = {"pages": n_pages, "failures": n_fail,
           "iou_honest": round(honest, 4),
           "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) or 0
                                for r in results),
           "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) or 0
                                    for r in results),
           "cost_usd": round(sum(r.get("usage", {}).get("cost", 0) or 0
                                 for r in results), 4)}
    if ok:
        out["iou_excl_failures"] = round(sum(m["iou"] for m in ok) / len(ok), 4)
        out["precision"] = round(sum(m["precision"] for m in ok) / len(ok), 4)
        out["recall"] = round(sum(m["recall"] for m in ok) / len(ok), 4)
    return out


async def run_model(model, pages, args, api_key):
    slug = model.replace("/", "_").replace(":", "_")
    out_path = os.path.join(args.output_dir, f"bench_{slug}.json")
    done = {}
    if os.path.exists(out_path):
        prior = json.load(open(out_path, encoding="utf-8"))
        # keep scored pages, retry failures
        done = {r["page_id"]: r for r in prior.get("results", [])
                if r.get("metrics") is not None}
    todo = [p for p in pages if p["page_id"] not in done]
    print(f"\n=== {model}: {len(todo)} pages to run, {len(done)} cached ===")

    results = list(done.values())
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    def save():
        results.sort(key=lambda r: r["page_id"])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"model": model,
                       "summary": summarize(results, len(pages)),
                       "results": results}, f, indent=1)

    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, read=300.0)) as client:
        tasks = [annotate(client, sem, model, p, api_key, args.retries,
                          args.max_tokens) for p in todo]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            results.append(await coro)
            if (i + 1) % 25 == 0:
                save()
                s = summarize(results, len(pages))
                print(f"  [{i + 1}/{len(todo)}] running honest IoU "
                      f"{s['iou_honest']:.3f} ({s['failures']} fails, "
                      f"{time.time() - t0:.0f}s)")
    save()
    s = summarize(results, len(pages))
    print(f"{model}: honest IoU {s['iou_honest']:.4f}  "
          f"excl-fails {s.get('iou_excl_failures', 0):.4f}  "
          f"P {s.get('precision', 0):.4f}  R {s.get('recall', 0):.4f}  "
          f"fails {s['failures']}/{s['pages']}  "
          f"tokens {s['prompt_tokens']/1e6:.2f}M in / "
          f"{s['completion_tokens']/1e3:.0f}k out  cost ${s['cost_usd']:.2f}")
    return s


async def main_async(args):
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    assert api_key, "OPENROUTER_API_KEY is not set"
    os.makedirs(args.output_dir, exist_ok=True)

    pages = load_test_pages(args.data_file, args.max_samples)
    print(f"{len(pages)} test pages")

    summaries = {}
    for model in [m.strip() for m in args.models.split(",")]:
        summaries[model] = await run_model(model, pages, args, api_key)

    print("\n" + "=" * 78)
    print(f"{'model':<42s} {'honest':>7s} {'excl':>7s} {'prec':>7s} "
          f"{'rec':>7s} {'fails':>6s}")
    print("-" * 78)
    for model, s in summaries.items():
        print(f"{model:<42s} {s['iou_honest']:>7.4f} "
              f"{s.get('iou_excl_failures', 0):>7.4f} "
              f"{s.get('precision', 0):>7.4f} {s.get('recall', 0):>7.4f} "
              f"{s['failures']:>6d}")


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
