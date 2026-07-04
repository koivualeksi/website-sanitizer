# Guidelines

## Philosophy

Think before writing. Surface assumptions and tradeoffs rather than guessing.

## Principles

1. **Clarify before coding.** If a task has ambiguous scope, competing approaches, or hidden tradeoffs — ask. A 30-second clarification beats a 30-minute redo.

2. **Minimum viable change.** Minimum code that solves the problem. Nothing speculative. No "while we're here" refactors, no extra abstractions, no future-proofing. Three similar lines > premature helper.

3. **Surgical edits.** Touch only what the task requires. Don't reformat, rename, or reorganise neighbouring code. Don't add docstrings or type hints to unchanged code.

4. **Verify success.** Define what "done" looks like before starting. Run the relevant test/check and loop until it passes. If there's no test, say so.

5. **Spare comments.** No boilerplate file headers, no restating obvious code. Comment only non-obvious behaviour or business rules.

6. **Background long tasks.** Anything over ~30 seconds (builds, large test suites, batch jobs) should run in background with logging.

For trivial tasks, use judgement — not every one-liner needs a discussion.

## Project structure

- `scraper/` — HTML fetcher + converter (HTML → structured markdown)
- `tools/` — CLI utilities
  - `auto_annotation/` — LLM annotation prompt, OpenRouter client, bulk annotate
  - `export_data.py` — Export train/test JSONL for LLM fine-tuning
  - `export_features.py` — Export per-line feature CSVs for BiGRU/CNN
  - `regenerate_markdown.py` — Regenerate all stored markdown
- `models/` — ML model training
  - `qwen/` — Qwen fine-tuning (SFT + GRPO)
  - `bigru/` — BiGRU training notebooks
- `ui/` — FastAPI annotation UI (Jinja2 + vanilla JS)
- `db/` — PostgreSQL connection pool, queries, and init.sql
- `tests/` — Evaluation suite (LLM accuracy vs ground truth)
- `data/` — CSVs, JSONL, model weights

## Key conventions

- Python 3.12+, BeautifulSoup4, psycopg3, httpx, FastAPI
- PostgreSQL in Docker (`docker-compose.yml`)
- `.env` for secrets (OPENROUTER_API_KEY, DB connection)
- Sequential line numbers in converter output (1, 2, 3...)
- Annotations store `{start, end}` line ranges referencing markdown line numbers
- After converter changes: run `python -m tools.regenerate_markdown --confirm`
