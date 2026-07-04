# Feature Reference

56 features per content line for CNN/BiGRU/XGBoost classifiers (32 base + 3 style-group + 9 new + 6 gap-analysis + 4 element-context + 2 derived in notebook).

Export: `python -m tools.export_features`

## Positional (3)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `position_pct` | float 0–1 | `line_num / total_lines`. Content clusters mid-page, boilerplate at edges. |
| 2 | `total_lines` | int | Total content lines in the page. Short pages are often all-content. |
| 3 | `depth` | int | Raw DOM nesting depth from the converter walk. Deeper = often widgets/sidebars. |

## Structural ancestry (8)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 4 | `parent_tag_idx` | int 0–8 | Nearest structural ancestor, label-encoded. Mapping: body=0, header=1, nav=2, main=3, article=4, section=5, footer=6, aside=7, form=8. |
| 5 | `in_header` | binary | Any ancestor is `<header>`. Almost always boilerplate. |
| 6 | `in_nav` | binary | Any ancestor is `<nav>`. Nearly always boilerplate. |
| 7 | `in_main` | binary | Any ancestor is `<main>`. Strongest single content signal. |
| 8 | `in_article` | binary | Any ancestor is `<article>`. Strong content signal. |
| 9 | `in_footer` | binary | Any ancestor is `<footer>`. Almost always boilerplate. |
| 10 | `in_aside` | binary | Any ancestor is `<aside>`. Sidebars, usually boilerplate. |
| 11 | `in_form` | binary | Any ancestor is `<form>`. Login forms, search, cookie consent. |

A line can have multiple `in_*` flags active (e.g. `in_main=1` and `in_article=1` for `[main] > [article] > [section]`).

## Text content (8)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 12 | `text_length` | int | Character count of the line text. Content paragraphs: 100–500+, nav items: 5–30. |
| 13 | `word_count` | int | Number of whitespace-separated words. |
| 14 | `link_ratio` | float 0–1 | Fraction of text inside `[...]()` markdown links. Nav lines ~1.0, content ~0.0. |
| 15 | `is_link_only` | binary | Line contains nothing but markdown links. |
| 16 | `is_heading` | binary | Line starts with `#` (markdown heading). |
| 17 | `heading_level` | int 0–6 | Number of `#` characters. 0 if not a heading. Main titles tend to be h1/h2. |
| 18 | `has_bold` | binary | Contains `**...**` markup. Often lead paragraphs or emphasis in articles. |
| 19 | `is_list_item` | binary | Line starts with `- ` or `N. ` (any ordered list index). Could be nav list or content list. |

## Collapsed item flags (3)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 20 | `is_link_summary` | binary | Row represents a collapsed run of 3+ consecutive link-only lines. Always boilerplate. |
| 21 | `is_cookie_summary` | binary | Row represents collapsed cookie banner lines. |
| — | `span_lines` | int | How many actual lines this row represents. 1 for normal text, N for summaries. Used in IoU evaluation to expand back to line numbers. Not a model feature in FEATURE_COLS — metadata only. |

## Text statistics (3)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 22 | `punctuation_ratio` | float 0–1 | Fraction of characters that are punctuation (`.,;:!?` etc.). Content prose ~0.05, nav labels ~0.0. |
| 23 | `sentence_count` | int | Count of sentence-ending patterns (`.` `?` `!` followed by space or end). Content paragraphs: 2–5, nav items: 0. |
| 24 | `avg_word_length` | float | Mean character length of words. Content has longer words than nav labels. |

## Text uniqueness (3)

Page-level IDF (within-page term frequency) plus corpus-level line frequency. Captures whether text is unique content or repeated boilerplate.

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 25 | `mean_idf` | float | Mean page-level IDF of words in the line. `IDF(word) = log(lines_on_page / lines_containing_word)`. High = word appears in few lines on this page (unique content), low = word appears in many lines (repeated nav/footer text). |
| 26 | `max_idf` | float | Page-level IDF of the rarest word in the line. Captures a single distinctive term even in an otherwise common line. |
| 27 | `line_frequency` | float 0–1 | Fraction of training pages containing this exact line text. "Hyppaa sisaltoon" on 200/1386 pages = 0.14. Unique article text = ~0.001. |

## Window context (2)

Rolling averages over a 5-line window centered on the current line. Captures local neighbourhood patterns.

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 28 | `mean_text_length_w5` | float | Average `text_length` in 5-line window. Content regions have consistently long lines. |
| 29 | `mean_link_ratio_w5` | float | Average `link_ratio` in 5-line window. Nav regions cluster high. |

## Block-level (3)

A "block" is a contiguous run of lines sharing the same `parent_tag` and `depth`.

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 30 | `block_size` | int | Number of lines in this structural block. Large blocks = likely content. |
| 31 | `block_text_density` | float | Average `text_length` across the block. Dense blocks = content. |
| 32 | `block_link_density` | float 0–1 | Total link characters / total text characters in the block. High = nav block. |

## Style-group (3)

CSS class-based grouping. Each line is assigned to its nearest styled ancestor's CSS class. Small groups (< 3 lines) are iteratively merged into their parent's class. Per-group aggregates are then computed.

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 33 | `style_group_size` | int | Number of lines sharing this CSS class group. Large groups = consistent styling (nav bars, article body). |
| 34 | `style_group_link_density` | float 0–1 | Mean `link_ratio` across the style group. High = nav/footer style class. |
| 35 | `style_group_mean_words` | float | Mean `word_count` across the style group. Content groups have higher word counts. |

## Derived (2)

Computed from existing columns in the notebook, not in the exported CSV.

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 36 | `tag_transition` | binary | 1 if `parent_tag` differs from the previous line (within same page). Marks structural boundaries where content/boilerplate regions switch. |
| 37 | `dist_to_heading` | int | Number of lines since the last heading line (within same page). Content tends to cluster near headings; boilerplate drifts far. |

## Single-feature ranking (10-epoch CNN, individual predictive power)

| Rank | Feature | IoU | Notes |
|------|---------|-----|-------|
| 1 | link_ratio | 0.592 | Best single predictor |
| 2 | mean_link_ratio_w5 | 0.565 | |
| 3 | avg_word_length | 0.559 | |
| 4 | is_link_only | 0.556 | |
| 5 | block_link_density | 0.547 | |
| 6 | is_list_item | 0.529 | |
| 7 | punctuation_ratio | 0.529 | |
| 8 | in_main | 0.525 | |
| 9 | parent_tag_idx | 0.521 | |
| 10 | in_nav | 0.513 | |
| 33 | in_article | 0.184 | Weak alone, strong in combination |
| 34 | has_bold | 0.165 | Weakest individual signal |

## Model results

| Model | N features | IoU | Prec | Rec | Notes |
|-------|-----------|-----|------|-----|-------|
| CNN | 17 (baseline) | 0.810 | 0.834 | 0.957 | |
| CNN | 20 (17 + IDF) | 0.815 | 0.859 | 0.930 | |
| CNN | 15 (greedy from 17) | 0.836 | 0.849 | 0.969 | confirmed 30 epochs |
| CNN | 15 (multistart #1) | 0.855 | — | — | search score |
| CNN | 21 (multistart best) | 0.872 | 0.892 | 0.979 | confirmed 30 epochs, best at ep15 |
| XGBoost | 27 | 0.809 | 0.823 | 0.968 | |
| **BiGRU** | **23 (feature search)** | **0.894** | **0.923** | **0.957** | 80/20 split (276 test), 3-seed confirmed |

## Multi-start feature selection (CNN, 10 starts x 15 random features)

Config: 10 random starting sets of 15 features, backward/forward cycling with 15-epoch trials, seed=42. Total search time: 299 min. Best set confirmed with 30-epoch run.

### All converged sets (sorted by IoU)

| Start | N feat | IoU | Time | Cycles |
|-------|--------|-----|------|--------|
| 4 | 21 | 0.8689 | 2695s | 3 |
| 9 | 22 | 0.8624 | 2452s | 2 |
| 1 | 15 | 0.8554 | 1972s | 2 |
| 7 | 13 | 0.8540 | 2277s | 2 |
| 10 | 16 | 0.8480 | 2318s | 2 |
| 5 | 19 | 0.8455 | 1634s | 1 |
| 3 | 14 | 0.8293 | 609s | 1 |
| 6 | 18 | 0.8224 | 1278s | 1 |
| 8 | 19 | 0.8218 | 1734s | 2 |
| 2 | 13 | 0.7909 | 958s | 1 |

Median IoU: 0.847, mean: 0.840. IoU range: 0.791–0.869. Feature set size range: 13–22.

### Best set (start 4, 21 features) — confirmed IoU 0.8719

`in_main`, `sentence_count`, `mean_text_length_w5`, `in_footer`, `total_lines`, `avg_word_length`, `is_link_only`, `has_bold`, `parent_tag_idx`, `word_count`, `depth`, `is_heading`, `is_list_item`, `dist_to_heading`, `tag_transition`, `position_pct`, `block_link_density`, `in_header`, `punctuation_ratio`, `is_link_summary`, `link_ratio`

30-epoch confirmation: IoU=0.872, P=0.892, R=0.979 (peak at epoch 15; slight overfit after).

### Feature frequency across 10 converged sets

| Feature | Count | Pct |
|---------|-------|-----|
| `in_main` | 10 | 100% |
| `in_footer` | 9 | 90% |
| `position_pct` | 9 | 90% |
| `in_header` | 7 | 70% |
| `depth` | 7 | 70% |
| `link_ratio` | 7 | 70% |
| `has_bold` | 6 | 60% |
| `block_text_density` | 6 | 60% |
| `tag_transition` | 6 | 60% |
| `word_count` | 6 | 60% |
| `is_heading` | 6 | 60% |
| `block_size` | 6 | 60% |
| `punctuation_ratio` | 5 | 50% |
| `block_link_density` | 5 | 50% |
| `is_list_item` | 5 | 50% |
| `in_article` | 5 | 50% |
| `sentence_count` | 5 | 50% |
| `in_form` | 5 | 50% |
| `avg_word_length` | 5 | 50% |
| `is_link_only` | 5 | 50% |
| `dist_to_heading` | 4 | 40% |
| `parent_tag_idx` | 4 | 40% |
| `in_nav` | 4 | 40% |
| `text_length` | 4 | 40% |
| `mean_link_ratio_w5` | 4 | 40% |
| `mean_text_length_w5` | 4 | 40% |
| `heading_level` | 3 | 30% |
| `is_link_summary` | 3 | 30% |
| `mean_idf` | 3 | 30% |
| `total_lines` | 3 | 30% |
| `is_cookie_summary` | 3 | 30% |
| `line_frequency` | 2 | 20% |
| `max_idf` | 2 | 20% |
| `in_aside` | 2 | 20% |

### Key observations

1. **`in_main` is the only universal feature** — selected by all 10 starts. `in_footer` and `position_pct` at 90% are near-universal.
2. **Structural ancestry dominates the top.** The 3 most frequent features are all structural/positional, confirming semantic HTML containers are the strongest signal.
3. **IDF features are weak and inconsistent.** `mean_idf` (30%), `max_idf` (20%), `line_frequency` (20%) — frequently dropped during backward pass. The corpus-level uniqueness signal doesn't help much when structural cues are present.
4. **Feature set size doesn't correlate with IoU.** The best (start 4, 21 feat) and second-best (start 9, 22 feat) are large, but start 7 hits 0.854 with only 13 features. Diminishing returns above ~15.
5. **Multiple cycles help.** The top 5 sets all needed 2–3 backward/forward cycles, while the bottom 5 converged in 1 cycle. Second-pass forward additions after backward pruning find complementary features.
6. **Local optima are real.** IoU spread of 0.078 across 10 starts. Multi-start was worthwhile — single greedy from one random set could land anywhere in the 0.79–0.87 range.

## New features — single-feature ranking (10-epoch CNN)

| Rank | Feature | IoU | P | R | Notes |
|------|---------|-----|---|---|-------|
| 1 | sibling_text_variance | 0.530 | 0.574 | 0.846 | Best new feature, on par with is_list_item |
| 2 | ancestor_depth_ratio | 0.448 | 0.514 | 0.826 | |
| 3 | cumulative_text_pct | 0.431 | 0.481 | 0.884 | |
| 4 | uppercase_ratio | 0.389 | 0.456 | 0.767 | |
| 5 | number_ratio | 0.344 | 0.456 | 0.536 | |
| 6 | relative_position_in_block | 0.340 | 0.423 | 0.538 | |
| 7 | type_token_ratio | 0.337 | 0.500 | 0.518 | |

## Core 6 search (single-vote, noisy — needs 5-vote rerun)

Started from 6 core features (in_main, in_footer, position_pct, in_header, depth, link_ratio). Forward/backward cycling with 15-epoch single-vote trials. High noise (~0.02 IoU variance per run) limited reliability.

| Step | N feat | IoU | Feature |
|------|--------|-----|---------|
| start | 6 | 0.819 | — |
| C1 add | 7 | 0.826 | in_form |
| C1 add | 8 | 0.846 | ancestor_depth_ratio |
| C2 add | 9 | 0.850 | text_length |
| converged | 9 | 0.850 | — |

Confirmed: the 9-feature set underperforms the 21-feature multistart best (0.872). Single-vote noise caused premature convergence — search couldn't reliably detect +0.01 improvements. Needs 5-vote averaging on GPU for trustworthy results.

## New features (implemented)

Language-agnostic, no regex word lists. All computable from existing data.

### Character-level (2)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 38 | `uppercase_ratio` | float 0–1 | `uppercase_alpha / total_alpha`. Prose ~3-5%, nav/CTA/cookie banners often 15-100%. Language-agnostic. |
| 39 | `number_ratio` | float 0–1 | `digit_chars / text_length`. Pagination, phone numbers, copyright years, price grids. Prose has low digit density. |

### Block-level (2)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 40 | `relative_position_in_block` | float 0–1 | `line_index_in_block / (block_size - 1)`. First line of `[main]` ≠ last line. Current block features say "big block" but not where in it. 0 for single-line blocks. |
| 41 | `sibling_text_variance` | float | Coefficient of variation (`std / mean`) of `text_length` within the block. Content blocks are heterogeneous (headings + paragraphs), nav blocks are uniform. |

### Structural (1)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 42 | `ancestor_depth_ratio` | float 0–1 | `len(struct_stack) / raw_depth`. High = richly semantic path (`[main]>[article]>[section]`), low = div-soup widget. Different from `depth` + `in_*` flags — captures nesting richness as a ratio. |

### Text diversity (1)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 43 | `type_token_ratio` | float 0–1 | `unique_tokens / total_tokens`. Lexical diversity within a line. Content uses varied vocabulary, boilerplate repeats ("Click here"). Corpus-independent — no IDF table needed. |

### Word novelty (2)

Page-level word importance based on within-page frequency. Unique content words score high, repeated boilerplate words score low.

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 44 | `word_novelty` | float 0–1 | Mean `1 / count(word on page)` across words in the line. Unique content words → 1.0, repeated boilerplate → low. |
| 45 | `word_novelty_sum` | float | Sum of `1 / count(word on page)` across words. Ranked #4 in BiGRU permutation importance. |

### Positional (1)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 46 | `cumulative_text_pct` | float 0–1 | Cumulative character count up to this line / total characters on page. Unlike `position_pct` (line-based), this is content-weighted — jumps fast through dense paragraphs, crawls through short nav items. |

### CSS class name patterns (3)

Readability/trafilatura-style regex matching against the full ancestor CSS class chain. Captures boilerplate signals from class names that semantic HTML tags miss (e.g. `<div class="sidebar">` vs `<aside>`).

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 47 | `has_positive_class` | binary | Any ancestor class matches content-positive regex: `article\|content\|entry\|post\|story\|text\|body\|main\|page\|blog`. r=+0.04 — weak alone (91% of lines match). |
| 48 | `has_negative_class` | binary | Any ancestor class matches boilerplate-negative regex: `hidden\|banner\|comment\|footer\|sidebar\|widget\|ad\|share\|promo\|sponsor` etc. r=-0.16, low redundancy (max Spearman 0.18 with existing). |
| 49 | `has_unlikely_class` | binary | Any ancestor class matches broad boilerplate regex: `menu\|header\|footer\|social\|breadcrumbs\|popup\|pagination\|sidebar\|comment\|banner` etc. r=-0.46 — strongest new feature, would rank #8 overall. |

### Element content flags (2)

Whether the text-bearing element contains embedded media or form controls. Inspired by Readability's conditional cleaning (list-vs-paragraph, input count heuristics).

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 50 | `has_image` | binary | Element or descendants contain `<img>`. Content images vs decorative icons. |
| 51 | `has_input` | binary | Element or descendants contain `<input>`, `<select>`, or `<textarea>`. Form-heavy regions are typically boilerplate. |

### Element context flags (4)

Tag-level structural context not captured by the `in_*` structural ancestry flags (which only track `_STRUCTURAL_TAGS`).

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 52 | `is_button` | binary | Element is `<button>` or inside a `<button>`. 58% of buttons are outside form/nav — cookie banners, filters, CTAs. |
| 53 | `has_aria_hidden` | binary | Element or ancestor has `aria-hidden="true"`. 7% of text nodes — invisible content leaking through the converter. |
| 54 | `in_table` | binary | Element is inside `<table>`. 4.2% of text — product specs, data tables, or layout boilerplate. |
| 55 | `in_details` | binary | Element is inside `<details>`/`<summary>`. 2.1% of text — FAQ accordions, collapsible sections. |

### Text statistics — additional (1)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 56 | `comma_count` | int | Count of commas (ASCII + fullwidth). Readability uses comma count as a prose signal (+1 score per comma). r=+0.19, content mean 0.59 vs boilerplate 0.07 (8.76x ratio). |

## Notes

- `span_lines` is in the CSV but NOT in `FEATURE_COLS` — it's metadata for IoU evaluation only.
- `parent_tag` (string) is in the CSV but replaced by `parent_tag_idx` (int) in `FEATURE_COLS`.
- IDF table is saved to `data/idf_table.json` for use at inference time.
- All features are normalised (zero-mean, unit-variance) using training set statistics before feeding to CNN/BiLSTM.
