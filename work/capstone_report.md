# Capstone Report

* **Author:** Saqib Khan
* **Lane:** Refresh / Content Opportunity Scoring (Lane 2)
* **Repo:** https://github.com/saqib0-cpu/flyrank-ml-internship
* **Date:** 2026-08-03

---

## 1. Problem framing

**Unit of analysis:** a single content item (`content_hash_id`), aggregated across two fixed time windows.

**Output:** a status label (growing / declining / recovering / stagnant / volatile) plus a mapped action (protect / rewrite / monitor / improve / review), ranked by click-momentum magnitude.

**Decision this supports:** which pages a content editor should review first, out of a portfolio too large to manually audit every cycle. The action a human takes from the output is triage — deciding where to spend limited rewrite/review effort this sprint, not a fully automated content decision.

**Cost of a wrong call:** a false "growing" (missed decline) means a page silently loses traffic before anyone notices — a delayed-cost error. A false "declining" (unnecessary rewrite flag) wastes editor time on a page that didn't need it — a wasted-effort error. These costs are asymmetric but both non-trivial, which is why the system is framed as a ranked priority list rather than a hard automated action.

**Why ML/data helps here:** with 500K+ content items and 78.8M daily rows, no team can manually trend-check every page every cycle. A repeatable score turns an unbounded manual review problem into a ranked, bounded weekly task.

---

## 2. Data safety

**Data used:**
- `fact_content_daily_performance` (78,835,655 rows) — daily clicks, impressions, position per content item, filtered to `gsc_data_available = True`.
- `dim_content` (519,606 rows) — content attributes: type, search volume, competition level, word count, publish status.
- `dim_clients` (104 rows) — used only to confirm valid GSC access; no client attributes used as model features.

**Columns deliberately excluded:**
- `client_hash_id` is used **only for grouping**, never as a model feature — including it as a feature would let the model learn client identity rather than content behavior, and would violate the "no client-identifying signal in output" rule.
- `sessions_ai` and related AI-referral columns (`ai_chatgpt`, `ai_claude`, etc.) were excluded from modeling — these are sparse in this window and the brief explicitly warns against treating sparse AI-referral sessions as a reliable classification signal.
- Raw `report_date` was not used directly as a feature; only window-aggregated deltas were used, to avoid the model keying on absolute calendar dates.

**Leakage risks considered:**
- The target label (`will_decline`) is derived from **Window C (2026-02 to 2026-04)** — a period strictly after both feature windows (A: 2025-08–10, B: 2025-11–2026-01). No feature is computed using any data from Window C.
- No label-derived field (e.g., a precomputed `trend_direction` or `trend_pct` from the source data) was used as an input feature; all trend features were computed independently from raw clicks/impressions/position.
- Pseudonymous IDs (`content_hash_id`, `client_hash_id`) are used strictly for joins and grouping, never passed into the feature matrix.

**Confirmation:** no client name, domain, URL, raw query text, or credential appears anywhere in `work/`, the capstone notebook, or the deployed paper. All identifiers shown are the dataset's own salted hash keys.

---

## 3. Baseline

**The baseline** is a transparent, rule-based classifier using only two signals already visible to a human editor without any modeling:

- `clicks_delta_pct` > 20% and `position_delta` > 0 → **growing**
- `clicks_delta_pct` < -20% and `position_delta` < 0 → **declining**
- `clicks_delta_pct` > 10% despite worse position → **recovering**
- `|clicks_delta_pct|` ≤ 10% → **stagnant**
- everything else → **volatile**

**Why this is a fair comparison:** it uses the exact same Window A vs. Window B feature basis as the ML model, is evaluated on the identical held-out test rows, and requires no training — it represents what an editor would likely do by eyeballing a trend report today.

**Baseline result (declining class, same test split as the model):**
- **F1 = 0.154**

This low score shows the momentum-only rule catches only a small fraction of true future declines — most declines in this data show up first as a CTR problem, not a clicks/position swing large enough to trip the rule thresholds.

---

## 4. Model / analysis

**Method:** `GradientBoostingClassifier` (scikit-learn), chosen because the feature set is small (7 features), tabular, and mixes continuous and ordinal-encoded categorical signals — a setting where gradient boosting reliably outperforms linear baselines without needing large data volumes or heavy tuning.

**Exact feature list used:**
`clicks_delta_pct`, `position_delta`, `ctr_b`, `ctr_a`, `search_volume`, `competition_level` (ordinal-encoded), `word_count`

**Left out on purpose:** `sessions_ai` (too sparse per brief guidance), `client_hash_id` (identity leakage risk), raw `report_date` (calendar leakage risk), and any field directly computed from Window C.

**Target/proxy definition (one sentence):** `will_decline` = 1 if a content item's clicks fell by more than 15% from Window B (2025-11–2026-01) to Window C (2026-02–2026-04), else 0 — a forward-looking proxy for meaningful traffic decay, not a measurement of ranking-algorithm behavior.

---

## 5. Evaluation

**Split:** stratified random 80/20 split on the feature matrix. A grouped-by-client or additional time-based split was not layered on top of this, because the leakage boundary is already enforced at the **label-definition** stage — the label itself only exists using data from a window strictly after the features. Splitting the *rows* by client or time further would reduce usable signal without adding leakage protection, since no row's features and label ever share a time window.

**Metrics, model vs. baseline (same test split, n = 29,331):**

| | Baseline (rule-based) | Model (gradient boosting) |
|---|---|---|
| F1 (declining class) | **0.154** | **0.60** |
| Precision (declining) | — | 0.70 |
| Recall (declining) | — | 0.52 |
| Overall accuracy | — | 88% |

**Base rate check (required honesty step):** the `will_decline` class makes up **16.5%** of the test set (4,849 of 29,331 rows). A trivial always-predict-"no decline" classifier would already reach **83.5%** accuracy. The model's 88% accuracy is therefore only about **4.5 points above the majority-class base rate** — the F1 score (0.60 vs. 0.154 baseline) is the more honest discrimination measure here, not the headline accuracy number.

**Error analysis:** recall (0.52) is meaningfully lower than precision (0.70) for the declining class — the model misses roughly half of true future declines rather than over-flagging healthy pages. In practice this means the ranked list is a reasonably trustworthy "top of the list is genuinely at risk" signal, but it should not be read as an exhaustive list of every page that will decline — a meaningful share of true declines fall below the model's detection threshold and would still need periodic manual spot-checks outside the ranked list.

---

## 6. Interpretation

**What the model found, in plain words:** recent click-through rate (`ctr_b`) is by far the strongest signal of future decline — it alone accounts for **77.9%** of the model's decision weight, dwarfing ranking-position movement (4.4%) and click momentum (3.7%), which the rule-based baseline relies on almost entirely.

**Full importance ranking:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `ctr_b` (recent CTR) | 77.9% |
| 2 | `word_count` | 11.0% |
| 3 | `position_delta` | 4.4% |
| 4 | `clicks_delta_pct` | 3.7% |
| 5 | `competition_level` | 1.8% |
| 6 | `ctr_a` (prior CTR) | 0.9% |
| 7 | `search_volume` | 0.3% |

**Surprise / negative result:** `search_volume` and `competition_level` — both intuitively "important SEO signals" — contributed almost nothing to the model's decisions (0.3% and 1.8% respectively). This is a well-understood negative result worth reporting as-is: within this dataset and window, a page's static market context matters far less to predicting decline than its own recent click efficiency.

**Caveat on the top feature:** `ctr_b`'s outsized weight deserves scrutiny rather than uncritical acceptance, since it is measured in the same window used to compute other features — its dominance may partly reflect that shared window rather than a uniquely superior signal. This is flagged rather than presented as a clean, final result.

---

## 7. Recommendation

Ranked action playbook a FlyRank editor could apply tomorrow, directly from the model's output list:

1. **Protect, don't touch:** pages the model/baseline both classify as growing with strong position gains — leave alone, monitor only.
2. **Rewrite the top of the ranked decline list first:** pages ranked highest by predicted decline probability, prioritized because recall is only 0.52 — the highest-confidence flags are the ones worth acting on first.
3. **Check CTR before rewriting content:** given `ctr_b`'s dominance, a stagnant page with a large CTR gap is often a title/metadata problem, not a content problem — cheaper to test first.
4. **Don't rewrite recovering pages:** pages showing organic recovery despite worse position should be left to stabilize.
5. **Manual spot-check outside the ranked list periodically:** because recall is 0.52, roughly half of true declines won't surface in the top of this list — a lightweight manual review cadence should remain in place alongside the model.

**Confidence stated explicitly:** this is a **decision-support** ranking, not an automated action system. Given F1 = 0.60 and recall = 0.52, the ranked list should be treated as "worth checking first," not as ground truth about which pages will decline.

---

## 8. Reproducibility

**Environment:**
```
pip install -q duckdb huggingface_hub pandas scikit-learn matplotlib
```

**Re-run from a fresh clone:**
```bash
git clone https://github.com/saqib0-cpu/flyrank-ml-internship.git
cd flyrank-ml-internship
jupyter notebook work/capstone_notebook.ipynb
```
Then, inside the notebook: run cells top to bottom. The first cell prompts for a Hugging Face read token (via `getpass`, never hardcoded) and creates a DuckDB secret; the notebook then reads directly from `hf://datasets/FlyRank/internship-warehouse` — no local data download required.

**Random seed:** `random_state=42` used consistently for both `train_test_split` and `GradientBoostingClassifier`.

**Key package versions (fill in from your environment):**
```
duckdb==<fill from `pip freeze | grep duckdb`>
scikit-learn==<fill from `pip freeze | grep scikit-learn`>
pandas==<fill from `pip freeze | grep pandas`>
```

---

## Claims checklist (confirmed before submission)

- [x] Language throughout uses observed / measured / directional / decision-support framing — no causal claims.
- [x] No claim of predicting or reverse-engineering Google's ranking algorithm.
- [x] No client-identifying details anywhere in this report or the repo.
- [x] Base rate (16.5%) reported alongside accuracy so the 88% figure isn't read as more impressive than it is.
- [x] F1 and precision/recall reported as the primary discrimination metrics, not accuracy alone.
- [ ] **To confirm before submitting:** re-run the notebook fresh once more and verify these exact numbers (F1 = 0.60 / 0.154, importance percentages) reproduce with `random_state=42`.
