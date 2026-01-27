# PROJECT_STATE_V3 — Basketball Transfer Machine

## 1. Project Identity

**Name:** Basketball Transfer Machine
**Season:** NBA 2023–24
**Purpose:** Interactive analytics application to evaluate player–team fit for transfers, based on team playstyle (dimensions + playtypes), with dynamic recalculation and player cards.

This document supersedes **PROJECT_STATE_V2** and includes all work completed up to the successful multi-team Streamlit demo.

---

## 2. Core Concept (Stable)

The application answers:

> Which players best fit a team’s current or hypothetical playstyle?

Fit is defined as a weighted combination of:

* **Macro dimensions fit** (pace, shot diet, efficiency, ball movement, physicality, defense)
* **Playtype identity fit** (OFF / DEF frequencies + PPP)

Scores are:

* Normalized to **0–100**
* Comparable across the entire league
* Position-aware where applicable

---

## 3. Data Sources & Processed Assets

### 3.1 Processed Data (authoritative)

Located under:

```
data/processed/
```

Key files:

* `player_card_2023-24_v10_with_volume_filters.csv`
* `team_dim_baseline_2023-24_scaled.csv`
* `player_dim_scores_2023-24_scaled_pos.csv`
* `team_playtype_season_2023-24_scaled.csv`
* `player_playtype_season_2023-24_scaled_pos.csv`
* `player_playtype_wide_2023-24_v2_ppp_def_good.csv`

Key conventions:

* All percentages scaled to **0–100**
* Playtypes split into `OFF` and `DEF`
* Percentiles computed **within position buckets** where relevant

---

## 4. Fit Score Architecture

### 4.1 Macro Fit

* Distance = L1 distance between team baseline and player dimensions
* Max distance = `100 * number_of_dimensions`
* Fit:

```
macro_fit = 100 * (1 - dist / max_dist)
```

### 4.2 Playtype Fit

* For each playtype: distance = |freq_diff| + |ppp_diff|
* Max per playtype = 200
* Aggregated across selected playtypes

### 4.3 Combined Fit

```
fit_score = alpha * macro_fit + (1 - alpha) * playtype_fit
```

Alpha = 0.5 (fixed for MVP)

---

## 5. Canonical Fit Score Files (CRITICAL)

### 5.1 Canonical Output Location

```
data/app/fit_scores/season=2023-24/
```

### 5.2 Canonical Naming Convention (FINAL)

```
fit_scores_canonical_side={OFF|DEF}_team={TEAM}_alpha=0.5_pos=ALL_excl=True.csv
```

### 5.3 Coverage Status

* **30 teams × 2 sides = 60 files**
* Verified present
* No missing OFF/DEF pairs

### 5.4 Generation

Generated via **Jupyter**, exporting and adapting logic from:

```
notebooks/00_ingest_entities.ipynb
```

Exported to:

```
notebooks/00_ingest_entities_export.py
```

Key resolution:

* Earlier confusion between:

  * `fit_scores_team=...`
  * `fit_scores_team=..._side=...`
  * `fit_scores_canonical_side=...`
* Canonical files are now the **only source** used by Streamlit

---

## 6. Streamlit Applications

### 6.1 Key App Files

* `app_player_cards_v2.py` — legacy Boston-only
* `app_fix_dynamic_fitscore.py` — dynamic recalculation proof
* **`app_all_teams.py` — CURRENT MAIN APP**

### 6.2 App Header

```
st.title("Basketball Transfer Machine")
```

---

## 7. UI Architecture (Current)

### 7.1 Team Selection

* All 30 NBA teams selectable
* Team change dynamically updates:

  * Team dimensions baseline
  * Team playtype identity

### 7.2 Team’s Playstyle Panel

**Components:**

* Header: `Team’s Playstyle`
* Toggle: `Edit playstyle`
* Sliders:

  * Macro dimensions
  * Playtype identity (OFF / DEF)

**Behavior:**

* Edit OFF → sliders locked
* Edit ON → sliders editable
* Turning OFF restores defaults

### 7.3 Recalculate Button

* Disabled by default
* Enabled only when:

  * Edit ON
  * At least one slider changed
* Clicking recalculates **DynamicFitScore**

---

## 8. Dynamic Fit Score (Key Fix)

### 8.1 Problem (Resolved)

* DynamicFitScore initially only resorted existing shortlist
* Did NOT reselect players

### 8.2 Fix (Final Behavior)

* DynamicFitScore is computed **before** top-N slicing
* Pool is re-ranked globally
* Shortlist now changes correctly

### 8.3 Visibility Rules

* `DynamicFitScore` column:

  * Hidden when Edit OFF
  * Visible only after Recalculate

---

## 9. Results Panel Logic (Stable)

Sorting modes:

1. FitScore
2. Archetype (1→2→3) then active score

Active score resolution:

```
score_col = DynamicFitScore if use_dynamic else FitScore
```

Archetype ranking:

* `_arch_rank` temporary column
* Dropped after sorting

---

## 10. Player Card Panel

### Implemented

* Position-normalized percentiles
* Spider charts (PPP + Frequency)
* Shot charts
* Advanced stats

### Deferred (Intentionally)

* vs Strong / Weak Defense filter
* Adjusted metrics (`_hat`)

These appear as **greyed-out UI elements** to tease future work.

---

## 11. Defensive Strength Filter (Deferred by Design)

### Why Deferred

* Would require recomputing **ALL stats**, not just top-level
* Includes:

  * Playtypes
  * Shot charts
  * Spider charts
* Too heavy for demo timeline

### Status

* Boolean flag exists (`is_strong_def`)
* No runtime wiring

---

## 12. Known Confusions Resolved

1. **Canonical vs team-side files**
2. **DynamicFitScore not affecting pool**
3. **Frequency percentiles smaller than PPP**

   * Root cause: mixed percentile bases
   * Fixed by enforcing consistent global distribution
4. **Streamlit cache loaders** (`@st.cache_data`)

   * Standardized loaders for dims, playtypes, scores

---

## 13. Development Conventions

* Streamlit only reads **canonical CSVs**
* Heavy computation done offline (Jupyter)
* App only recomputes lightweight distances
* One source of truth per dataset

---

## 14. Current Status

✅ Multi-team support (30 teams)
✅ Dynamic playstyle editing
✅ Correct shortlist recomputation
✅ Stable demo-ready app

---

## 15. Immediate Next Steps (Optional)

* Player Card panel minor fix (pending)
* Strong/Weak defense split v2
* Adjusted metrics (`_hat`)
* Position-aware team baselines
* Multi-season support

---

**This file is safe to paste into a new chat as full project context.**
