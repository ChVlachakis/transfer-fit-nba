# Transfer Fit NBA — PROJECT_STATE_V2

Last updated: 2026-01-26

## 0) What this project is
A **Streamlit MVP** that shortlists players for a selected NBA team based on:
- **Team style dimensions** (6 macro dims)
- **Identity / playtypes** (Synergy-style buckets with FREQ + PPP)
- **Volume filters** (games, MPG, possessions decided per game)
- **FitScore** (currently still read from precomputed fit files; sliders are UI-ready, but on-the-fly fit is still the next big step)
- **NEW:** **Archetype filter** (position-aware), plus an alternate ranking view that groups results by archetype rank (1→2→3).

We worked step-by-step, fixing repeated path/merge issues, normalizing tokens, and building a player archetype pipeline that outputs **top-1 + 2 runner-ups** per player.

---

## 1) Repo + environment
**Repo root (important):**
`transfer-fit-nba/`

**Python env:**
- Uses a local venv: `.venv`
- You sometimes also had `(base)` active; avoid mixing shells. Prefer: activate `.venv` only.

**How to run Streamlit:**
- Start: `streamlit run app.py`
- Stop: `Ctrl + C` in the terminal running Streamlit
- Keep Streamlit running while using terminal: open a **second terminal tab** (same folder + same venv)

---

## 2) Current app entrypoints

### 2.1 `app.py`
Working UI includes:
- Team selector
- Position selector
- Addition focus (OFF/DEF)
- Volume filters:
  - min games
  - min MPG
  - min possessions decided per game (back to integer per-game, not rate)
- Team Specific Parameters section:
  - **Dimensions** section:
    - “Edit dimensions” toggle enables sliders
    - Turning OFF “Edit dimensions” resets sliders to **team defaults**
    - Per-dimension ON/OFF switch disables slider when OFF and excludes from calculation (concept established)
  - **Identity / Playtypes**:
    - Includes playtype sliders + switches (exclude from playtype_fit when OFF)
    - “Misc” removed from UI
    - “Handoff” included (was missing earlier and was re-added)

Tables:
- Top 10 best fits, Bottom 5 worst fits
- Display columns were simplified (Name/Team/Position/FitScore/Games/MPG/poss_decided_pg), excluding internal score columns.

### 2.2 `app_with_archetypes_filter.py`
Everything from `app.py` **plus**:
- **Archetype filter**:
  - Dynamically changes options based on selected Position (G/F/C)
  - Filters pool to players having the selected archetype in their **top-3**
- **Sort mode toggle** above results:
  - `FitScore` (default): sort by FitScore, archetype just acts as a filter
  - `Archetype (1→2→3) then FitScore`: group players where selected archetype is archetype_1 first, then archetype_2, then archetype_3; within each group sort by FitScore.
  - Fixed bug where it was incorrectly sorting primarily by archetype_2/3.

---

## 3) Data layout and key produced artifacts

### 3.1 Directories (as used by code)
- `data/raw/`  
  Raw pulls / ingestion outputs (not the focus of recent work)
- `data/processed/`  
  Canonical processed CSVs used by notebooks + app
- `data/app/fit_scores/season=2023-24/`  
  Precomputed team fit score files used by the MVP

### 3.2 Player card versions (processed)
- `player_card_2023-24_v10_with_volume_filters.csv`
- `player_card_2023-24_v11_with_oncourt_rates.csv`
  - Added “on-court rate” columns (kept even though the “rate” concept was later deemphasized for MVP)
- `player_card_2023-24_v12_with_archetypes.csv` (first archetypes integration)
- `player_card_2023-24_v13_with_archetypes_defpppfix.csv` (after DEF PPP polarity fix attempt)
- `player_card_2023-24_v14_1_with_archetypes_defpppfix.csv` (final stable archetype run with tokens normalized 0..100)

### 3.3 Archetype outputs (processed)
- `player_archetypes_2023-24.csv` (initial run)
- `player_archetypes_2023-24_v2_defpppfix.csv`
- `player_archetypes_2023-24_v3_1_defpppfix_tokens100.csv`  ✅ current best

Each archetype file contains (per player):
- `archetype_1`, `archetype_1_score`
- `archetype_2`, `archetype_2_score`
- `archetype_3`, `archetype_3_score`

### 3.4 Playtype data (processed)
Inputs:
- `player_playtype_season_2023-24_scaled_pos.csv` (long format)
- `team_playtype_season_2023-24_scaled.csv`

Fixes + derived:
- `player_playtype_season_2023-24_scaled_pos_v2_ppp_def_good.csv` ✅
  - Defensive PPP polarity corrected (details below)
- `player_playtype_wide_2023-24_v2_ppp_def_good.csv` ✅
  - Wide features like:
    - `PT_FREQ__OFF__Isolation`, `PT_PPP__OFF__Isolation`
    - `PT_FREQ__DEF__Post_Up`, `PT_PPP__DEF__Post_Up`
    - etc.

### 3.5 Dimension scores (processed)
- `player_dim_scores_2023-24_scaled_pos.csv`  
Provides scaled macro-dimension columns that the app and archetype scoring rely on:
- `dim_pace_scaled`
- `dim_shot_diet_scaled`
- `dim_shoot_eff_scaled`
- `dim_ball_move_scaled`
- `dim_physical_scaled`
- `dim_defense_scaled`

### 3.6 Usage / creation (processed)
- `player_creation_season_2023-24_usage.csv`  
Used as source for usage/decision related tokens; note it does **not** contain `poss_decided_pg` (that column exists in player card versions).

---

## 4) Archetypes: current design
We implemented **positional multi-label archetype scoring** (not mutually exclusive).
- Each player gets a score per archetype relevant to their `primary_pos` bucket (G/F/C).
- Output is **top-1 + top-2 + top-3** archetypes per player.
- You provided the “formula language” as weighted sums of tokens (weights sum to 10).
- Key tricky archetypes:
  - “Switchable big” approximated using DEF response to **Pick & Roll Ball Handler**
  - “Post defending big” approximated using DEF response to **Post Up**
- We explicitly removed any plan to hard-restrict playtypes by position (edge cases like Jokic as handler), but archetypes themselves are computed by position group.

---

## 5) Repeatable errors we hit (and how we fixed them)

### 5.1 Running Python code in the terminal shell
Symptom: `-bash: import: command not found`
Cause: pasted Python lines directly into bash.
Fix: put code into `app.py` / notebook cell; run via `streamlit run ...` or `python ...`.

### 5.2 Wrong ROOT when executing from `/notebooks`
Symptom: FileNotFoundError paths like `.../transfer-fit-nba/notebooks/data/processed/...`
Cause: using `Path(__file__)` or `Path.cwd()` inconsistently.
Fix: ROOT detection logic that finds repo root (so reads from `transfer-fit-nba/data/...`).

### 5.3 `KeyError: 'team_abbr'` and other merge suffix issues
Symptom: after merges, columns became `team_abbr_x / team_abbr_y`, breaking downstream filters.
Fix: after merge, explicitly choose canonical columns and/or rename (e.g. pick `team_abbr_y` as `team_abbr`).

### 5.4 `ValueError: column label 'player_id' is not unique`
Cause: merging frames that already contained duplicate `player_id` columns.
Fix: drop duplicate columns before/after merges and assert uniqueness.

### 5.5 `KeyError: playtype_name` (schema mismatch)
Cause: playtype file uses `playtype_label`, not `playtype_name`.
Fix: standardize the code to use `playtype_label`.

### 5.6 Defensive PPP polarity was backwards
Cause: for defense, **lower PPP allowed is better**, but our scoring treated higher as better.
Fix: in DEF rows, set `ppp_scaled_use = 100 - ppp_scaled` (after confirming scale was 0..100).

### 5.7 Token scaling inconsistency (0..1 vs 0..100)
Symptom: archetype scores exploded or became nonsense.
Cause: some tokens were percentiles (0..1), some were already 0..100.
Fix: normalize all tokens to consistent 0..100 percentiles before scoring.
We also discovered:
- `C3` ended up constant in one run (nunique = 1), indicating the underlying source feature was degenerate/mapped wrong in that version; we treated it as non-informative for that run.

### 5.8 Possessions decided share filter confusion
We tried converting `poss_decided_pg` into rates; it produced nonsense (values > 1, weird slider max like 64.6).
Decision: revert MVP filter to **integer `poss_decided_pg`**, because it behaves cleanly.

---

## 6) Notebooks involved
- `notebooks/00_ingest_entities.ipynb`  
Updated and committed as part of initial MVP build steps.
- Archetype + playtype engineering happened inside the working notebook(s) with:
  - building the playtype DEF PPP fix output
  - building wide playtype table
  - building archetype scoring and writing v12/v13/v14_1 card outputs

(Notebook names beyond `00_ingest_entities.ipynb` exist in the repo history, but the key deliverable outputs are in `data/processed/` as listed above.)

---

## 7) Naming conventions (do not break these)
### 7.1 Core columns expected by apps
Player card / pool:
- `player_id`, `player_name`, `team_abbr`, `primary_pos`, `games`, `mpg`, `poss_decided_pg`
Fit:
- `fit_score_pctile` (currently used as FitScore display)
Archetypes:
- `archetype_1`, `archetype_2`, `archetype_3` (+ optional scores)

### 7.2 Playtype wide naming
- `PT_FREQ__OFF__{Playtype}`
- `PT_PPP__OFF__{Playtype}`
- `PT_FREQ__DEF__{Playtype}`
- `PT_PPP__DEF__{Playtype}`
where `{Playtype}` is sanitized (spaces/& replaced).

---

## 8) Git / committing guidance we used
Push only necessary artifacts:
- `app.py`
- `app_with_archetypes_filter.py`
- `notebooks/00_ingest_entities.ipynb`
- new `data/processed/...` outputs (player card + archetypes + playtype fixed tables)
Avoid committing:
- `.venv/`
- large raw data dumps unless intended
- ephemeral caches

---

## 9) Next steps (in the next chat)
1) **Create player cards UI**
   - Player detail component (card view) for shortlist rows
   - Include key stats: shot diet, TS/eFG, usage/decision, playtype splits, archetypes, etc.

2) **Generalize current developments to all teams (2023–2024)**
   - Ensure precomputed files exist for every team or switch to on-the-fly scoring
   - Verify team defaults for dims/identity sliders come from team baselines across all teams

3) **Machine learning: Expected PIR/40**
   - Train a model to estimate expected impact (PIR/40 proxy) if transferred to selected team
   - Use player features + team context as inputs (with careful leakage control)
   - Output as another ranking dimension or “expected outcome” panel

---

## 10) Post-MVP work
1) Add season **2024–2025**
2) Implement **cross-season transfers** (player changes team between seasons)
3) Implement **adjusted metrics** (`*_hat`)
   - e.g. context-adjusted shooting, playtype efficiency, role-adjusted creation/defense

---
