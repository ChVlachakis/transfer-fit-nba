# app.py
from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Transfer Fit NBA — MVP", layout="wide")

ROOT = Path(__file__).resolve().parent

# ---- Paths (your repo layout) ----
PLAYER_CARD_PATH = ROOT / "data" / "processed" / "player_card_2023-24_v10_with_volume_filters.csv"
SCORES_DIR = ROOT / "data" / "app" / "fit_scores" / "season=2023-24"

def score_file(team_abbr: str, side: str, alpha: float = 0.5, pos: str = "ALL", excl: bool = True) -> Path:
    # Using your canonical naming convention
    return SCORES_DIR / f"fit_scores_canonical_side={side}_team={team_abbr}_alpha={alpha}_pos={pos}_excl={str(excl)}.csv"

# ---- Loaders ----
@st.cache_data(show_spinner=False)
def load_player_card() -> pd.DataFrame:
    df = pd.read_csv(PLAYER_CARD_PATH)
    # canonicalize some cols we rely on
    required = ["player_id", "player_name", "team_abbr", "primary_pos", "games", "mpg", "poss_decided_pg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"player_card missing columns: {missing}. Available: {list(df.columns)[:40]}")
    return df

@st.cache_data(show_spinner=False)
def load_scores(team_abbr: str, side: str) -> pd.DataFrame:
    # For MVP we hardcode alpha=0.5, pos=ALL, excl=True to match your saved files.
    p = score_file(team_abbr=team_abbr, side=side, alpha=0.5, pos="ALL", excl=True)
    if not p.exists():
        raise FileNotFoundError(f"Score file not found: {p}")
    df = pd.read_csv(p)

    # We expect at least these
    need = ["player_id", "fit_score", "fit_score_pctile", "macro_fit_v2", "playtype_fit_v2"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"score file missing columns: {missing}. Available: {list(df.columns)[:40]}")
    return df

def apply_position_filter(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    if pos == "ALL":
        return df
    # your primary_pos values are already coarse: G/F/C
    return df[df["primary_pos"] == pos].copy()

# ---- UI ----
st.title("Transfer Fit NBA — MVP (BOS, 2023–24)")

card = load_player_card()

# Teams dropdown (we only have BOS score files now, but let’s keep it a dropdown)
teams = sorted(card["team_abbr"].dropna().unique().tolist())
default_team = "BOS" if "BOS" in teams else teams[0]

with st.sidebar:
    st.header("Filters")

    team = st.selectbox("Choose team", teams, index=teams.index(default_team))
    pos = st.selectbox("Position", ["ALL", "G", "F", "C"], index=0)
    addition_focus = st.selectbox("Addition focus", ["OFF", "DEF"], index=0)

    st.divider()
    st.header("Volume filters")
    # Defaults minimal: show everything unless user changes
    min_games = st.slider("Minimum games", min_value=1, max_value=int(card["games"].max()), value=1, step=1)
    min_mpg = st.slider("Minimum minutes/game (mpg)", min_value=0.0, max_value=48.0, value=1.0, step=0.5)
    # bound based on data so it feels sane
    max_poss_pg = float(max(1.0, card["poss_decided_pg"].max()))
    min_poss_dec_pg = st.slider(
        "Minimum possessions decided per game",
        min_value=0.0,
        max_value=round(max_poss_pg, 1),
        value=1.0,
        step=0.5,
    )

st.subheader("Team specific parameters")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.slider("Relative importance (α) — dims vs identity", 0.0, 1.0, 0.5, 0.05, disabled=True,
              help="Locked for now (file is alpha=0.5). Next step: compute live so this works.")
with c2:
    st.toggle("Edit dimensions", value=False, disabled=True,
              help="Not wired yet (needs live scoring).")
with c3:
    st.toggle("Edit identity", value=False, disabled=True,
              help="Not wired yet (needs live scoring). OFF/DEF focus works via file selection on the left.")

st.caption("For this MVP slice, the OFF/DEF toggle is implemented via selecting the corresponding precomputed score file.")

# ---- Load scores for selected team + side ----
try:
    scores = load_scores(team_abbr=team, side=addition_focus)
except Exception as e:
    st.error(str(e))
    st.stop()

# ---- Build pool: merge scores + volume fields ----
pool = scores.merge(
    card[["player_id", "player_name", "team_abbr", "primary_pos", "games", "mpg", "poss_decided_pg"]],
    on="player_id",
    how="left",
    suffixes=("", "_card"),
)

# Exclude selected team players (even if file already did it; idempotent)
pool = pool[pool["team_abbr"] != team].copy()

# Position filter
pool = apply_position_filter(pool, pos)

# Volume filters
pool = pool[(pool["games"] >= min_games) & (pool["mpg"] >= min_mpg) & (pool["poss_decided_pg"] >= min_poss_dec_pg)].copy()

st.write(f"Pool size after filters: **{len(pool)}**  | teams represented: **{pool['team_abbr'].nunique()}**")

# ---- Display: top 10 + bottom 5 ----
show_cols = [
    "player_id", "player_name", "team_abbr", "primary_pos",
    "fit_score_pctile", "fit_score", "macro_fit_v2", "playtype_fit_v2",
    "games", "mpg", "poss_decided_pg",
]

# sort desc for best
best = pool.sort_values("fit_score_pctile", ascending=False).head(10)[show_cols]
worst = pool.sort_values("fit_score_pctile", ascending=True).head(5)[show_cols]

left, right = st.columns(2)
with left:
    st.markdown("### Top 10 best fits")
    st.dataframe(best, use_container_width=True, hide_index=True)
with right:
    st.markdown("### Bottom 5 worst fits")
    st.dataframe(worst, use_container_width=True, hide_index=True)

st.caption(
    "Note: fit_score_pctile here comes from the saved score file; "
    "next step is live scoring so percentiles update under filtering and slider edits."
)
