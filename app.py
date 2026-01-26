from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(page_title="Transfer Fit NBA — MVP", layout="wide")

ROOT = Path(__file__).resolve().parent
SEASON = "2023-24"

# Data paths
PLAYER_CARD_PATH = ROOT / "data" / "processed" / "player_card_2023-24_v14_1_with_archetypes_defpppfix.csv"
SCORES_DIR = ROOT / "data" / "app" / "fit_scores" / f"season={SEASON}"

TEAM_DIM_BASELINE_PATH = ROOT / "data" / "processed" / "team_dim_baseline_2023-24_scaled.csv"
TEAM_PLAYTYPE_PATH = ROOT / "data" / "processed" / "team_playtype_season_2023-24_scaled.csv"
PLAYER_PLAYTYPE_PATH = ROOT / "data" / "processed" / "player_playtype_season_2023-24_scaled_pos.csv"

# identity labels we expose (no Misc)
IDENTITY_ORDER = [
    "Pick & Roll Ball Handler",
    "Pick & Roll Roll Man",
    "Isolation",
    "Post Up",
    "Spot Up",
    "Off Screen",
    "Handoff",
    "Cut",
    "Transition",
]

DIM_ORDER = [
    ("dim_pace_scaled", "Pace"),
    ("dim_shot_diet_scaled", "Shot Diet"),
    ("dim_shoot_eff_scaled", "Shooting Efficiency"),
    ("dim_ball_move_scaled", "Ball Movement"),
    ("dim_physical_scaled", "Physicality"),
    ("dim_defense_scaled", "Defense"),
]


# =========================
# Helpers
# =========================
def score_file(team_abbr: str, side: str, alpha: float = 0.5, pos: str = "ALL", excl: bool = True) -> Path:
    return SCORES_DIR / f"fit_scores_canonical_side={side}_team={team_abbr}_alpha={alpha}_pos={pos}_excl={str(excl)}.csv"


def _coalesce_column(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            return df
    return df


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}. cols sample: {list(df.columns)[:60]}")


def _safe_pct(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def apply_position_filter(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    if pos == "ALL":
        return df
    return df[df["primary_pos"] == pos].copy()


def archetypes_for_pos(pos: str) -> list[str]:
    if pos == "G":
        return [
            "OFFENSIVE_ENGINE",
            "ALL_AROUND",
            "PURE_POINT_GUARD",
            "SLASHER",
            "SHOOTING_SPECIALIST",
            "POINT_OF_ATTACK_DEFENDER",
            "MICROWAVE_SCORER",
            "3ANDD_GUARD",
        ]
    if pos == "F":
        return [
            "ELITE_WING",
            "GLUE_GUY",
            "3ANDD_WING",
            "LOCK_DOWN_DEFENDER",
            "REBOUND_AND_3_WING",
            "HUSTLER",
        ]
    if pos == "C":
        return [
            "UNICORN",
            "OFFENSIVE_HUB",
            "RIM_PROTECTOR",
            "ROLL_MAN",
            "STRETCH_BIG",
            "MOBILE_BIG",
            "POST_SCORER",
            "POST_DEFENDER",
            "REBOUND_SPECIALIST",
        ]
    # ALL
    return sorted(set(archetypes_for_pos("G") + archetypes_for_pos("F") + archetypes_for_pos("C")))


# =========================
# Loaders
# =========================
@st.cache_data(show_spinner=False)
def load_player_card() -> pd.DataFrame:
    if not PLAYER_CARD_PATH.exists():
        raise FileNotFoundError(f"Player card not found: {PLAYER_CARD_PATH}")
    df = pd.read_csv(PLAYER_CARD_PATH)

    required = ["player_id", "player_name", "team_abbr", "primary_pos", "games", "mpg", "poss_decided_pg"]
    _require_cols(df, required, "player_card")

    return df


@st.cache_data(show_spinner=False)
def load_scores(team_abbr: str, side: str) -> pd.DataFrame:
    p = score_file(team_abbr=team_abbr, side=side, alpha=0.5, pos="ALL", excl=True)
    if not p.exists():
        raise FileNotFoundError(f"Score file not found: {p}")
    df = pd.read_csv(p)
    need = ["player_id", "fit_score", "fit_score_pctile", "macro_fit_v2", "playtype_fit_v2"]
    _require_cols(df, need, "score file")
    return df


@st.cache_data(show_spinner=False)
def load_team_dims() -> pd.DataFrame:
    if not TEAM_DIM_BASELINE_PATH.exists():
        raise FileNotFoundError(f"Team dims not found: {TEAM_DIM_BASELINE_PATH}")
    df = pd.read_csv(TEAM_DIM_BASELINE_PATH)
    # expect: team_abbr + dim_*_scaled
    needed = ["team_abbr"] + [c for c, _ in DIM_ORDER]
    _require_cols(df, needed, "team_dim_baseline")
    return df


@st.cache_data(show_spinner=False)
def load_team_playtypes() -> pd.DataFrame:
    if not TEAM_PLAYTYPE_PATH.exists():
        raise FileNotFoundError(f"Team playtypes not found: {TEAM_PLAYTYPE_PATH}")
    df = pd.read_csv(TEAM_PLAYTYPE_PATH)
    # expected columns in your file: playtype_label, type_grouping, freq_scaled, ppp_scaled, plus team_abbr
    needed = ["team_abbr", "playtype_label", "type_grouping", "freq_scaled", "ppp_scaled"]
    _require_cols(df, needed, "team_playtypes")
    return df


@st.cache_data(show_spinner=False)
def load_player_playtypes() -> pd.DataFrame:
    if not PLAYER_PLAYTYPE_PATH.exists():
        raise FileNotFoundError(f"Player playtypes not found: {PLAYER_PLAYTYPE_PATH}")
    df = pd.read_csv(PLAYER_PLAYTYPE_PATH)
    needed = ["player_id", "playtype_label", "type_grouping", "freq_scaled", "ppp_scaled", "freq", "ppp"]
    _require_cols(df, needed, "player_playtypes")
    return df


# =========================
# UI
# =========================
st.title("Transfer Fit NBA — MVP")

card = load_player_card()
team_dims = load_team_dims()
team_pt = load_team_playtypes()
player_pt = load_player_playtypes()

teams = sorted(card["team_abbr"].dropna().unique().tolist())
default_team = "BOS" if "BOS" in teams else teams[0]

with st.sidebar:
    st.header("Filters")

    team = st.selectbox("Choose team", teams, index=teams.index(default_team))
    pos = st.selectbox("Position", ["ALL", "G", "F", "C"], index=0)
    addition_focus = st.selectbox("Addition focus", ["OFF", "DEF"], index=0)

    st.divider()
    st.header("Volume filters")
    min_games = st.slider("Minimum games", 1, int(card["games"].max()), 1, 1)
    min_mpg = st.slider("Minimum minutes/game (mpg)", 0.0, 48.0, 1.0, 0.5)
    max_poss_pg = float(max(1.0, card["poss_decided_pg"].max()))
    min_poss_dec_pg = st.slider(
        "Minimum possessions decided per game",
        min_value=0.0,
        max_value=round(max_poss_pg, 1),
        value=0.0,
        step=0.5,
    )

    st.divider()
    st.header("Archetypes filter")
    pos_for_arch = pos if pos != "ALL" else "ALL"
    arch_options = ["(Any)"] + archetypes_for_pos(pos_for_arch)
    selected_arch = st.selectbox("Include archetype", arch_options, index=0)


# =========================
# Team Specific Parameters
# =========================
st.subheader("Team specific parameters")
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    st.slider(
        "Relative importance (α) — dims vs identity",
        0.0,
        1.0,
        0.5,
        0.05,
        disabled=True,
        help="Locked for now (alpha=0.5). Next step is on-the-fly scoring so this works.",
    )
with c2:
    edit_dims = st.toggle("Edit dimensions", value=False, key="toggle_edit_dims")
with c3:
    edit_id = st.toggle("Edit identity (playtypes)", value=False, key="toggle_edit_id")

st.caption("Dimensions and Identity sliders are currently UI-only; scoring will be wired next step.")

# ---- Fetch team defaults ----
team_dims_row = team_dims[team_dims["team_abbr"] == team]
if team_dims_row.empty:
    st.warning(f"No team dims row found for {team}. Defaults will be neutral (50).")
    team_dims_default = {k: 50.0 for k, _ in DIM_ORDER}
else:
    team_dims_default = {k: float(team_dims_row.iloc[0][k]) for k, _ in DIM_ORDER}

team_pt_rows = team_pt[(team_pt["team_abbr"] == team) & (team_pt["playtype_label"].isin(IDENTITY_ORDER))].copy()
# defaults for playtypes (freq/ppp in scaled space)
team_pt_default = {}
for label in IDENTITY_ORDER:
    r = team_pt_rows[team_pt_rows["playtype_label"] == label]
    if r.empty:
        team_pt_default[label] = {"freq_scaled": 50.0, "ppp_scaled": 50.0, "freq": None, "ppp": None}
    else:
        team_pt_default[label] = {
            "freq_scaled": float(r.iloc[0]["freq_scaled"]),
            "ppp_scaled": float(r.iloc[0]["ppp_scaled"]),
            "freq": float(r.iloc[0].get("freq", float("nan"))) if "freq" in r.columns else None,
            "ppp": float(r.iloc[0].get("ppp", float("nan"))) if "ppp" in r.columns else None,
        }

# ---- Reset behavior: when master toggle is off, force values back to defaults ----
# Dimensions
for dim_key, dim_label in DIM_ORDER:
    inc_key = f"inc_dim__{dim_key}"
    val_key = f"val_dim__{dim_key}"

    if inc_key not in st.session_state:
        st.session_state[inc_key] = True
    if val_key not in st.session_state:
        st.session_state[val_key] = float(team_dims_default.get(dim_key, 50.0))

    if not edit_dims:
        st.session_state[inc_key] = True
        st.session_state[val_key] = float(team_dims_default.get(dim_key, 50.0))

# Identity
for label in IDENTITY_ORDER:
    inc_key = f"inc_pt__{label}"
    f_key = f"val_pt_freq__{label}"
    p_key = f"val_pt_ppp__{label}"

    if inc_key not in st.session_state:
        st.session_state[inc_key] = True
    if f_keyo_key := f_key not in st.session_state:
        st.session_state[f_key] = float(team_pt_default[label]["freq_scaled"])
    if p_key not in st.session_state:
        st.session_state[p_key] = float(team_pt_default[label]["ppp_scaled"])

    if not edit_id:
        st.session_state[inc_key] = True
        st.session_state[f_key] = float(team_pt_default[label]["freq_scaled"])
        st.session_state[p_key] = float(team_pt_default[label]["ppp_scaled"])

# ---- Render Dimensions ----
st.markdown("### Dimensions")
for dim_key, dim_label in DIM_ORDER:
    left, mid, right = st.columns([1, 6, 2])
    with left:
        inc = st.toggle(
            "",
            value=st.session_state[f"inc_dim__{dim_key}"],
            key=f"inc_dim__{dim_key}",
            disabled=not edit_dims,
            help="Include in calculation (will matter once live scoring is enabled).",
        )
    with mid:
        st.slider(
            f"{dim_label} (Low → High)",
            0.0,
            100.0,
            float(st.session_state[f"val_dim__{dim_key}"]),
            0.5,
            key=f"val_dim__{dim_key}",
            disabled=(not edit_dims) or (not inc),
        )
    with right:
        st.caption(f"Team default: {team_dims_default.get(dim_key, 50.0):.1f}")

# ---- Render Identity ----
st.markdown("### Identity (Playtypes)")
st.caption("Each playtype has two sliders: Frequency and PPP. PPP is in scaled space; defensive PPP was polarity-fixed upstream.")

for label in IDENTITY_ORDER:
    inc_key = f"inc_pt__{label}"
    f_key = f"val_pt_freq__{label}"
    p_key = f"val_pt_ppp__{label}"

    inc = st.toggle(
        label,
        value=st.session_state[inc_key],
        key=inc_key,
        disabled=not edit_id,
        help="Include in calculation (will matter once live scoring is enabled).",
    )

    # Show actual team stats if present
    freq_actual = team_pt_default[label].get("freq", None)
    ppp_actual = team_pt_default[label].get("ppp", None)
    actual_txt = []
    if freq_actual is not None and pd.notna(freq_actual):
        actual_txt.append(f"Freq: {freq_actual:.3f}")
    if ppp_actual is not None and pd.notna(ppp_actual):
        actual_txt.append(f"PPP: {ppp_actual:.3f}")
    actual_str = " | ".join(actual_txt) if actual_txt else "Freq/PPP not available"

    row1, row2 = st.columns([1, 1])
    with row1:
        st.slider(
            f"{label} — Frequency percentile ({actual_str})",
            0.0,
            100.0,
            float(st.session_state[f_key]),
            0.5,
            key=f_key,
            disabled=(not edit_id) or (not inc),
        )
    with row2:
        st.slider(
            f"{label} — PPP percentile ({actual_str})",
            0.0,
            100.0,
            float(st.session_state[p_key]),
            0.5,
            key=p_key,
            disabled=(not edit_id) or (not inc),
        )

st.divider()

# =========================
# Load scores and build pool
# =========================
try:
    scores = load_scores(team_abbr=team, side=addition_focus)
except Exception as e:
    st.error(str(e))
    st.stop()

pool = scores.merge(
    card[
        [
            "player_id",
            "player_name",
            "team_abbr",
            "primary_pos",
            "games",
            "mpg",
            "poss_decided_pg",
            "archetype_1", "archetype_1_score",
            "archetype_2", "archetype_2_score",
            "archetype_3", "archetype_3_score",
        ]
    ],
    on="player_id",
    how="left",
    suffixes=("_scorefile", "_card"),
)

# Canonicalize columns (avoid team_abbr/games regressions)
pool = _coalesce_column(pool, "player_name", ["player_name", "player_name_card", "player_name_y", "player_name_x"])
pool = _coalesce_column(pool, "team_abbr",   ["team_abbr", "team_abbr_card", "team_abbr_y", "team_abbr_x"])
pool = _coalesce_column(pool, "primary_pos", ["primary_pos", "primary_pos_card", "primary_pos_y", "primary_pos_x"])
pool = _coalesce_column(pool, "games",           ["games", "games_card", "games_y", "games_x"])
pool = _coalesce_column(pool, "mpg",             ["mpg", "mpg_card", "mpg_y", "mpg_x"])
pool = _coalesce_column(pool, "poss_decided_pg", ["poss_decided_pg", "poss_decided_pg_card", "poss_decided_pg_y", "poss_decided_pg_x"])

required = ["player_name", "team_abbr", "primary_pos", "games", "mpg", "poss_decided_pg"]
missing = [c for c in required if c not in pool.columns]
if missing:
    raise KeyError(f"Missing required columns after merge: {missing}. cols sample: {list(pool.columns)[:60]}")

pool["games"] = pd.to_numeric(pool["games"], errors="coerce")
pool["mpg"] = pd.to_numeric(pool["mpg"], errors="coerce")
pool["poss_decided_pg"] = pd.to_numeric(pool["poss_decided_pg"], errors="coerce")
pool = pool.dropna(subset=["games", "mpg", "poss_decided_pg"]).copy()

# Exclude selected team players (idempotent)
pool = pool[pool["team_abbr"] != team].copy()

# Position filter
pool = apply_position_filter(pool, pos)

# Archetype filter (top-1/2/3)
if selected_arch != "(Any)":
    pool = pool[
        (pool["archetype_1"] == selected_arch)
        | (pool["archetype_2"] == selected_arch)
        | (pool["archetype_3"] == selected_arch)
    ].copy()

# Volume filters
pool = pool[
    (pool["games"] >= float(min_games))
    & (pool["mpg"] >= float(min_mpg))
    & (pool["poss_decided_pg"] >= float(min_poss_dec_pg))
].copy()

st.write(f"Pool size after filters: **{len(pool)}** | teams represented: **{pool['team_abbr'].nunique()}**")

import re

def _norm_archetype(s: pd.Series) -> pd.Series:
    # Normalize: "All Around" -> "ALL_AROUND", "Pick & Roll" -> "PICK_ROLL"
    x = s.fillna("").astype(str).str.strip().str.upper()
    x = x.str.replace(r"[^\w]+", "_", regex=True)          # non-word -> _
    x = x.str.replace(r"_+", "_", regex=True).str.strip("_")
    return x

def _norm_one(x: str) -> str:
    x = str(x).strip().upper()
    x = re.sub(r"[^\w]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x

# =========================
# View mode (FitScore vs Archetype-grouped)
# =========================
st.markdown("### Results view")

view_mode = st.radio(
    "Sort mode",
    options=["FitScore", "Archetype (1→2→3) then FitScore"],
    horizontal=True,
    index=0,
)

pool_sorted = pool.copy()

if view_mode == "Archetype (1→2→3) then FitScore" and selected_arch != "(Any)":
    sel = _norm_one(selected_arch)

    a1 = _norm_archetype(pool_sorted["archetype_1"])
    a2 = _norm_archetype(pool_sorted["archetype_2"])
    a3 = _norm_archetype(pool_sorted["archetype_3"])

    pool_sorted["arch_rank_for_view"] = 99
    pool_sorted.loc[a1.eq(sel), "arch_rank_for_view"] = 1
    pool_sorted.loc[(pool_sorted["arch_rank_for_view"] == 99) & a2.eq(sel), "arch_rank_for_view"] = 2
    pool_sorted.loc[(pool_sorted["arch_rank_for_view"] == 99) & a3.eq(sel), "arch_rank_for_view"] = 3

    arch_pool = pool_sorted[pool_sorted["arch_rank_for_view"].isin([1, 2, 3])].copy()

    # Top 10: archetype_1 first, then _2, then _3; within rank best FitScore
    top_df = arch_pool.sort_values(
        by=["arch_rank_for_view", "fit_score_pctile"],
        ascending=[True, False],
    ).head(10).copy()

    # Bottom 5: still archetype_1 first, but within rank worst FitScore
    bot_df = arch_pool.sort_values(
        by=["arch_rank_for_view", "fit_score_pctile"],
        ascending=[True, True],
    ).head(5).copy()

else:
    pool_sorted = pool_sorted.sort_values("fit_score_pctile", ascending=False)
    top_df = pool_sorted.head(10).copy()
    bot_df = pool_sorted.tail(5).copy()



# =========================
# Display shortlist
# =========================
def make_pool_view(df: pd.DataFrame) -> pd.DataFrame:
    v = df.copy()
    v = v.rename(
        columns={
            "player_name": "Name",
            "team_abbr": "Team",
            "primary_pos": "Position",
            "fit_score_pctile": "FitScore",
            "games": "Games",
            "mpg": "MPG",
        }
    )
    show_cols = ["Name", "Team", "Position", "FitScore", "Games", "MPG", "poss_decided_pg", "archetype_1", "archetype_2", "archetype_3"]
    if "arch_rank_for_view" in v.columns:
        show_cols = ["arch_rank_for_view"] + show_cols
        v = v.rename(columns={"arch_rank_for_view": "ArchetypeRank"})
        show_cols[0] = "ArchetypeRank"

    show_cols = [c for c in show_cols if c in v.columns]
    return v[show_cols]

left, right = st.columns(2)
with left:
    st.markdown("### Top 10")
    st.dataframe(make_pool_view(top_df), use_container_width=True, hide_index=True)

with right:
    st.markdown("### Bottom 5")
    st.dataframe(make_pool_view(bot_df), use_container_width=True, hide_index=True)


