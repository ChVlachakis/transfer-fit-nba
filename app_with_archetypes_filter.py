# app_with_archetypes_filter.py
# Paste this WHOLE file (replace contents). It fixes:
# (1) KeyError 'team_abbr' via robust post-merge canonicalization
# (2) Archetype dropdown is position-aware (G/F/C) and updates when Position changes
# (3) Includes the full MVP feature set (team/pos/addition focus/volume filters + dims + identity UI scaffold)
# (4) Ensures "Handoff" is included in playtypes (and excludes "Misc" from the UI)

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Transfer Fit NBA — MVP (with archetypes)", layout="wide")

ROOT = Path(__file__).resolve().parent

# ---- Paths (keep versions consistent with your repo) ----
PLAYER_CARD_PATH = ROOT / "data" / "processed" / "player_card_2023-24_v14_1_with_archetypes_defpppfix.csv"
SCORES_DIR = ROOT / "data" / "app" / "fit_scores" / "season=2023-24"

# Optional (UI-only, doesn’t affect scoring yet): for showing team defaults in Identity section
TEAM_PLAYTYPE_PATH = ROOT / "data" / "processed" / "team_playtype_season_2023-24_scaled.csv"


# ---- Archetype list (positional gating for the dropdown) ----
ARCHETYPES_BY_POS: dict[str, list[str]] = {
    "G": [
        "OFFENSIVE_ENGINE",
        "ALL_AROUND",
        "PURE_POINT_GUARD",
        "SLASHER",
        "SHOOTING_SPECIALIST",
        "POINT_OF_ATTACK_DEFENDER",
        "MICROWAVE_SCORER",
        "3ANDD_GUARD",
    ],
    "F": [
        "ELITE_WING",
        "GLUE_GUY",
        "3ANDD_WING",
        "LOCK_DOWN_DEFENDER",
        "REBOUND_AND_3_WING",
        "HUSTLER",
    ],
    "C": [
        "UNICORN",
        "OFFENSIVE_HUB",
        "RIM_PROTECTOR",
        "ROLL_MAN",
        "STRETCH_BIG",
        "MOBILE_BIG",
        "POST_SCORER",
        "POST_DEFENDER",
        "REBOUND_SPECIALIST",
    ],
}


# ---- Playtypes for Identity UI (include Handoff, exclude Misc) ----
PLAYTYPES_UI_OFF = [
    "Isolation",
    "Pick & Roll Ball Handler",
    "Pick & Roll Roll Man",
    "Post Up",
    "Spot Up",
    "Off Screen",
    "Handoff",      # ✅ explicitly included
    "Cut",
    "Transition",
]
PLAYTYPES_UI_DEF = [
    "Isolation",
    "Pick & Roll Ball Handler",
    "Pick & Roll Roll Man",
    "Post Up",
    "Spot Up",
    "Off Screen",
    "Handoff",      # ✅ explicitly included
]


def score_file(team_abbr: str, side: str, alpha: float = 0.5, pos: str = "ALL", excl: bool = True) -> Path:
    return SCORES_DIR / f"fit_scores_canonical_side={side}_team={team_abbr}_alpha={alpha}_pos={pos}_excl={str(excl)}.csv"


def _coalesce_column(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    """
    Ensure df[target] exists by taking the first existing candidate column.
    """
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: target})
            return df
    return df


@st.cache_data(show_spinner=False)
def load_player_card() -> pd.DataFrame:
    if not PLAYER_CARD_PATH.exists():
        raise FileNotFoundError(f"player_card not found: {PLAYER_CARD_PATH}")
    df = pd.read_csv(PLAYER_CARD_PATH)

    required = [
        "player_id", "player_name", "team_abbr", "primary_pos", "games", "mpg", "poss_decided_pg",
        "archetype_1", "archetype_2", "archetype_3",
        "archetype_1_score", "archetype_2_score", "archetype_3_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"player_card missing columns: {missing}. Available sample: {list(df.columns)[:40]}")
    return df


@st.cache_data(show_spinner=False)
def load_scores(team_abbr: str, side: str) -> pd.DataFrame:
    p = score_file(team_abbr=team_abbr, side=side, alpha=0.5, pos="ALL", excl=True)
    if not p.exists():
        raise FileNotFoundError(f"Score file not found: {p}")
    df = pd.read_csv(p)

    need = ["player_id", "fit_score", "fit_score_pctile", "macro_fit_v2", "playtype_fit_v2"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"score file missing columns: {missing}. Available sample: {list(df.columns)[:40]}")
    return df


@st.cache_data(show_spinner=False)
def load_team_playtypes() -> pd.DataFrame | None:
    if not TEAM_PLAYTYPE_PATH.exists():
        return None
    return pd.read_csv(TEAM_PLAYTYPE_PATH)


def apply_position_filter(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    if pos == "ALL":
        return df
    return df[df["primary_pos"] == pos].copy()


def archetype_universe(card: pd.DataFrame) -> list[str]:
    s = pd.concat(
        [card["archetype_1"].dropna(), card["archetype_2"].dropna(), card["archetype_3"].dropna()],
        ignore_index=True,
    )
    vals = sorted({x for x in s.astype(str).tolist() if x and x != "nan"})
    return vals


def archetype_choices_for_pos(pos: str, card: pd.DataFrame) -> list[str]:
    if pos in ("G", "F", "C"):
        # Use your designed positional archetypes (stable, not inferred)
        return ARCHETYPES_BY_POS[pos]
    # pos == ALL -> show union of actually present archetypes
    return archetype_universe(card)


def archetype_match_mask(df: pd.DataFrame, selected: str, mode: str) -> pd.Series:
    if not selected or selected == "(any)":
        return pd.Series(True, index=df.index)

    a1 = df["archetype_1"].astype(str)
    a2 = df["archetype_2"].astype(str)
    a3 = df["archetype_3"].astype(str)

    if mode == "Top-1 only":
        return a1.eq(selected)
    return a1.eq(selected) | a2.eq(selected) | a3.eq(selected)


def archetype_score_for_selected(df: pd.DataFrame, selected: str) -> pd.Series:
    a1 = df["archetype_1"].astype(str)
    a2 = df["archetype_2"].astype(str)
    a3 = df["archetype_3"].astype(str)

    s = pd.Series(pd.NA, index=df.index, dtype="Float64")
    s = s.mask(a1.eq(selected), df["archetype_1_score"])
    s = s.mask(a2.eq(selected), df["archetype_2_score"])
    s = s.mask(a3.eq(selected), df["archetype_3_score"])
    return s


def _reset_slider_state(prefix: str, keys: list[str], defaults: dict[str, float]) -> None:
    """
    Used so when user turns OFF "Edit ..." toggles, sliders snap back to defaults.
    """
    for k in keys:
        st.session_state[f"{prefix}:{k}"] = float(defaults.get(k, 50.0))


# ---- UI ----
st.title("Transfer Fit NBA — MVP (with Archetypes filter)")

card = load_player_card()
teams = sorted(card["team_abbr"].dropna().unique().tolist())
default_team = "BOS" if "BOS" in teams else teams[0]

team_playtypes = load_team_playtypes()

with st.sidebar:
    st.header("Filters")

    team = st.selectbox("Choose team", teams, index=teams.index(default_team))
    pos = st.selectbox("Position", ["ALL", "G", "F", "C"], index=0)
    addition_focus = st.selectbox("Addition focus", ["OFF", "DEF"], index=0)

    st.divider()
    st.header("Archetype filter")

    # Dynamic archetype list per Position (requirement #2)
    choices = archetype_choices_for_pos(pos, card)
    archetype_options = ["(any)"] + choices

    # If position changes and previous selection becomes invalid, reset it cleanly
    prev = st.session_state.get("archetype_selected", "(any)")
    if prev not in archetype_options:
        st.session_state["archetype_selected"] = "(any)"

    archetype = st.selectbox("Archetype", archetype_options, key="archetype_selected")
    match_mode = st.selectbox("Match mode", ["Top-3", "Top-1 only"], index=0)

    min_arch_score = None
    if archetype != "(any)":
        tmp = archetype_score_for_selected(card, archetype).dropna()
        if len(tmp) == 0:
            st.warning("No players have this archetype in Top-3.")
            min_arch_score = 0.0
        else:
            lo, hi = float(tmp.min()), float(tmp.max())
            step = (hi - lo) / 100 if hi > lo else 1.0
            min_arch_score = st.slider("Minimum archetype score", lo, hi, lo, step=step)

    st.divider()
    st.header("Volume filters")

    min_games = st.slider("Minimum games", 1, int(card["games"].max()), 1, 1)
    min_mpg = st.slider("Minimum minutes/game (mpg)", 0.0, 48.0, 1.0, 0.5)

    max_poss_pg = float(max(1.0, card["poss_decided_pg"].max()))
    min_poss_dec_pg = st.slider(
        "Minimum possessions decided per game",
        min_value=0.0,
        max_value=round(max_poss_pg, 1),
        value=1.0,
        step=0.5,
    )

    st.divider()
    st.header("Team specific parameters")

    c1, c2 = st.columns([1, 1])

    # Dimensions UI scaffold (kept from prior app; not wired into scoring yet)
    with c1:
        edit_dims = st.toggle("Edit dimensions", value=False, key="edit_dims")

    # When toggled OFF, reset to defaults (requirement from prior spec)
    # We don’t have a per-team dims-default table in this file; default to 50 until live scoring.
    dim_keys = ["pace", "shot_diet", "shoot_eff", "ball_move", "physical", "defense"]
    dim_defaults = {k: 50.0 for k in dim_keys}
    if not edit_dims:
        _reset_slider_state("dim", dim_keys, dim_defaults)

    st.caption("Dimensions are UI-only for now; live scoring will wire them into results.")

    st.markdown("### Dimensions")
    dim_edge = {
        "pace": ("Slower", "Faster"),
        "shot_diet": ("Mid/Rim heavy", "3pt heavy"),
        "shoot_eff": ("Lower TS/eFG", "Higher TS/eFG"),
        "ball_move": ("Iso-heavy", "More passing"),
        "physical": ("Less physical", "More physical"),
        "defense": ("Weaker defense", "Stronger defense"),
    }

    for k in dim_keys:
        left_lbl, right_lbl = dim_edge[k]
        st.caption(f"{k.upper()}  |  {left_lbl}  ↔  {right_lbl}")
        st.slider(
            label=k,
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get(f"dim:{k}", 50.0)),
            step=1.0,
            key=f"dim:{k}",
            disabled=not edit_dims,
        )

    st.divider()
    st.markdown("### Identity (Playtypes)")

    # Identity UI scaffold (kept; excludes Misc; includes Handoff; requirement #4)
    # Not wired into scoring yet, but we show team defaults (freq/ppp) if file is present.
    playtypes = PLAYTYPES_UI_OFF if addition_focus == "OFF" else PLAYTYPES_UI_DEF

    # Defaults: if we can read team_playtypes, show actual freq/ppp in captions; sliders are percentile scales 0..100
    tp_team = None
    if team_playtypes is not None:
        # team_playtypes expected to have: team_abbr, type_grouping, playtype_label, freq, ppp, freq_scaled, ppp_scaled (names in your pipeline)
        tp_team = team_playtypes[team_playtypes["team_abbr"] == team].copy()

    for pt in playtypes:
        on_key = f"pt_on:{addition_focus}:{pt}"
        if on_key not in st.session_state:
            st.session_state[on_key] = True

        cols = st.columns([1, 3])
        with cols[0]:
            st.toggle("On", value=st.session_state[on_key], key=on_key)

        # Defaults from data if available
        freq_txt = ""
        ppp_txt = ""
        freq_default = 50.0
        ppp_default = 50.0

        if tp_team is not None:
            # type_grouping in your files is "Offensive"/"Defensive"
            grouping = "Offensive" if addition_focus == "OFF" else "Defensive"
            row = tp_team[(tp_team["type_grouping"] == grouping) & (tp_team["playtype_label"] == pt)]
            if len(row) > 0:
                r0 = row.iloc[0]
                if "freq" in row.columns and pd.notna(r0.get("freq")):
                    freq_txt = f" | freq={float(r0['freq'])*100:.1f}%" if float(r0["freq"]) <= 1.5 else f" | freq={float(r0['freq']):.1f}"
                if "ppp" in row.columns and pd.notna(r0.get("ppp")):
                    ppp_txt = f" | ppp={float(r0['ppp']):.2f}"
                if "freq_scaled" in row.columns and pd.notna(r0.get("freq_scaled")):
                    freq_default = float(r0["freq_scaled"])
                if "ppp_scaled" in row.columns and pd.notna(r0.get("ppp_scaled")):
                    ppp_default = float(r0["ppp_scaled"])

        with cols[1]:
            st.caption(f"{pt}{freq_txt}{ppp_txt}")
            freq_slider_key = f"pt_freq:{addition_focus}:{pt}"
            ppp_slider_key = f"pt_ppp:{addition_focus}:{pt}"

            if not st.session_state[on_key]:
                # grey out + keep values, but conceptually excluded later when live scoring happens
                st.slider("Freq (percentile)", 0.0, 100.0, float(st.session_state.get(freq_slider_key, freq_default)),
                          step=1.0, key=freq_slider_key, disabled=True)
                st.slider("PPP (percentile)", 0.0, 100.0, float(st.session_state.get(ppp_slider_key, ppp_default)),
                          step=1.0, key=ppp_slider_key, disabled=True)
            else:
                st.slider("Freq (percentile)", 0.0, 100.0, float(st.session_state.get(freq_slider_key, freq_default)),
                          step=1.0, key=freq_slider_key, disabled=False)
                st.slider("PPP (percentile)", 0.0, 100.0, float(st.session_state.get(ppp_slider_key, ppp_default)),
                          step=1.0, key=ppp_slider_key, disabled=False)

st.caption("FitScore still comes from saved precomputed score files; archetypes filter gates the shortlist.")

# ---- Load scores for selected team + side ----
scores = load_scores(team_abbr=team, side=addition_focus)

# ---- Build pool ----
# ---- Build pool ----
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

# Canonicalize expected columns after merge (handles games_x/games_y etc.)
pool = _coalesce_column(pool, "player_name", ["player_name", "player_name_card", "player_name_y", "player_name_x"])
pool = _coalesce_column(pool, "team_abbr",   ["team_abbr", "team_abbr_card", "team_abbr_y", "team_abbr_x"])
pool = _coalesce_column(pool, "primary_pos", ["primary_pos", "primary_pos_card", "primary_pos_y", "primary_pos_x"])

pool = _coalesce_column(pool, "games",           ["games", "games_card", "games_y", "games_x"])
pool = _coalesce_column(pool, "mpg",             ["mpg", "mpg_card", "mpg_y", "mpg_x"])
pool = _coalesce_column(pool, "poss_decided_pg", ["poss_decided_pg", "poss_decided_pg_card", "poss_decided_pg_y", "poss_decided_pg_x"])

# Hard fail early with a useful message
required = ["player_name", "team_abbr", "primary_pos", "games", "mpg", "poss_decided_pg"]
missing = [c for c in required if c not in pool.columns]
if missing:
    raise KeyError(f"Missing required columns after merge: {missing}. cols sample: {list(pool.columns)[:60]}")

# Ensure numeric for filters
pool["games"] = pd.to_numeric(pool["games"], errors="coerce")
pool["mpg"] = pd.to_numeric(pool["mpg"], errors="coerce")
pool["poss_decided_pg"] = pd.to_numeric(pool["poss_decided_pg"], errors="coerce")
pool = pool.dropna(subset=["games", "mpg", "poss_decided_pg"]).copy()


# Exclude selected team players (idempotent)
pool = pool[pool["team_abbr"] != team].copy()

# Position + volume filters
pool = apply_position_filter(pool, pos)
pool = pool[
    (pool["games"] >= min_games)
    & (pool["mpg"] >= min_mpg)
    & (pool["poss_decided_pg"] >= min_poss_dec_pg)
].copy()

# Archetype filter
if archetype != "(any)":
    mask = archetype_match_mask(pool, archetype, match_mode)
    pool = pool[mask].copy()
    pool["archetype_selected_score"] = archetype_score_for_selected(pool, archetype)
    if min_arch_score is not None:
        pool = pool[pool["archetype_selected_score"].fillna(-1e9) >= float(min_arch_score)].copy()

st.write(
    f"Pool size after filters: **{len(pool)}**  | teams represented: **{pool['team_abbr'].nunique()}**"
)

# ---- Display (same conventions you requested previously) ----
pool_view = pool.rename(
    columns={
        "player_name": "Name",
        "team_abbr": "Team",
        "primary_pos": "Position",
        "fit_score_pctile": "FitScore",
        "games": "Games",
        "mpg": "MPG",
    }
).copy()

show_cols = ["Name", "Team", "Position", "FitScore", "Games", "MPG", "poss_decided_pg", "archetype_1", "archetype_2", "archetype_3"]
if "archetype_selected_score" in pool_view.columns:
    show_cols.insert(show_cols.index("poss_decided_pg") + 1, "archetype_selected_score")

best = pool_view.sort_values("FitScore", ascending=False).head(15)[show_cols]
worst = pool_view.sort_values("FitScore", ascending=True).head(8)[show_cols]

left, right = st.columns(2)
with left:
    st.markdown("### Top fits")
    st.dataframe(best, use_container_width=True, hide_index=True)
with right:
    st.markdown("### Bottom fits")
    st.dataframe(worst, use_container_width=True, hide_index=True)
