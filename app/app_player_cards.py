# app_player_cards.py
# 4-panel desktop layout with SAME-HEIGHT outlined panels (no sidebar)
# Panels: Filters | Team Params | Results | Player Card (placeholder)
# Fix: Dimensions sliders now load TEAM-SPECIFIC defaults from:
#   data/processed/team_dim_baseline_2023-24_scaled.csv
# and reset correctly when team changes (when Edit dimensions is OFF).

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Transfer Fit NBA — Player Cards", layout="wide")

ROOT = Path(__file__).resolve().parent

# ---- Paths ----
PLAYER_CARD_PATH = ROOT / "data" / "processed" / "player_card_2023-24_v14_1_with_archetypes_defpppfix.csv"
SCORES_DIR = ROOT / "data" / "app" / "fit_scores" / "season=2023-24"
TEAM_PLAYTYPE_PATH = ROOT / "data" / "processed" / "team_playtype_season_2023-24_scaled.csv"

# ✅ Source of truth for team dimension defaults (from your other chat)
TEAM_DIM_BASELINE_PATH = ROOT / "data" / "processed" / "team_dim_baseline_2023-24_scaled.csv"

# ---- Sizing ----
PANEL_HEIGHT = 860
TEAM_INNER_HEIGHT = 780
RESULTS_INNER_HEIGHT = 780

# ---- Archetype list (positional gating) ----
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

# ---- Playtypes for Identity UI ----
PLAYTYPES_UI_OFF = [
    "Isolation",
    "Pick & Roll Ball Handler",
    "Pick & Roll Roll Man",
    "Post Up",
    "Spot Up",
    "Off Screen",
    "Handoff",
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
    "Handoff",
]

# ---- Dimensions wiring ----
DIM_KEYS = ["pace", "shot_diet", "shoot_eff", "ball_move", "physical", "defense"]
DIM_COL_MAP = {
    "pace": "dim_pace_scaled",
    "shot_diet": "dim_shot_diet_scaled",
    "shoot_eff": "dim_shoot_eff_scaled",
    "ball_move": "dim_ball_move_scaled",
    "physical": "dim_physical_scaled",
    "defense": "dim_defense_scaled",
}
DIM_LABELS = {
    "pace": "Pace",
    "shot_diet": "Shot Diet",
    "shoot_eff": "Shooting Efficiency",
    "ball_move": "Ball Movement",
    "physical": "Physicality",
    "defense": "Defense",
}
DIM_EDGE = {
    "pace": ("Low", "High"),
    "shot_diet": ("Low", "High"),
    "shoot_eff": ("Low", "High"),
    "ball_move": ("Low", "High"),
    "physical": ("Low", "High"),
    "defense": ("Low", "High"),
}


def score_file(team_abbr: str, side: str, alpha: float = 0.5, pos: str = "ALL", excl: bool = True) -> Path:
    return SCORES_DIR / f"fit_scores_canonical_side={side}_team={team_abbr}_alpha={alpha}_pos={pos}_excl={str(excl)}.csv"


def _coalesce_column(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: target})
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


@st.cache_data(show_spinner=False)
def load_team_dim_baseline() -> pd.DataFrame:
    if not TEAM_DIM_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Team dim baseline file not found: {TEAM_DIM_BASELINE_PATH}\n"
            "This file is required to set team-specific dimension defaults (otherwise you see 50s)."
        )
    df = pd.read_csv(TEAM_DIM_BASELINE_PATH)

    if "team_abbr" not in df.columns:
        raise KeyError(f"{TEAM_DIM_BASELINE_PATH.name} missing required column: team_abbr")

    missing_dim_cols = [DIM_COL_MAP[k] for k in DIM_KEYS if DIM_COL_MAP[k] not in df.columns]
    if missing_dim_cols:
        raise KeyError(
            f"{TEAM_DIM_BASELINE_PATH.name} missing dim columns: {missing_dim_cols}. "
            f"Available sample: {list(df.columns)[:40]}"
        )
    return df


def apply_position_filter(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    if pos == "ALL":
        return df
    return df[df["primary_pos"] == pos].copy()


def archetype_universe(card: pd.DataFrame) -> list[str]:
    s = pd.concat(
        [card["archetype_1"].dropna(), card["archetype_2"].dropna(), card["archetype_3"].dropna()],
        ignore_index=True,
    )
    return sorted({x for x in s.astype(str).tolist() if x and x != "nan"})


def archetype_choices_for_pos(pos: str, card: pd.DataFrame) -> list[str]:
    if pos in ("G", "F", "C"):
        return ARCHETYPES_BY_POS[pos]
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


def get_team_dim_defaults(team_dim_df: pd.DataFrame, team_abbr: str) -> dict[str, float]:
    row = team_dim_df.loc[team_dim_df["team_abbr"] == team_abbr]
    if row.empty:
        # fall back to 50 (but we should basically never hit this if file is correct)
        return {k: 50.0 for k in DIM_KEYS}

    r0 = row.iloc[0]
    out = {}
    for k in DIM_KEYS:
        out[k] = float(r0[DIM_COL_MAP[k]])
    return out


def reset_dim_state_to_team_defaults(team_defaults: dict[str, float]) -> None:
    for k in DIM_KEYS:
        st.session_state[f"dim:{k}"] = float(team_defaults.get(k, 50.0))


# -------------------------
# MAIN APP
# -------------------------
st.title("Transfer Fit NBA — Player Cards (WIP)")

card = load_player_card()
teams = sorted(card["team_abbr"].dropna().unique().tolist())
default_team = "BOS" if "BOS" in teams else teams[0]

team_playtypes = load_team_playtypes()
team_dim_df = load_team_dim_baseline()  # ✅ required for correct dim defaults

col_filters, col_team, col_results, col_card = st.columns([1.05, 1.60, 2.45, 2.20])

# -------------------------
# Filters panel
# -------------------------
with col_filters:
    with st.container(border=True, height=PANEL_HEIGHT):
        st.subheader("Filters")

        team = st.selectbox("Team", teams, index=teams.index(default_team))
        pos = st.selectbox("Position", ["ALL", "G", "F", "C"], index=0)
        addition_focus = st.selectbox("Addition focus", ["OFF", "DEF"], index=0)

        st.divider()
        st.subheader("Archetype")

        choices = archetype_choices_for_pos(pos, card)
        archetype_options = ["(any)"] + choices

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
                min_arch_score = st.slider("Min archetype score", lo, hi, lo, step=step)

        st.divider()
        st.subheader("Volume")

        min_games = st.slider("Min games", 1, int(card["games"].max()), 1, 1)
        min_mpg = st.slider("Min mpg", 0.0, 48.0, 1.0, 0.5)

        max_poss_pg = float(max(1.0, card["poss_decided_pg"].max()))
        min_poss_dec_pg = st.slider(
            "Min poss decided / game",
            min_value=0.0,
            max_value=round(max_poss_pg, 1),
            value=1.0,
            step=0.5,
        )

# -------------------------
# Team specific parameters panel
# -------------------------
with col_team:
    with st.container(border=True, height=PANEL_HEIGHT):
        st.subheader("Team specific parameters")

        inner = st.container(height=TEAM_INNER_HEIGHT)
        with inner:
            tabs = st.tabs(["Dimensions", "Identity (Playtypes)"])

            with tabs[0]:
                team_defaults = get_team_dim_defaults(team_dim_df, team)

                # track team changes for resetting sliders when Edit dims is OFF
                prev_team = st.session_state.get("_prev_team_for_dims", None)

                edit_dims = st.toggle("Edit dimensions", value=False, key="edit_dims")

                # If team changed and we are not editing dims, reset to new team defaults
                if prev_team != team:
                    st.session_state["_prev_team_for_dims"] = team
                    if not edit_dims:
                        reset_dim_state_to_team_defaults(team_defaults)

                # Also enforce defaults whenever edit_dims is OFF
                if not edit_dims:
                    reset_dim_state_to_team_defaults(team_defaults)

                st.caption("Dimensions are UI-only for now; live scoring will wire them into results.")

                for k in DIM_KEYS:
                    label = DIM_LABELS[k]
                    left_lbl, right_lbl = DIM_EDGE[k]
                    default_val = float(team_defaults[k])
                    current_val = float(st.session_state.get(f"dim:{k}", default_val))

                    cols = st.columns([7, 2])
                    with cols[0]:
                        st.caption(f"{label} ({left_lbl} → {right_lbl})")
                        st.slider(
                            label=label,
                            min_value=0.0,
                            max_value=100.0,
                            value=current_val,
                            step=0.25,
                            key=f"dim:{k}",
                            disabled=not edit_dims,
                            label_visibility="collapsed",
                        )
                    with cols[1]:
                        st.caption(f"Team default: {default_val:.1f}")

            with tabs[1]:
                playtypes = PLAYTYPES_UI_OFF if addition_focus == "OFF" else PLAYTYPES_UI_DEF

                tp_team = None
                if team_playtypes is not None:
                    tp_team = team_playtypes[team_playtypes["team_abbr"] == team].copy()

                for pt in playtypes:
                    on_key = f"pt_on:{addition_focus}:{pt}"
                    if on_key not in st.session_state:
                        st.session_state[on_key] = True

                    row_cols = st.columns([1, 3])
                    with row_cols[0]:
                        st.toggle("On", value=st.session_state[on_key], key=on_key)

                    freq_txt = ""
                    ppp_txt = ""
                    freq_default = 50.0
                    ppp_default = 50.0

                    if tp_team is not None:
                        grouping = "Offensive" if addition_focus == "OFF" else "Defensive"
                        row = tp_team[(tp_team["type_grouping"] == grouping) & (tp_team["playtype_label"] == pt)]
                        if len(row) > 0:
                            r0 = row.iloc[0]
                            if "freq" in row.columns and pd.notna(r0.get("freq")):
                                freq_txt = (
                                    f" | freq={float(r0['freq'])*100:.1f}%"
                                    if float(r0["freq"]) <= 1.5
                                    else f" | freq={float(r0['freq']):.1f}"
                                )
                            if "ppp" in row.columns and pd.notna(r0.get("ppp")):
                                ppp_txt = f" | ppp={float(r0['ppp']):.2f}"
                            if "freq_scaled" in row.columns and pd.notna(r0.get("freq_scaled")):
                                freq_default = float(r0["freq_scaled"])
                            if "ppp_scaled" in row.columns and pd.notna(r0.get("ppp_scaled")):
                                ppp_default = float(r0["ppp_scaled"])

                    with row_cols[1]:
                        st.caption(f"{pt}{freq_txt}{ppp_txt}")
                        freq_slider_key = f"pt_freq:{addition_focus}:{pt}"
                        ppp_slider_key = f"pt_ppp:{addition_focus}:{pt}"

                        st.slider(
                            "Freq (pctile)",
                            0.0, 100.0,
                            float(st.session_state.get(freq_slider_key, freq_default)),
                            step=1.0,
                            key=freq_slider_key,
                            disabled=not st.session_state[on_key],
                        )
                        st.slider(
                            "PPP (pctile)",
                            0.0, 100.0,
                            float(st.session_state.get(ppp_slider_key, ppp_default)),
                            step=1.0,
                            key=ppp_slider_key,
                            disabled=not st.session_state[on_key],
                        )

# -------------------------
# Results panel
# -------------------------
with col_results:
    with st.container(border=True, height=PANEL_HEIGHT):
        st.subheader("Results")

        sort_mode = st.radio(
            "Sort mode",
            ["FitScore", "Archetype (1→2→3) then FitScore"],
            index=0,
            horizontal=False,
        )

        scores = load_scores(team_abbr=team, side=addition_focus)

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

        pool = pool[pool["team_abbr"] != team].copy()
        pool = apply_position_filter(pool, pos)
        pool = pool[
            (pool["games"] >= min_games)
            & (pool["mpg"] >= min_mpg)
            & (pool["poss_decided_pg"] >= min_poss_dec_pg)
        ].copy()

        if archetype != "(any)":
            mask = archetype_match_mask(pool, archetype, match_mode)
            pool = pool[mask].copy()
            pool["archetype_selected_score"] = archetype_score_for_selected(pool, archetype)
            if min_arch_score is not None:
                pool = pool[pool["archetype_selected_score"].fillna(-1e9) >= float(min_arch_score)].copy()

        st.caption(f"Pool size: {len(pool)} | Teams represented: {pool['team_abbr'].nunique()}")

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

        if sort_mode == "Archetype (1→2→3) then FitScore" and archetype != "(any)":
            def _arch_rank(row) -> int:
                if str(row.get("archetype_1")) == archetype:
                    return 1
                if str(row.get("archetype_2")) == archetype:
                    return 2
                if str(row.get("archetype_3")) == archetype:
                    return 3
                return 99

            pool_view["_arch_rank"] = pool_view.apply(_arch_rank, axis=1)
            top_sorted = pool_view.sort_values(["_arch_rank", "FitScore"], ascending=[True, False])
            bot_sorted = pool_view.sort_values(["_arch_rank", "FitScore"], ascending=[True, True])
            pool_view = pool_view.drop(columns=["_arch_rank"], errors="ignore")
        else:
            top_sorted = pool_view.sort_values("FitScore", ascending=False)
            bot_sorted = pool_view.sort_values("FitScore", ascending=True)

        inner_results = st.container(height=RESULTS_INNER_HEIGHT)
        with inner_results:
            st.markdown("### Top fits")
            st.dataframe(top_sorted.head(15)[show_cols], use_container_width=True, hide_index=True)

            st.markdown("### Bottom fits")
            st.dataframe(bot_sorted.head(8)[show_cols], use_container_width=True, hide_index=True)

# -------------------------
# Player Card panel (placeholder)
# -------------------------
with col_card:
    with st.container(border=True, height=PANEL_HEIGHT):
        st.subheader("Player card")
        st.info("Next step: select a player row + click 'Show card' to populate this panel.")
