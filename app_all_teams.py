from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Transfer Fit NBA — Fix Dynamic FitScore", layout="wide")

ROOT = Path(__file__).resolve().parent


# ---- Paths ----
PLAYER_CARD_PATH = ROOT / "data" / "processed" / "player_card_2023-24_v14_1_with_archetypes_defpppfix.csv"
SCORES_DIR = ROOT / "data" / "app" / "fit_scores" / "season=2023-24"
TEAM_PLAYTYPE_PATH = ROOT / "data" / "processed" / "team_playtype_season_2023-24_scaled.csv"
TEAM_DIM_BASELINE_PATH = ROOT / "data" / "processed" / "team_dim_baseline_2023-24_scaled.csv"
PLAYER_PLAYTYPE_WIDE_PATH = ROOT / "data" / "processed" / "player_playtype_wide_2023-24_v2_ppp_def_good.csv"
PLAYER_AST_SHOTBUCKET_PATH = ROOT / "data" / "processed" / "player_assist_shotbucket_2023-24.csv"
PLAYER_DIM_SCORES_PATH = ROOT / "data" / "processed" / "player_dim_scores_2023-24_scaled_pos.csv"

# shots dataset (canonical)
SHOTS_PATH = ROOT / "data" / "processed" / "shots_2023-24_regular_canonical.csv"

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
DIM_EDGE = {k: ("Low", "High") for k in DIM_KEYS}


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


@st.cache_data(show_spinner=False)
def load_shots() -> pd.DataFrame:
    if not SHOTS_PATH.exists():
        raise FileNotFoundError(f"Shots file not found: {SHOTS_PATH}")
    df = pd.read_csv(SHOTS_PATH)

    need = ["player_id", "shot_zone_basic", "shot_zone_area", "loc_x", "shot_attempted", "shot_made"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"shots file missing columns: {missing}. Available sample: {list(df.columns)[:40]}")
    return df


@st.cache_data(show_spinner=False)
def load_player_playtype_wide() -> pd.DataFrame:
    if not PLAYER_PLAYTYPE_WIDE_PATH.exists():
        raise FileNotFoundError(f"playtype wide not found: {PLAYER_PLAYTYPE_WIDE_PATH}")
    df = pd.read_csv(PLAYER_PLAYTYPE_WIDE_PATH)
    if "player_id" not in df.columns:
        raise KeyError("playtype wide missing player_id")
    
    # --- Fix: PT_FREQ columns in this file are not true percentiles.
    # Create percentile-by-rank versions (0..100) to use in the radar.
    freq_cols = [c for c in df.columns if c.startswith("PT_FREQ__")]

    for c in freq_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        # rank(pct=True) gives 0..1; multiply to 0..100
        # NaNs stay NaN and won't affect ranks
        df[c + "__PCTILE"] = s.rank(pct=True).mul(100.0)

    return df


@st.cache_data(show_spinner=False)
def load_player_ast_shotbucket() -> pd.DataFrame:
    if not PLAYER_AST_SHOTBUCKET_PATH.exists():
        raise FileNotFoundError(f"assist shotbucket not found: {PLAYER_AST_SHOTBUCKET_PATH}")
    df = pd.read_csv(PLAYER_AST_SHOTBUCKET_PATH)
    need = ["player_id", "ast_rim", "ast_mid", "ast_three", "ast_corner3"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"assist shotbucket missing columns: {missing}. cols sample: {list(df.columns)[:40]}")
    return df


@st.cache_data(show_spinner=False)
def load_ast_per100_decided_with_pos_pctiles(card_df: pd.DataFrame) -> pd.DataFrame:
    ast = load_player_ast_shotbucket()

    base = card_df[["player_id", "primary_pos", "player_poss_decided"]].copy()
    base = base.dropna(subset=["primary_pos", "player_poss_decided"]).copy()
    base["player_poss_decided"] = pd.to_numeric(base["player_poss_decided"], errors="coerce")
    base = base.dropna(subset=["player_poss_decided"]).copy()
    base = base[base["player_poss_decided"] > 0].copy()

    df = ast.merge(base, on="player_id", how="inner")

    for zone in ["rim", "mid", "three", "corner3"]:
        df[f"ast_{zone}_per100_poss_decided"] = (df[f"ast_{zone}"] / df["player_poss_decided"]) * 100.0

    df["ast_total_per100_poss_decided"] = (
        df["ast_rim_per100_poss_decided"]
        + df["ast_mid_per100_poss_decided"]
        + df["ast_three_per100_poss_decided"]
        + df["ast_corner3_per100_poss_decided"]
    )

    metric_cols = [
        "ast_total_per100_poss_decided",
        "ast_rim_per100_poss_decided",
        "ast_mid_per100_poss_decided",
        "ast_three_per100_poss_decided",
        "ast_corner3_per100_poss_decided",
    ]

    for col in metric_cols:
        df[f"{col}_pos_pctile"] = df.groupby("primary_pos")[col].rank(pct=True).mul(100.0)

    keep = ["player_id"] + metric_cols + [f"{c}_pos_pctile" for c in metric_cols]
    return df[keep].copy()


@st.cache_data(show_spinner=False)
def load_player_dim_scores() -> pd.DataFrame:
    if not PLAYER_DIM_SCORES_PATH.exists():
        raise FileNotFoundError(f"dim scores not found: {PLAYER_DIM_SCORES_PATH}")
    df = pd.read_csv(PLAYER_DIM_SCORES_PATH)
    need = [
        "player_id",
        "dim_pace_scaled", "dim_shot_diet_scaled", "dim_shoot_eff_scaled",
        "dim_ball_move_scaled", "dim_physical_scaled", "dim_defense_scaled"
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"dim scores missing columns: {missing}. cols sample: {list(df.columns)[:40]}")
    return df

import re

@st.cache_data(show_spinner=False)
def list_score_teams(scores_dir: Path, alpha: float = 0.5, pos: str = "ALL", excl: bool = True) -> list[str]:
    pattern = re.compile(
        rf"fit_scores_canonical_side=(OFF|DEF)_team=([A-Z]{{3}})_alpha={alpha}_pos={pos}_excl={str(excl)}\.csv$"
    )
    found: dict[str, set[str]] = {}
    for p in scores_dir.glob(f"fit_scores_canonical_side=*_team=*_alpha={alpha}_pos={pos}_excl={str(excl)}.csv"):
        m = pattern.search(p.name)
        if not m:
            continue
        side, team = m.group(1), m.group(2)
        found.setdefault(team, set()).add(side)

    teams = sorted([t for t, sides in found.items() if {"OFF", "DEF"}.issubset(sides)])
    return teams


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
        return {k: 50.0 for k in DIM_KEYS}
    r0 = row.iloc[0]
    return {k: float(r0[DIM_COL_MAP[k]]) for k in DIM_KEYS}


def reset_dim_state_to_team_defaults(team_defaults: dict[str, float]) -> None:
    for k in DIM_KEYS:
        st.session_state[f"dim:{k}"] = float(team_defaults.get(k, 50.0))


# -------------------------
# Shot region FG map helpers
# -------------------------
REGION_ORDER = [
    "Rim",
    "Paint (Non-RA)",
    "Mid-Range Left", "Mid-Range Right",
    "Corner 3 Left", "Corner 3 Right",
    "ATB 3 Left", "ATB 3 Right",
]


def shot_region(row) -> str | None:
    basic = str(row["shot_zone_basic"])
    x = float(row["loc_x"])

    if basic == "Restricted Area":
        return "Rim"

    if basic == "In The Paint (Non-RA)":
        return "Paint (Non-RA)"

    if basic == "Mid-Range":
        return "Mid-Range Left" if x < 0 else "Mid-Range Right"

    if basic in ("Left Corner 3", "Right Corner 3"):
        return "Corner 3 Left" if basic == "Left Corner 3" else "Corner 3 Right"

    if basic == "Above the Break 3":
        return "ATB 3 Left" if x < 0 else "ATB 3 Right"

    return None


def player_region_fg(shots: pd.DataFrame, player_id: int) -> pd.DataFrame:
    p = shots[(shots["player_id"] == player_id) & (shots["shot_attempted"] == 1)].copy()
    if p.empty:
        out = pd.DataFrame({"region": REGION_ORDER, "attempts": 0, "makes": 0, "fg_pct": np.nan})
        return out

    p["region"] = p.apply(shot_region, axis=1)
    p = p.dropna(subset=["region"])
    if p.empty:
        out = pd.DataFrame({"region": REGION_ORDER, "attempts": 0, "makes": 0, "fg_pct": np.nan})
        return out

    agg = (
        p.groupby("region", as_index=False)
         .agg(attempts=("shot_attempted", "size"), makes=("shot_made", "sum"))
    )
    agg["fg_pct"] = np.where(agg["attempts"] > 0, agg["makes"] / agg["attempts"], np.nan)

    full = pd.DataFrame({"region": REGION_ORDER})
    out = full.merge(agg, on="region", how="left")
    out["attempts"] = out["attempts"].fillna(0).astype(int)
    out["makes"] = out["makes"].fillna(0).astype(int)
    out["fg_pct"] = out["fg_pct"].astype(float)
    return out


def render_region_fg_plot(region_df: pd.DataFrame, min_attempts: int):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Arc

    THRESH = {
        "Rim": [(0.30, "Very bad"), (0.40, "Below avg"), (0.50, "Average"), (0.65, "Above avg"), (1.00, "Elite")],
        "Mid": [(0.25, "Very bad"), (0.35, "Below avg"), (0.45, "Average"), (0.55, "Above avg"), (1.00, "Elite")],
        "C3":  [(0.24, "Very bad"), (0.34, "Below avg"), (0.40, "Average"), (0.46, "Above avg"), (1.00, "Elite")],
        "ATB": [(0.23, "Very bad"), (0.33, "Below avg"), (0.37, "Average"), (0.40, "Above avg"), (1.00, "Elite")],
        "Paint": [(0.25, "Very bad"), (0.35, "Below avg"), (0.45, "Average"), (0.55, "Above avg"), (1.00, "Elite")],
    }

    def family(region: str) -> str:
        if region == "Rim":
            return "Rim"
        if region == "Paint (Non-RA)":
            return "Paint"
        if region.startswith("Mid-Range"):
            return "Mid"
        if region.startswith("Corner 3"):
            return "C3"
        if region.startswith("ATB 3"):
            return "ATB"
        return "Mid"

    PALETTE = {
        "Very bad": (0.20, 0.35, 0.80, 0.70),
        "Below avg": (0.35, 0.55, 0.90, 0.70),
        "Average": (0.80, 0.80, 0.80, 0.60),
        "Above avg": (0.95, 0.60, 0.35, 0.70),
        "Elite": (0.85, 0.20, 0.20, 0.70),
    }

    fg = {
        r["region"]: (float(r["fg_pct"]) if pd.notna(r["fg_pct"]) else np.nan, int(r["attempts"]))
        for _, r in region_df.iterrows()
    }

    def bucket(region: str) -> str | None:
        fg_pct, att = fg.get(region, (np.nan, 0))
        if att < min_attempts or np.isnan(fg_pct):
            return None
        fam = family(region)
        for upper, label in THRESH[fam]:
            if fg_pct <= upper:
                return label
        return "Elite"

    def rgba_for(region: str):
        b = bucket(region)
        if b is None:
            return (0.90, 0.90, 0.90, 0.35)
        return PALETTE[b]

    HOOP_X, HOOP_Y = 0.0, 5.25
    R3 = 23.75
    CORNER_X = 22.0
    CORNER_Y = 14.0

    PAINT_X0, PAINT_X1 = -8.0, 8.0
    PAINT_Y0, PAINT_Y1 = 0.0, 19.0

    RIM_X0, RIM_X1 = -8.0, 8.0
    RIM_Y0, RIM_Y1 = 0.0, 8.0

    nx, ny = 420, 420
    xs = np.linspace(-25, 25, nx)
    ys = np.linspace(0, 47, ny)
    X, Y = np.meshgrid(xs, ys)

    dist = np.sqrt((X - HOOP_X) ** 2 + (Y - HOOP_Y) ** 2)
    outside_arc = dist >= R3
    inside_arc = ~outside_arc

    in_corner_band = (Y <= CORNER_Y) & (np.abs(X) >= CORNER_X)
    in_rim = (X >= RIM_X0) & (X <= RIM_X1) & (Y >= RIM_Y0) & (Y <= RIM_Y1)
    in_paint = (X >= PAINT_X0) & (X <= PAINT_X1) & (Y >= PAINT_Y0) & (Y <= PAINT_Y1)

    region_mask = np.full(X.shape, "", dtype=object)

    region_mask[in_rim] = "Rim"
    paint_non_ra = in_paint & (~in_rim)
    region_mask[paint_non_ra] = "Paint (Non-RA)"

    left_corner = in_corner_band & (X < 0) & (~in_rim)
    right_corner = in_corner_band & (X >= 0) & (~in_rim)
    region_mask[left_corner] = "Corner 3 Left"
    region_mask[right_corner] = "Corner 3 Right"

    atb = outside_arc & (~in_corner_band)
    region_mask[atb & (X < 0)] = "ATB 3 Left"
    region_mask[atb & (X >= 0)] = "ATB 3 Right"

    mid = inside_arc & (~in_paint) & (~in_rim)
    region_mask[mid & (X < 0)] = "Mid-Range Left"
    region_mask[mid & (X >= 0)] = "Mid-Range Right"

    img = np.zeros((ny, nx, 4), dtype=float)
    for region in REGION_ORDER:
        mask = region_mask == region
        if not mask.any():
            continue
        img[mask] = rgba_for(region)

    fig = plt.figure(figsize=(5.6, 5.6))
    ax = fig.add_subplot(111)
    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 47)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.imshow(img, extent=(-25, 25, 0, 47), origin="lower", zorder=1)

    rm = region_mask
    boundary = np.zeros(rm.shape, dtype=bool)
    boundary[:, 1:] |= (rm[:, 1:] != rm[:, :-1]) & (rm[:, 1:] != "") & (rm[:, :-1] != "")
    boundary[1:, :] |= (rm[1:, :] != rm[:-1, :]) & (rm[1:, :] != "") & (rm[:-1, :] != "")

    ax.contour(
        boundary.astype(float),
        levels=[0.5],
        extent=(-25, 25, 0, 47),
        colors=[(0, 0, 0, 0.25)],
        linewidths=1.0,
        zorder=15,
    )

    label_pos = {
        "ATB 3 Left": (-12, 40),
        "ATB 3 Right": (12, 40),
        "Mid-Range Left": (-12, 22),
        "Mid-Range Right": (12, 22),
        "Paint (Non-RA)": (0, 12),
        "Corner 3 Left": (-22.5, 7),
        "Corner 3 Right": (22.5, 7),
        "Rim": (0, 4),
    }
    for region, (tx, ty) in label_pos.items():
        fg_pct, att = fg.get(region, (np.nan, 0))
        b = bucket(region)
        short = region.replace("Corner 3", "C3").replace("ATB 3", "ATB").replace("Mid-Range", "Mid")
        if att >= min_attempts and not np.isnan(fg_pct):
            txt = f"{short}\n{att} att\n{fg_pct*100:.1f}%\n{b}"
        else:
            txt = f"{short}\n{att} att"
        ax.text(tx, ty, txt, ha="center", va="center", fontsize=8, zorder=20)

    yb = CORNER_Y
    dx = np.sqrt(max(0.0, R3**2 - (yb - HOOP_Y) ** 2))
    theta_break = np.degrees(np.arctan2(yb - HOOP_Y, dx))
    theta1 = theta_break
    theta2 = 180 - theta_break

    ax.add_patch(Rectangle((-25, 0), 50, 47, fill=False, linewidth=2, edgecolor="black", zorder=30))
    ax.add_patch(Circle((0, 5.25), radius=0.75, fill=False, linewidth=2, edgecolor="black", zorder=30))
    ax.add_patch(Rectangle((-8, 0), 16, 19, fill=False, linewidth=2, edgecolor="black", zorder=30))
    ax.add_patch(Arc((0, 19), 12, 12, theta1=0, theta2=180, linewidth=2, edgecolor="black", zorder=30))
    ax.add_patch(Arc((HOOP_X, HOOP_Y), 2 * R3, 2 * R3, theta1=theta1, theta2=theta2, linewidth=2, edgecolor="black", zorder=30))
    ax.plot([-CORNER_X, -CORNER_X], [0, CORNER_Y], linewidth=2, color="black", zorder=30)
    ax.plot([CORNER_X, CORNER_X], [0, CORNER_Y], linewidth=2, color="black", zorder=30)

    return fig


DIM_KEY_TO_COL = {
    "pace": "dim_pace_scaled",
    "shot_diet": "dim_shot_diet_scaled",
    "shoot_eff": "dim_shoot_eff_scaled",
    "ball_move": "dim_ball_move_scaled",
    "physical": "dim_physical_scaled",
    "defense": "dim_defense_scaled",
}

DIM_KEY_TO_LABEL = {
    "pace": "Pace",
    "shot_diet": "Shot diet",
    "shoot_eff": "Shooting efficiency",
    "ball_move": "Ball movement",
    "physical": "Physicality",
    "defense": "Defense",
}

PLAYTYPE_TO_SUFFIX = {
    "Isolation": "Isolation",
    "Pick & Roll Ball Handler": "Pick_&_Roll_Ball_Handler",
    "Pick & Roll Roll Man": "Pick_&_Roll_Roll_Man",
    "Post Up": "Post_Up",
    "Spot Up": "Spot_Up",
    "Off Screen": "Off_Screen",
    "Handoff": "Handoff",
    "Cut": "Cut",
    "Transition": "Transition",
}


def compute_top_fit_stats_for_player(
    pid: int,
    addition_focus: str,
    dim_df: pd.DataFrame,
    pt_wide_df: pd.DataFrame,
    top_k: int = 4,
) -> pd.DataFrame:
    rows = []

    drow = dim_df.loc[dim_df["player_id"] == pid]
    if not drow.empty:
        d0 = drow.iloc[0]
        for dim_key, col in DIM_KEY_TO_COL.items():
            target = float(st.session_state.get(f"dim:{dim_key}", 50.0))
            player_val = d0.get(col, None)
            if player_val is None or pd.isna(player_val):
                continue
            player_val = float(player_val)
            gap = abs(player_val - target)
            rows.append({
                "kind": "DIM",
                "label": DIM_KEY_TO_LABEL.get(dim_key, dim_key),
                "player": player_val,
                "target": target,
                "gap": gap,
            })

    prow = pt_wide_df.loc[pt_wide_df["player_id"] == pid]
    if not prow.empty:
        p0 = prow.iloc[0]
        side = "OFF" if addition_focus == "OFF" else "DEF"

        playtypes = PLAYTYPES_UI_OFF if side == "OFF" else PLAYTYPES_UI_DEF
        for pt in playtypes:
            if not st.session_state.get(f"pt_on:{side}:{pt}", True):
                continue

            suffix = PLAYTYPE_TO_SUFFIX.get(pt, None)
            if suffix is None:
                continue

            t_freq = float(st.session_state.get(f"pt_freq:{side}:{pt}", 50.0))
            t_ppp = float(st.session_state.get(f"pt_ppp:{side}:{pt}", 50.0))

            c_freq = f"PT_FREQ__{side}__{suffix}"
            c_ppp = f"PT_PPP__{side}__{suffix}"

            if c_freq in pt_wide_df.columns and pd.notna(p0.get(c_freq)):
                pv = float(p0.get(c_freq))
                rows.append({
                    "kind": "PT_FREQ",
                    "label": f"{pt} · Freq",
                    "player": pv,
                    "target": t_freq,
                    "gap": abs(pv - t_freq),
                })

            if c_ppp in pt_wide_df.columns and pd.notna(p0.get(c_ppp)):
                pv = float(p0.get(c_ppp))
                rows.append({
                    "kind": "PT_PPP",
                    "label": f"{pt} · PPP",
                    "player": pv,
                    "target": t_ppp,
                    "gap": abs(pv - t_ppp),
                })

    if not rows:
        return pd.DataFrame(columns=["kind", "label", "player", "target", "gap"])

    out = pd.DataFrame(rows).sort_values(["gap", "label"], ascending=[True, True]).head(top_k).copy()
    return out


# --- Player card: playtype radar (OFF/DEF) ---
SPIDER_PLAYTYPES_OFF = [
    ("Isolation", "Isolation"),
    ("Transition", "Transition"),
    ("PR Ball Handler", "Pick_&_Roll_Ball_Handler"),
    ("PR Roll Man", "Pick_&_Roll_Roll_Man"),
    ("Spot Up", "Spot_Up"),
    ("Off Screen", "Off_Screen"),
    ("Post Up", "Post_Up"),
    ("Handoff", "Handoff"),
    ("Cut", "Cut"),
]

SPIDER_PLAYTYPES_DEF = [
    ("Isolation", "Isolation"),
    ("PR Ball Handler", "Pick_&_Roll_Ball_Handler"),
    ("PR Roll Man", "Pick_&_Roll_Roll_Man"),
    ("Spot Up", "Spot_Up"),
    ("Off Screen", "Off_Screen"),
    ("Post Up", "Post_Up"),
    ("Handoff", "Handoff"),
]


def get_player_playtype_vectors(playtype_wide_df: pd.DataFrame, player_id: int, side: str):
    side = side.upper().strip()
    playtypes = SPIDER_PLAYTYPES_OFF if side == "OFF" else SPIDER_PLAYTYPES_DEF

    row = playtype_wide_df.loc[playtype_wide_df["player_id"] == player_id]
    labels = [lbl for (lbl, _) in playtypes]

    if row.empty:
        return labels, [0.0] * len(labels), [0.0] * len(labels)

    row = row.iloc[0]
    freq_vals, ppp_vals = [], []

    for _label, suffix in playtypes:
        # Use the new true-percentile freq column
        freq_col = f"PT_FREQ__{side}__{suffix}__PCTILE"
        ppp_col = f"PT_PPP__{side}__{suffix}"

        fv = row.get(freq_col, 0.0)
        pv = row.get(ppp_col, 0.0)

        # Convert NaN -> 0 so radar is stable and min/max works
        fv = 0.0 if (fv is None or (isinstance(fv, float) and pd.isna(fv))) else float(fv)
        pv = 0.0 if (pv is None or (isinstance(pv, float) and pd.isna(pv))) else float(pv)

        freq_vals.append(fv)
        ppp_vals.append(pv)

    return labels, freq_vals, ppp_vals


def make_spider_chart(labels, freq_vals, ppp_vals, title: str):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    f = list(freq_vals) + [freq_vals[0]]
    p = list(ppp_vals) + [ppp_vals[0]]

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)

    ax.plot(angles, f, linewidth=2, label="FREQ (pctile)")
    ax.fill(angles, f, alpha=0.10)

    ax.plot(angles, p, linewidth=2, label="PPP (pctile)")
    ax.fill(angles, p, alpha=0.10)

    ax.set_title(title, fontsize=10, pad=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)

    return fig


def _fmt_num(x, digits=1):
    try:
        if x is None or (isinstance(x, (float, int)) and pd.isna(x)):
            return "—"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


def _fmt_pct(x, digits=1):
    try:
        if x is None or (isinstance(x, (float, int)) and pd.isna(x)):
            return "—"
        return f"{float(x) * 100:.{digits}f}%"
    except Exception:
        return "—"


def _get_any(p: pd.Series, candidates: list[str]):
    for c in candidates:
        if c in p.index:
            v = p.get(c)
            if v is not None and not pd.isna(v):
                return v, c
    return None, None


def stat_pair(p: pd.Series, value_cols: list[str], pctile_cols: list[str], *, value_fmt="num", digits=1):
    v, _ = _get_any(p, value_cols)
    q, _ = _get_any(p, pctile_cols)

    if value_fmt == "pct":
        v_str = _fmt_pct(v, digits=digits)
    else:
        v_str = _fmt_num(v, digits=digits)

    if q is None:
        q_str = "—"
    else:
        qf = float(q)
        q_str = f"{(qf * 100 if qf <= 1.0 else qf):.0f}"

    return v_str, q_str

def compute_dynamic_fit_scores_for_pool(
    pool_df: pd.DataFrame,
    addition_focus: str,
    dim_scores_df: pd.DataFrame,
    pt_wide_df: pd.DataFrame,
) -> pd.Series:
    """
    Returns a Series aligned to pool_df.index with a 0..100-ish DynamicFitScore.
    Uses:
      - Dimension targets from st.session_state["dim:<key>"]
      - Playtype targets from st.session_state["pt_freq:<side>:<pt>"], ["pt_ppp:<side>:<pt>"]
      - Includes playtypes only if pt_on:<side>:<pt> is True
    """
    if pool_df.empty:
        return pd.Series([], index=pool_df.index, dtype=float)

    side = "OFF" if addition_focus == "OFF" else "DEF"

    # --- Targets ---
    dim_targets = {k: float(st.session_state.get(f"dim:{k}", 50.0)) for k in DIM_KEYS}

    playtypes = PLAYTYPES_UI_OFF if side == "OFF" else PLAYTYPES_UI_DEF
    pt_targets = {}
    for pt in playtypes:
        if not st.session_state.get(f"pt_on:{side}:{pt}", True):
            continue
        pt_targets[pt] = (
            float(st.session_state.get(f"pt_freq:{side}:{pt}", 50.0)),
            float(st.session_state.get(f"pt_ppp:{side}:{pt}", 50.0)),
        )

    # --- Join player features for this pool ---
    ids = pool_df["player_id"].astype(int).unique()

    dim_sub = dim_scores_df[dim_scores_df["player_id"].isin(ids)].copy()
    pt_sub = pt_wide_df[pt_wide_df["player_id"].isin(ids)].copy()

    feat = (
        pool_df[["player_id"]].copy()
        .merge(dim_sub, on="player_id", how="left")
        .merge(pt_sub, on="player_id", how="left")
    )

    # --- Distance components (L1 / Manhattan) ---
    # Dimensions: average absolute error
    dim_errs = []
    for k in DIM_KEYS:
        col = DIM_KEY_TO_COL[k]
        if col in feat.columns:
            dim_errs.append((feat[col].astype(float) - dim_targets[k]).abs())
    dim_err = sum(dim_errs) / max(1, len(dim_errs)) if dim_errs else pd.Series(0.0, index=feat.index)

    # Playtypes: average abs error over enabled playtypes (freq + ppp)
    pt_errs = []
    for pt, (t_freq, t_ppp) in pt_targets.items():
        suffix = PLAYTYPE_TO_SUFFIX.get(pt)
        if not suffix:
            continue
        cf = f"PT_FREQ__{side}__{suffix}__PCTILE"
        cp = f"PT_PPP__{side}__{suffix}"  # PPP already pctile in your file
        if cf in feat.columns:
            pt_errs.append((feat[cf].astype(float).fillna(0.0) - t_freq).abs())
        if cp in feat.columns:
            pt_errs.append((feat[cp].astype(float).fillna(0.0) - t_ppp).abs())

    pt_err = sum(pt_errs) / max(1, len(pt_errs)) if pt_errs else pd.Series(0.0, index=feat.index)

    # --- Combine (simple 50/50 for now) ---
    total_err = 0.5 * dim_err + 0.5 * pt_err

    # Turn distance into a score (higher is better)
    # cap: error 0 -> 100, error 100 -> 0 (roughly)
    dyn_score = (100.0 - total_err).clip(lower=0.0, upper=100.0)

    # Align to pool_df index
    out = pd.Series(dyn_score.values, index=pool_df.index, dtype=float)
    return out

def _mark_playstyle_dirty():
    st.session_state["playstyle_dirty"] = True


def _reset_playstyle_flags():
    st.session_state["playstyle_dirty"] = False
    st.session_state["playstyle_recalculated"] = False


# -------------------------
# MAIN APP
# -------------------------
st.title("Basketball Transfer Machine")

card = load_player_card()
card = _coalesce_column(card, "poss_decided_pct", ["poss_decided_pct", "poss_decided_pct_x", "poss_decided_pct_y"])

teams = list_score_teams(SCORES_DIR, alpha=0.5, pos="ALL", excl=True)
if not teams:
    raise FileNotFoundError(f"No canonical score files found in: {SCORES_DIR}")
default_team = "BOS" if "BOS" in teams else teams[0]


team_playtypes = load_team_playtypes()
team_dim_df = load_team_dim_baseline()
shots = load_shots()

# Load once (cached) so everything uses the same df handle
pt_wide_global = load_player_playtype_wide()
dim_scores_global = load_player_dim_scores()
ast_dec_tbl_global = load_ast_per100_decided_with_pos_pctiles(card)

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
# Team's Playstyle panel
# -------------------------
def _get_team_playtype_defaults_for_side(
    team_playtypes_df: pd.DataFrame | None,
    team_abbr: str,
    side: str,
    playtypes: list[str],
) -> dict[str, dict[str, float]]:
    """
    Returns {pt: {"freq": <default>, "ppp": <default>}} in 0..100 scaled space.
    Falls back to 50s if team file missing or row missing.
    """
    out = {pt: {"freq": 50.0, "ppp": 50.0} for pt in playtypes}
    if team_playtypes_df is None:
        return out

    tp_team = team_playtypes_df[team_playtypes_df["team_abbr"] == team_abbr].copy()
    if tp_team.empty:
        return out

    grouping = "Offensive" if side == "OFF" else "Defensive"

    for pt in playtypes:
        row = tp_team[(tp_team["type_grouping"] == grouping) & (tp_team["playtype_label"] == pt)]
        if row.empty:
            continue
        r0 = row.iloc[0]
        if "freq_scaled" in row.columns and pd.notna(r0.get("freq_scaled")):
            out[pt]["freq"] = float(r0["freq_scaled"])
        if "ppp_scaled" in row.columns and pd.notna(r0.get("ppp_scaled")):
            out[pt]["ppp"] = float(r0["ppp_scaled"])
    return out


def _reset_playtype_state_to_team_defaults(team_abbr: str) -> None:
    """
    Resets BOTH OFF and DEF playtype sliders (and On toggles) to team defaults.
    Uses your team_playtypes file if available, else 50s.
    """
    for side, pts in [("OFF", PLAYTYPES_UI_OFF), ("DEF", PLAYTYPES_UI_DEF)]:
        defaults = _get_team_playtype_defaults_for_side(team_playtypes, team_abbr, side, pts)
        for pt in pts:
            st.session_state[f"pt_on:{side}:{pt}"] = True
            st.session_state[f"pt_freq:{side}:{pt}"] = float(defaults[pt]["freq"])
            st.session_state[f"pt_ppp:{side}:{pt}"] = float(defaults[pt]["ppp"])


with col_team:
    with st.container(border=True, height=PANEL_HEIGHT):
        st.subheader("Team's Playstyle")

        # One toggle controls BOTH Dimensions + Identity
        edit_playstyle = st.toggle("Edit playstyle", value=False, key="edit_playstyle")

        # --- init flags once ---
        if "playstyle_dirty" not in st.session_state:
            st.session_state["playstyle_dirty"] = False
        if "playstyle_recalculated" not in st.session_state:
            st.session_state["playstyle_recalculated"] = False

        # If coach turns edit OFF, we hide DynamicFitScore again and require a new recalc next time.
        prev_edit = st.session_state.get("_prev_edit_playstyle", None)
        if prev_edit is True and edit_playstyle is False:
            _reset_playstyle_flags()

        # Recalculate button (disabled unless edit ON + at least one slider changed)
        recalc_clicked = st.button(
            "Recalculate",
            disabled=(not edit_playstyle) or (not st.session_state.get("playstyle_dirty", False)),
            use_container_width=True,
        )

        if recalc_clicked:
            st.session_state["playstyle_recalculated"] = True
            st.session_state["playstyle_dirty"] = False


        # Defaults for dimensions for the selected team
        team_defaults = get_team_dim_defaults(team_dim_df, team)

        # --- Reset rules ---
        # 1) When team changes AND edit_playstyle is OFF: snap to new team defaults.
        # 2) When edit_playstyle flips from ON -> OFF: revert to defaults (dims + playtypes).
        prev_team = st.session_state.get("_prev_team_for_playstyle", None)
        prev_edit = st.session_state.get("_prev_edit_playstyle", None)

        team_changed = (prev_team is not None) and (prev_team != team)
        turned_off = (prev_edit is True) and (edit_playstyle is False)

        if team_changed and not edit_playstyle:
            reset_dim_state_to_team_defaults(team_defaults)
            _reset_playtype_state_to_team_defaults(team)

        if turned_off:
            reset_dim_state_to_team_defaults(team_defaults)
            _reset_playtype_state_to_team_defaults(team)

        st.session_state["_prev_team_for_playstyle"] = team
        st.session_state["_prev_edit_playstyle"] = edit_playstyle

        inner = st.container(height=TEAM_INNER_HEIGHT)
        with inner:
            tabs = st.tabs(["Dimensions", "Identity (Playtypes)"])

            # --------
            # Dimensions
            # --------
            with tabs[0]:
                st.caption("Dimensions are UI-only for now; live scoring wires them into DynamicFitScore.")

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
                            disabled=not edit_playstyle,
                            label_visibility="collapsed",
                            on_change=_mark_playstyle_dirty,
                        )
                    with cols[1]:
                        st.caption(f"Team default: {default_val:.1f}")

            # --------
            # Identity (Playtypes)
            # --------
            with tabs[1]:
                side = "OFF" if addition_focus == "OFF" else "DEF"
                playtypes = PLAYTYPES_UI_OFF if side == "OFF" else PLAYTYPES_UI_DEF

                # Per-team defaults (scaled 0..100)
                pt_defaults = _get_team_playtype_defaults_for_side(team_playtypes, team, side, playtypes)

                for pt in playtypes:
                    on_key = f"pt_on:{side}:{pt}"
                    if on_key not in st.session_state:
                        st.session_state[on_key] = True

                    row_cols = st.columns([1, 3])

                    # Disable ON toggles unless editing playstyle
                    with row_cols[0]:
                        st.toggle("On", value=st.session_state[on_key], key=on_key, disabled=not edit_playstyle)

                    with row_cols[1]:
                        # Keep your captions with real freq/ppp if you want (unchanged logic),
                        # but defaults here are always the team's scaled defaults.
                        st.caption(f"{pt}")

                        freq_slider_key = f"pt_freq:{side}:{pt}"
                        ppp_slider_key = f"pt_ppp:{side}:{pt}"

                        # Initialize keys once to team defaults
                        if freq_slider_key not in st.session_state:
                            st.session_state[freq_slider_key] = float(pt_defaults[pt]["freq"])
                        if ppp_slider_key not in st.session_state:
                            st.session_state[ppp_slider_key] = float(pt_defaults[pt]["ppp"])

                        st.slider(
                            "Freq (pctile)",
                            0.0, 100.0,
                            float(st.session_state.get(freq_slider_key, float(pt_defaults[pt]["freq"]))),
                            step=1.0,
                            key=freq_slider_key,
                            disabled=(not edit_playstyle) or (not st.session_state[on_key]),
                            on_change=_mark_playstyle_dirty,
                        )
                        st.slider(
                            "PPP (pctile)",
                            0.0, 100.0,
                            float(st.session_state.get(ppp_slider_key, float(pt_defaults[pt]["ppp"]))),
                            step=1.0,
                            key=ppp_slider_key,
                            disabled=(not edit_playstyle) or (not st.session_state[on_key]),
                            on_change=_mark_playstyle_dirty,
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

        # ---- Build view (full pool, no head() yet) ----
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

        # keep id for downstream selection
        pool_view["player_id"] = pool["player_id"].values

        # ---- Dynamic score gating ----
        use_dynamic = bool(st.session_state.get("playstyle_recalculated", False))

        if use_dynamic and len(pool_view) > 0:
            # compute RAW dynamic score for the ENTIRE candidate pool
            dim_scores = load_player_dim_scores()
            pt_wide = load_player_playtype_wide()

            dyn_raw = compute_dynamic_fit_scores_for_pool(
                pool_df=pool[["player_id"]].reset_index(drop=True),
                addition_focus=addition_focus,
                dim_scores_df=dim_scores,
                pt_wide_df=pt_wide,
            )

            # normalize to 0..100 percentile within this pool (so max ~ 100)
            dyn_raw_s = pd.to_numeric(pd.Series(dyn_raw), errors="coerce")
            pool_view["DynamicFitScore"] = dyn_raw_s.rank(pct=True).mul(100.0).values
        else:
            pool_view["DynamicFitScore"] = np.nan

        # ---- Columns to show ----
        show_cols = ["Name", "Team", "Position", "FitScore", "Games", "MPG", "poss_decided_pg", "archetype_1", "archetype_2", "archetype_3"]

        if use_dynamic:
            insert_at = show_cols.index("FitScore") + 1
            show_cols.insert(insert_at, "DynamicFitScore")

        if "archetype_selected_score" in pool_view.columns:
            show_cols.insert(show_cols.index("poss_decided_pg") + 1, "archetype_selected_score")

        # ---- Sorting (sort full pool_view, then head() for display) ----
        score_col = "DynamicFitScore" if (use_dynamic and pool_view["DynamicFitScore"].notna().any()) else "FitScore"

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

            top_sorted = pool_view.sort_values(["_arch_rank", score_col], ascending=[True, False])
            bot_sorted = pool_view.sort_values(["_arch_rank", score_col], ascending=[True, True])

            top_sorted = top_sorted.drop(columns=["_arch_rank"], errors="ignore")
            bot_sorted = bot_sorted.drop(columns=["_arch_rank"], errors="ignore")
        else:
            top_sorted = pool_view.sort_values(score_col, ascending=False)
            bot_sorted = pool_view.sort_values(score_col, ascending=True)

        # ---- Display (ONLY now do head()) ----
        inner_results = st.container(height=RESULTS_INNER_HEIGHT)
        with inner_results:
            st.markdown("### Top fits")
            st.dataframe(top_sorted.head(15)[show_cols], use_container_width=True, hide_index=True)

            st.markdown("### Bottom fits")
            st.dataframe(bot_sorted.head(8)[show_cols], use_container_width=True, hide_index=True)


# -------------------------
# Player Card panel
# -------------------------
with col_card:
    with st.container(border=True, height=PANEL_HEIGHT):
        st.subheader("Player card")

        # Build picker options from the SAME data the table uses
        sel_map = top_sorted[["player_id", "Name", "Team", "Position", "DynamicFitScore"]].copy()
        sel_map = sel_map.rename(columns={"Name": "player_name", "Team": "team_abbr", "Position": "primary_pos"})

        sel_map["label"] = (
            sel_map["player_name"].astype(str)
            + " | " + sel_map["team_abbr"].astype(str)
            + " | " + sel_map["primary_pos"].astype(str)
            + " | Dyn " + sel_map["DynamicFitScore"].round(1).astype(str)
        )

        # already sorted by DynamicFitScore, but keep it explicit
        sel_map = sel_map.sort_values("DynamicFitScore", ascending=False)

        options = sel_map["label"].tolist()
        label_to_id = dict(zip(sel_map["label"], sel_map["player_id"]))


        if "pc_selected_label" not in st.session_state:
            st.session_state["pc_selected_label"] = options[0] if options else None

        picked = st.selectbox(
            "Pick a player (from the filtered pool)",
            options=options,
            key="pc_selected_label",
            disabled=(len(options) == 0),
        )

        cbtn1, cbtn2 = st.columns([1, 1])
        with cbtn1:
            show = st.button("Show card", disabled=(picked is None))
        with cbtn2:
            clear = st.button("Clear")

        if clear:
            st.session_state.pop("selected_player_id", None)

        if show and picked is not None:
            st.session_state["selected_player_id"] = int(label_to_id[picked])

        pid = st.session_state.get("selected_player_id", None)

        st.divider()

        if pid is None:
            st.info("Select a player above and click Show card.")
        else:
            prow = card.loc[card["player_id"] == pid]
            if prow.empty:
                st.error(f"Player id not found in player card dataset: {pid}")
            else:
                p = prow.iloc[0]
                srow = scores.loc[scores["player_id"] == pid]
                fit_pct = float(srow["fit_score_pctile"].iloc[0]) if not srow.empty else None
                dyn_row = pool_view.loc[pool_view["player_id"] == pid]
                dyn_fit = float(dyn_row["DynamicFitScore"].iloc[0]) if not dyn_row.empty else None


                st.markdown(f"### {p['player_name']}  ({p['primary_pos']})")
                st.caption(f"Current team: {p['team_abbr']}")

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("DynamicFitScore", f"{dyn_fit:.1f}" if dyn_fit is not None else "—")
                k2.metric("Games", f"{int(p['games'])}")
                k3.metric("MPG", f"{float(p['mpg']):.1f}")
                k4.metric("Poss decided / g", f"{float(p['poss_decided_pg']):.1f}")

                st.divider()

                # ---- Coming soon: contextual filters (UI only) ----
                st.markdown("### Contextual filters (coming soon)")

                c_ctx1, c_ctx2 = st.columns([2, 1], vertical_alignment="center")

                with c_ctx1:
                    st.caption("Opponent defensive strength")
                    dd, _sp = st.columns([6, 4], vertical_alignment="center")
                    with dd:
                        st.selectbox(
                            label="",
                            options=[
                                "All games",
                                "Vs strong defenses (top-15)",
                                "Vs weak defenses (bottom-15)",
                            ],
                            index=0,
                            disabled=True,
                            key="ctx_def_split_disabled",
                            label_visibility="collapsed",
                        )

                with c_ctx2:
                    st.caption("Adjusted metrics")
                    st.toggle(
                        label="",
                        value=False,
                        disabled=True,
                        key="ctx_adjusted_metrics_disabled",
                        label_visibility="collapsed",
                    )

                st.divider()

                # --- Top fit stats (closest-to-target) ---
                top_fit = compute_top_fit_stats_for_player(
                    pid=int(pid),
                    addition_focus=addition_focus,
                    dim_df=dim_scores_global,
                    pt_wide_df=pt_wide_global,
                    top_k=4,
                )

                st.markdown("### Top fit stats (closest-to-target)")
                if top_fit.empty:
                    st.caption("No fit stats available for this player.")
                else:
                    for _, r in top_fit.iterrows():
                        st.write(
                            f"**{r['label']}**: player {r['player']:.1f} vs target {r['target']:.1f} "
                            f"(gap {r['gap']:.1f})"
                        )

                st.divider()

                # -------------------------
                # Tactics performance (OFF/DEF spider)
                # -------------------------
                h1, h2 = st.columns([3, 2], vertical_alignment="center")
                with h1:
                    st.subheader("Tactics performance")
                with h2:
                    tactics_side = st.radio(
                        "Side",
                        ["OFF", "DEF"],
                        index=0,
                        horizontal=True,
                        key=f"tactics_side:{int(pid)}",
                        label_visibility="collapsed",
                    )

                pt_wide_global = load_player_playtype_wide()
                labels, freq_vals, ppp_vals = get_player_playtype_vectors(pt_wide_global, int(pid), tactics_side)

                # lazy-load once
                pt_wide = load_player_playtype_wide()

                # side for spider (OFF/DEF tab selection; keep OFF for now if DEF disabled)
                side = "OFF"  # or whatever variable you use from your OFF/DEF UI

                labels, freq_vals, ppp_vals = get_player_playtype_vectors(pt_wide, int(pid), side=side)

                fig_spider = make_spider_chart(
                    labels,
                    freq_vals,
                    ppp_vals,
                    title=f"{p['player_name']} — {side} Playtypes (percentiles)",
                )
                st.pyplot(fig_spider, clear_figure=True)

                st.divider()

                # -------------------------
                # Advanced stats
                # -------------------------
                st.markdown("### Advanced stats")
                st.caption("Percentiles are position normalized")

                region_df = player_region_fg(shots, int(pid))
                fg_by_region = {r["region"]: float(r["fg_pct"]) for _, r in region_df.iterrows()}

                ast_val, _ = _get_any(p, ["ast_per100_poss_used"])
                tov_val, _ = _get_any(p, ["tov_per100_poss_used"])
                ast_tov = "—"
                if ast_val is not None and tov_val is not None and float(tov_val) > 0:
                    ast_tov = f"{float(ast_val)/float(tov_val):.2f}"

                c_shoot, c_dec, c_def, c_phys = st.columns(4)

                with c_shoot:
                    st.markdown("**Shooting Eff.**")
                    ts_v, ts_q = stat_pair(p, ["ts_pct"], ["ts_pct_pos_pctile"], value_fmt="pct", digits=1)
                    efg_v, efg_q = stat_pair(p, ["efg_pct"], ["efg_pct_pos_pctile"], value_fmt="pct", digits=1)
                    st.write(f"TS: {ts_v}  · pctile {ts_q}")
                    st.write(f"eFG: {efg_v}  · pctile {efg_q}")

                with c_dec:
                    st.markdown("**Decision Making**")
                    st.caption("Number are per 100 poss decided")

                    tov_v2, tov_q2 = stat_pair(
                        p,
                        ["tov_per100_poss_used"],
                        ["tov_per100_poss_used_pos_pctile"],  # position-normalized
                        value_fmt="num",
                        digits=1,
                    )

                    drow = ast_dec_tbl_global.loc[ast_dec_tbl_global["player_id"] == int(pid)]
                    if drow.empty:
                        st.write("AST (per 100 poss decided): — · pctile —")
                        st.write("AST Rim: — · pctile —")
                        st.write("AST Mid: — · pctile —")
                        st.write("AST 3: — · pctile —")
                        st.write("AST C3: — · pctile —")
                    else:
                        d0 = drow.iloc[0]
                        st.write(f"AST (per 100 poss decided): {_fmt_num(d0['ast_total_per100_poss_decided'], 1)} · pctile {float(d0['ast_total_per100_poss_decided_pos_pctile']):.0f}")
                        st.write(f"AST Rim: {_fmt_num(d0['ast_rim_per100_poss_decided'], 2)} · pctile {float(d0['ast_rim_per100_poss_decided_pos_pctile']):.0f}")
                        st.write(f"AST Mid: {_fmt_num(d0['ast_mid_per100_poss_decided'], 2)} · pctile {float(d0['ast_mid_per100_poss_decided_pos_pctile']):.0f}")
                        st.write(f"AST 3: {_fmt_num(d0['ast_three_per100_poss_decided'], 2)} · pctile {float(d0['ast_three_per100_poss_decided_pos_pctile']):.0f}")
                        st.write(f"AST C3: {_fmt_num(d0['ast_corner3_per100_poss_decided'], 2)} · pctile {float(d0['ast_corner3_per100_poss_decided_pos_pctile']):.0f}")

                    st.write(f"TOV (per 100 poss used): {tov_v2} · pctile {tov_q2}")

                with c_def:
                    st.markdown("**Defense**")
                    stl_v, stl_q = stat_pair(p, ["stl_per100_min"], ["stl_per100_min_pos_pctile"], value_fmt="num", digits=2)
                    blk_v, blk_q = stat_pair(p, ["blk_per100_min"], ["blk_per100_min_pos_pctile"], value_fmt="num", digits=2)
                    dreb_v, dreb_q = stat_pair(p, ["dreb_per100_min"], ["dreb_per100_min_pos_pctile"], value_fmt="num", digits=2)
                    st.write(f"STL: {stl_v} · pctile {stl_q}")
                    st.write(f"BLK: {blk_v} · pctile {blk_q}")
                    st.write(f"DREB: {dreb_v} · pctile {dreb_q}")

                with c_phys:
                    st.markdown("**Physicality**")
                    fta_v, fta_q = stat_pair(p, ["fta_per_fga"], ["fta_per_fga_pos_pctile"], value_fmt="num", digits=2)
                    oreb_v, oreb_q = stat_pair(p, ["oreb_per100_min"], ["oreb_per100_min_pos_pctile"], value_fmt="num", digits=2)
                    st.write(f"FTA/FGA: {fta_v} · pctile {fta_q}")
                    st.write(f"OREB: {oreb_v} · pctile {oreb_q}")

                st.divider()

                viz = st.radio(
                    "Shot visualization",
                    ["Shot chart (dots)", "Shot efficiency map (FG% by region)"],
                    index=0,
                    horizontal=True,
                    key=f"shot_viz_mode:{int(pid)}",
                )

                if viz == "Shot chart (dots)":
                    st.markdown("**Shot chart (2023–24)**")
                    shot_dir = ROOT / "data" / "app" / "shotcharts" / "season=2023-24"
                    matches = sorted(shot_dir.glob(f"{int(pid)}__*.png")) if shot_dir.exists() else []
                    if not matches:
                        st.warning("No shot chart image found for this player_id.")
                    else:
                        st.image(str(matches[0]), use_container_width=True)

                else:
                    st.markdown("**Shot efficiency map (FG% by region)**")
                    min_att = st.slider(
                        "Min attempts per region",
                        0,
                        50,
                        10,
                        1,
                        key=f"min_att_region:{int(pid)}",
                    )

                    region_df = player_region_fg(shots, int(pid))
                    fig = render_region_fg_plot(region_df, min_attempts=int(min_att))
                    st.pyplot(fig, clear_figure=True)

                    show_tbl = st.toggle("Show region table", value=False, key=f"show_region_table:{int(pid)}")
                    if show_tbl:
                        st.dataframe(region_df, use_container_width=True, hide_index=True)

