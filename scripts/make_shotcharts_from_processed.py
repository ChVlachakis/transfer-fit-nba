from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle



SEASON = "2023-24"
SHOTS_PATH = Path("data/processed/shots_2023-24_regular_canonical.csv")
CARD_PATH = Path("data/processed/player_card_2023-24_v14_1_with_archetypes_defpppfix.csv")

OUT_DIR = Path(f"data/app/shotcharts/season={SEASON}")
DPI = 160
FIGSIZE = (6, 5)

ONLY_FIRST_N = None  # set e.g. 25 for testing
OVERWRITE = False


def draw_half_court(ax: plt.Axes) -> None:
    ax.set_xlim(-250, 250)
    ax.set_ylim(-50, 470)
    ax.set_aspect("equal")
    ax.axis("off")

    hoop = plt.Circle((0, 0), radius=7.5, linewidth=2, fill=False)
    ax.add_patch(hoop)
    ax.plot([-30, 30], [-7.5, -7.5], linewidth=2)  # backboard

    ax.add_patch(Rectangle((-80, -47.5), 160, 190, fill=False, linewidth=2))
    ax.add_patch(Rectangle((-60, -47.5), 120, 190, fill=False, linewidth=2))

    ax.add_patch(Circle((0, 142.5), 60, fill=False, linewidth=2))
    ax.add_patch(Circle((0, 142.5), 60, fill=False, linewidth=2, linestyle="dashed"))

    ax.add_patch(Circle((0, 0), 40, fill=False, linewidth=2))

    ax.plot([-220, -220], [-47.5, 92.5], linewidth=2)
    ax.plot([220, 220], [-47.5, 92.5], linewidth=2)
    ax.add_patch(Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=2, fill=False))

    ax.add_patch(Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=2))


def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s)  # remove weird punctuation
    s = s.strip().replace(" ", "_")
    return s


def pick_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Your canonical shots file might use different column names. Try common ones.
    """
    candidates = [
        ("LOC_X", "LOC_Y"),        # nba_api style
        ("loc_x", "loc_y"),
        ("x", "y"),
        ("shot_x", "shot_y"),
        ("SHOT_X", "SHOT_Y"),
    ]
    cols = set(df.columns)
    for x, y in candidates:
        if x in cols and y in cols:
            return x, y
    raise KeyError(f"Cannot find shot location columns. Has sample cols: {list(df.columns)[:40]}")


def main() -> None:
    if not SHOTS_PATH.exists():
        raise FileNotFoundError(f"Missing shots file: {SHOTS_PATH.resolve()}")
    if not CARD_PATH.exists():
        raise FileNotFoundError(f"Missing player card: {CARD_PATH.resolve()}")

    shots = pd.read_csv(SHOTS_PATH)
    card = pd.read_csv(CARD_PATH)

    if "player_id" not in shots.columns:
        raise KeyError(f"shots missing 'player_id'. Has sample cols: {list(shots.columns)[:40]}")
    if "player_id" not in card.columns or "player_name" not in card.columns:
        raise KeyError(f"card missing ['player_id','player_name']. Has sample cols: {list(card.columns)[:40]}")

    xcol, ycol = pick_xy_columns(shots)

    # Ensure numeric
    shots[xcol] = pd.to_numeric(shots[xcol], errors="coerce")
    shots[ycol] = pd.to_numeric(shots[ycol], errors="coerce")
    shots = shots.dropna(subset=[xcol, ycol, "player_id"])

    players = card[["player_id", "player_name"]].drop_duplicates().copy()
    players["player_id"] = players["player_id"].astype(int)

    if ONLY_FIRST_N:
        players = players.head(int(ONLY_FIRST_N))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    empty = 0

    for _, r in players.iterrows():
        pid = int(r["player_id"])
        name = str(r["player_name"])
        out_path = OUT_DIR / f"{pid}__{safe_filename(name)}.png"

        if out_path.exists() and not OVERWRITE:
            continue

        pshots = shots.loc[shots["player_id"] == pid, [xcol, ycol]]
        if pshots.empty:
            empty += 1
            # still write an empty court so Streamlit has something to show
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            draw_half_court(ax)
            ax.set_title(f"{name} — Shot Chart ({SEASON})", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            saved += 1
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        draw_half_court(ax)

        ax.scatter(pshots[xcol].to_numpy(), pshots[ycol].to_numpy(), s=6, alpha=0.35)
        ax.set_title(f"{name} — Shot Chart ({SEASON})", fontsize=10)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    print(f"Done. saved={saved}, empty_players={empty}, out_dir={OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
