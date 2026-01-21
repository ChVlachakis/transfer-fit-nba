from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd

from nba_api.stats.static import teams as static_teams
from nba_api.stats.static import players as static_players
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import synergyplaytypes
from requests.exceptions import ConnectionError as RequestsConnectionError

NBA_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    def ensure(self) -> "ProjectPaths":
        self.raw.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)
        return self


def fetch_teams_players(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Saves data/raw/teams.csv and data/raw/players.csv."""
    teams = pd.DataFrame(static_teams.get_teams())
    teams = teams.rename(columns={"id": "team_id", "full_name": "team_name", "abbreviation": "team_abbr"})[
        ["team_id", "team_name", "team_abbr"]
    ].sort_values("team_id")

    players = pd.DataFrame(static_players.get_players())
    players = players.rename(columns={"id": "player_id", "full_name": "player_name"})[
        ["player_id", "player_name"]
    ].sort_values("player_id")

    paths.raw.mkdir(parents=True, exist_ok=True)
    (paths.raw / "teams.csv").write_text(teams.to_csv(index=False))
    (paths.raw / "players.csv").write_text(players.to_csv(index=False))
    return teams, players


def fetch_team_games_raw(season: str) -> pd.DataFrame:
    """Raw team game table from LeagueGameFinder (regular season)."""
    lgf = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        player_or_team_abbreviation="T",
    )
    return lgf.get_data_frames()[0].copy()

def fetch_team_game_fact(season: str, paths: ProjectPaths, league: str = "NBA") -> pd.DataFrame:
    """
    Builds and saves: data/processed/team_game_fact_{season}_regular.csv
    Includes opponent_team_id derived from MATCHUP + teams.csv join.
    """
    raw = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        player_or_team_abbreviation="T",
    ).get_data_frames()[0].copy()

    df = raw.rename(columns={
        "SEASON_ID": "season_id_raw",
        "TEAM_ID": "team_id",
        "TEAM_ABBREVIATION": "team_abbr",
        "TEAM_NAME": "team_name",
        "GAME_ID": "game_id",
        "GAME_DATE": "game_date",
        "MATCHUP": "matchup",
        "WL": "wl",
        "MIN": "min",
        "FGM": "fgm",
        "FGA": "fga",
        "FG_PCT": "fg_pct",
        "FG3M": "fg3m",
        "FG3A": "fg3a",
        "FG3_PCT": "fg3_pct",
        "FTM": "ftm",
        "FTA": "fta",
        "FT_PCT": "ft_pct",
        "OREB": "oreb",
        "DREB": "dreb",
        "REB": "reb",
        "AST": "ast",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "PF": "pf",
        "PTS": "pts",
        "PLUS_MINUS": "plus_minus",
        "VIDEO_AVAILABLE": "video_available",
    })

    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["team_id"] = df["team_id"].astype("int64")

    df["league"] = league
    df["season"] = season
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home_away"] = np.where(df["matchup"].str.contains("vs\\."), "H", "A")
    df["opponent_abbr"] = df["matchup"].str.split().str[-1]

    # join opponent_team_id from team_dim (teams.csv)
    team_dim = pd.read_csv(paths.raw / "teams.csv")[["team_id", "team_abbr"]]
    df = df.merge(
        team_dim.rename(columns={"team_id": "opponent_team_id", "team_abbr": "opponent_abbr"}),
        on="opponent_abbr",
        how="left",
    )

    out = paths.processed / f"team_game_fact_{season}_regular.csv"
    df.to_csv(out, index=False)
    return df

def fetch_player_game_fact(season: str, paths: ProjectPaths, league: str = "NBA") -> pd.DataFrame:
    """
    Builds and saves: data/processed/player_game_fact_{season}_regular.csv
    Enriches with opponent_team_id by joining team_game_fact on (game_id, team_id).
    """
    raw = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        player_or_team_abbreviation="P",
    ).get_data_frames()[0].copy()

    df = raw.rename(columns={
        "SEASON_ID": "season_id_raw",
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "TEAM_ID": "team_id",
        "TEAM_ABBREVIATION": "team_abbr",
        "TEAM_NAME": "team_name",
        "GAME_ID": "game_id",
        "GAME_DATE": "game_date",
        "MATCHUP": "matchup",
        "WL": "wl",
        "MIN": "min",
        "FGM": "fgm",
        "FGA": "fga",
        "FG_PCT": "fg_pct",
        "FG3M": "fg3m",
        "FG3A": "fg3a",
        "FG3_PCT": "fg3_pct",
        "FTM": "ftm",
        "FTA": "fta",
        "FT_PCT": "ft_pct",
        "OREB": "oreb",
        "DREB": "dreb",
        "REB": "reb",
        "AST": "ast",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "PF": "pf",
        "PTS": "pts",
        "PLUS_MINUS": "plus_minus",
        "FANTASY_PTS": "fantasy_pts",
        "VIDEO_AVAILABLE": "video_available",
    })

    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["team_id"] = df["team_id"].astype("int64")

    df["league"] = league
    df["season"] = season
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home_away"] = np.where(df["matchup"].str.contains("vs\\."), "H", "A")
    df["opponent_abbr"] = df["matchup"].str.split().str[-1]

    # Ensure team_game_fact exists (you can choose to call it explicitly instead)
    tgf_path = paths.processed / f"team_game_fact_{season}_regular.csv"
    if not tgf_path.exists():
        fetch_team_game_fact(season, paths, league=league)

    tgf = pd.read_csv(tgf_path)[["game_id", "team_id", "opponent_team_id", "opponent_abbr"]]

    # normalize join key dtypes
    df["game_id"] = df["game_id"].astype(str)
    tgf["game_id"] = tgf["game_id"].astype(str)
    df["team_id"] = df["team_id"].astype("int64")
    tgf["team_id"] = tgf["team_id"].astype("int64")

    tgf["game_id"] = tgf["game_id"].astype(str).str.zfill(10)
    tgf["team_id"] = tgf["team_id"].astype("int64")


    df = df.merge(tgf, on=["game_id", "team_id"], how="left")


    out = paths.processed / f"player_game_fact_{season}_regular.csv"
    df.to_csv(out, index=False)
    return df

from nba_api.stats.endpoints import shotchartdetail


def fetch_shots_season(
    season: str,
    paths: ProjectPaths,
    league: str = "NBA",
    season_type: str = "Regular Season",
    out_name: str | None = None,
    sleep_s: float = 0.6,
) -> pd.DataFrame:
    """
    Downloads season shot chart for ALL players (one call), saves to data/processed.
    Output columns follow our canonical naming.
    """
    # nba_api shotchartdetail expects season like "2023-24"
    sc = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,
        season_nullable=season,
        season_type_all_star=season_type,
        context_measure_simple="FGA",
        timeout=80,
    )
    df = sc.get_data_frames()[0].copy()
    time.sleep(sleep_s)

    # Canonical rename
    df = df.rename(columns={
        "GAME_ID": "game_id",
        "GAME_EVENT_ID": "game_event_id",
        "GAME_DATE": "game_date",
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "TEAM_ID": "team_id",
        "TEAM_NAME": "team_name",
        "PERIOD": "period",
        "MINUTES_REMAINING": "minutes_remaining",
        "SECONDS_REMAINING": "seconds_remaining",
        "SHOT_TYPE": "shot_type",
        "ACTION_TYPE": "action_type",
        "SHOT_ZONE_BASIC": "shot_zone_basic",
        "SHOT_ZONE_AREA": "shot_zone_area",
        "SHOT_ZONE_RANGE": "shot_zone_range",
        "SHOT_DISTANCE": "shot_distance",
        "LOC_X": "loc_x",
        "LOC_Y": "loc_y",
        "SHOT_ATTEMPTED_FLAG": "shot_attempted",
        "SHOT_MADE_FLAG": "shot_made",
        "HTM": "htm",
        "VTM": "vtm",
    })

    # Keep only canonical columns (drop GRID_TYPE and any surprises)
    keep = [
        "league","season",
        "game_id","game_event_id","game_date",
        "team_id","team_name","player_id","player_name",
        "period","minutes_remaining","seconds_remaining",
        "shot_type","action_type",
        "shot_zone_basic","shot_zone_area","shot_zone_range",
        "shot_distance","loc_x","loc_y",
        "shot_attempted","shot_made","htm","vtm",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()


    df["league"] = league
    df["season"] = season
    df["game_id"] = df["game_id"].astype(str).str.zfill(10)
    df["game_event_id"] = pd.to_numeric(df["game_event_id"], errors="coerce").astype("Int64")

    out = paths.processed / (out_name or f"shots_{season.replace('-', '')}_{season_type.lower().replace(' ', '_')}.csv")
    df.to_csv(out, index=False)
    return df

def add_shot_bucket(shots: pd.DataFrame) -> pd.DataFrame:
    """
    Adds shot_bucket: rim | mid | three | corner3 (corner3 is a subset of three)
    """
    df = shots.copy()

    # Corner 3 detection from zone basic (nba labels)
    is_corner3 = df["shot_zone_basic"].isin(["Left Corner 3", "Right Corner 3"])

    # Rim proxy from Restricted Area
    is_rim = df["shot_zone_basic"].eq("Restricted Area")

    # 3pt proxy from zone basic
    is_three = df["shot_zone_basic"].isin(["Above the Break 3", "Left Corner 3", "Right Corner 3"])

    bucket = np.where(is_corner3, "corner3",
             np.where(is_three, "three",
             np.where(is_rim, "rim", "mid")))

    df["shot_bucket"] = bucket
    return df


def build_player_shot_profile_season(
    shots_canonical_csv: Path,
    season: str,
    paths: ProjectPaths,
    league: str = "NBA",
    min_fga: int = 50,
) -> pd.DataFrame:
    """
    Builds per-player season shot diet + efficiency + percentiles.
    Saves: data/processed/player_shot_profile_{season}.csv
    """
    shots = pd.read_csv(shots_canonical_csv)
    shots = add_shot_bucket(shots)

    # attempts/makes by bucket
    g = shots.groupby(["player_id", "team_id", "shot_bucket"], as_index=False).agg(
        fga=("shot_attempted", "sum"),
        fgm=("shot_made", "sum"),
    )

    wide = g.pivot_table(
        index=["player_id", "team_id"],
        columns="shot_bucket",
        values=["fga", "fgm"],
        aggfunc="sum",
        fill_value=0,
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    # totals
    wide["fga_total"] = wide.filter(like="fga_").sum(axis=1)

    # rates
    for b in ["rim", "mid", "three", "corner3"]:
        if f"fga_{b}" not in wide.columns:
            wide[f"fga_{b}"] = 0
        wide[f"{b}_rate"] = np.where(wide["fga_total"] > 0, wide[f"fga_{b}"] / wide["fga_total"], np.nan)

    # fg% per bucket
    for b in ["rim", "mid", "three", "corner3"]:
        wide[f"{b}_fg_pct"] = np.where(
            wide[f"fga_{b}"] > 0,
            wide.get(f"fgm_{b}", 0) / wide[f"fga_{b}"],
            np.nan,
        )

    # filter for percentile calc (avoid noise)
    eligible = wide.loc[wide["fga_total"] >= min_fga].copy()

    # percentiles (league-wide)
    def pctile(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)

    for col in ["rim_rate", "mid_rate", "three_rate", "corner3_rate",
                "rim_fg_pct", "mid_fg_pct", "three_fg_pct", "corner3_fg_pct"]:
        pcol = f"{col}_pctile"
        wide[pcol] = np.nan
        if col in eligible.columns:
            ranks = pctile(eligible[col])
            wide.loc[eligible.index, pcol] = ranks.values


    # add league/season
    wide["league"] = league
    wide["season"] = season

    out = paths.processed / f"player_shot_profile_{season}.csv"
    wide.to_csv(out, index=False)
    return wide

def build_team_shot_profile_season(
    shots_canonical_csv: Path,
    season: str,
    paths: ProjectPaths,
    league: str = "NBA",
) -> pd.DataFrame:
    """
    Builds per-team season shot diet + efficiency + percentiles.
    Saves: data/processed/team_shot_profile_{season}.csv
    """
    shots = pd.read_csv(shots_canonical_csv)
    shots = add_shot_bucket(shots)

    g = shots.groupby(["team_id", "shot_bucket"], as_index=False).agg(
        fga=("shot_attempted", "sum"),
        fgm=("shot_made", "sum"),
    )

    wide = g.pivot_table(
        index=["team_id"],
        columns="shot_bucket",
        values=["fga", "fgm"],
        aggfunc="sum",
        fill_value=0,
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    wide["fga_total"] = wide.filter(like="fga_").sum(axis=1)

    for b in ["rim", "mid", "three", "corner3"]:
        if f"fga_{b}" not in wide.columns:
            wide[f"fga_{b}"] = 0
        wide[f"{b}_rate"] = np.where(wide["fga_total"] > 0, wide[f"fga_{b}"] / wide["fga_total"], np.nan)

    for b in ["rim", "mid", "three", "corner3"]:
        wide[f"{b}_fg_pct"] = np.where(
            wide[f"fga_{b}"] > 0,
            wide.get(f"fgm_{b}", 0) / wide[f"fga_{b}"],
            np.nan,
        )

    # team percentiles (all 30 teams, no min_fga needed)
    def pctile(s: pd.Series) -> pd.Series:
        return s.rank(pct=True)

    for col in ["rim_rate", "mid_rate", "three_rate", "corner3_rate",
                "rim_fg_pct", "mid_fg_pct", "three_fg_pct", "corner3_fg_pct"]:
        wide[f"{col}_pctile"] = pctile(wide[col])

    wide["league"] = league
    wide["season"] = season

    out = paths.processed / f"team_shot_profile_{season}.csv"
    wide.to_csv(out, index=False)
    return wide

def build_playtype_dict(paths: ProjectPaths) -> pd.DataFrame:
    """
    Hardcoded mapping (because nba_api expects special tokens, not the display labels).
    Saves: data/processed/playtype_dict.csv
    """
    rows = [
        # label, api_token, type_grouping, side
        ("Isolation", "Isolation", "Offensive", "OFF"),
        ("Isolation", "Isolation", "Defensive", "DEF"),
        ("Transition", "Transition", "Offensive", "OFF"),
        ("Transition", "Transition", "Defensive", "DEF"),
        ("Pick & Roll Ball Handler", "PRBallHandler", "Offensive", "OFF"),
        ("Pick & Roll Ball Handler", "PRBallHandler", "Defensive", "DEF"),
        ("Pick & Roll Roll Man", "PRRollMan", "Offensive", "OFF"),
        ("Pick & Roll Roll Man", "PRRollMan", "Defensive", "DEF"),
        ("Post Up", "Postup", "Offensive", "OFF"),
        ("Post Up", "Postup", "Defensive", "DEF"),
        ("Spot Up", "Spotup", "Offensive", "OFF"),
        ("Spot Up", "Spotup", "Defensive", "DEF"),
        ("Handoff", "Handoff", "Offensive", "OFF"),
        ("Handoff", "Handoff", "Defensive", "DEF"),
        ("Off Screen", "OffScreen", "Offensive", "OFF"),
        ("Off Screen", "OffScreen", "Defensive", "DEF"),
        ("Cut", "Cut", "Offensive", "OFF"),
        ("Cut", "Cut", "Defensive", "DEF"),
        ("Offensive Rebound Putbacks", "Putbacks", "Offensive", "OFF"),
        ("Offensive Rebound Putbacks", "Putbacks", "Defensive", "DEF"),
    ]
    df = pd.DataFrame(rows, columns=["playtype_label", "playtype_api", "type_grouping", "side"])
    out = paths.processed / "playtype_dict.csv"
    df.to_csv(out, index=False)
    return df


import random

def _synergy_fetch(
    season: str,
    player_or_team: str,
    play_type: str,
    type_grouping: str,
    timeout: int = 180,
    max_retries: int = 8,
    base_sleep: float = 2.0,
) -> pd.DataFrame:
    last_err = None
    for i in range(1, max_retries + 1):
        try:
            st = synergyplaytypes.SynergyPlayTypes(
                league_id="00",
                season=season,
                season_type_all_star="Regular Season",
                per_mode_simple="Per Possession",
                player_or_team_abbreviation=player_or_team,  # "P" or "T"
                play_type_nullable=play_type,
                type_grouping_nullable=type_grouping,
                headers=NBA_HEADERS,
                timeout=timeout,
            )
            df = st.get_data_frames()[0].copy()

            # small courtesy pause even on success (reduces bans)
            time.sleep(base_sleep + random.uniform(0.0, 0.8))
            return df

        except Exception as e:
            last_err = e
            # exponential backoff + jitter
            sleep_s = base_sleep * (2 ** (i - 1)) + random.uniform(0.0, 1.5)
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Synergy fetch failed after retries for {player_or_team}/{type_grouping}/{play_type}: {last_err}"
    )



def fetch_player_playtypes_season(
    season: str,
    paths: ProjectPaths,
    playtype_dict_csv: Path | None = None,
    league: str = "NBA",
    sleep_s: float = 0.9,
) -> pd.DataFrame:
    """
    Saves: data/processed/player_playtype_season_{season}.csv
    """
    dpath = playtype_dict_csv or (paths.processed / "playtype_dict.csv")
    ptd = pd.read_csv(dpath)

    parts = []
    for i, row in ptd.iterrows():
        pt_label = row["playtype_label"]
        pt_api = row["playtype_api"]
        grouping = row["type_grouping"]
        side = row["side"]

        try:
            df = _synergy_fetch(season, "P", pt_api, grouping)
        except (KeyError, RequestsConnectionError) as e:
            # fail fast: your earlier experience shows "resultSet" errors come from invalid tokens or throttling
            raise RuntimeError(f"Synergy fetch failed for P/{grouping}/{pt_api}: {e}") from e

        df = df.rename(columns={
            "SEASON_ID": "season_id_raw",
            "PLAYER_ID": "player_id",
            "PLAYER_NAME": "player_name",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbr",
            "TEAM_NAME": "team_name",
            "PLAY_TYPE": "playtype_api_returned",
            "TYPE_GROUPING": "type_grouping",
            "PERCENTILE": "percentile",
            "GP": "gp",
            "PPP": "ppp",
            "POSS_PCT": "freq",
            "POSS": "poss",
            "PTS": "pts",
            "FGM": "fgm",
            "FGA": "fga",
            "EFG_PCT": "efg_pct",
            "TOV_POSS_PCT": "tov_poss_pct",
        })

        df["league"] = league
        df["season"] = season
        df["playtype_label"] = pt_label
        df["playtype_api"] = pt_api
        df["side"] = side

        keep = [
            "league","season","season_id_raw",
            "player_id","player_name","team_id","team_abbr","team_name",
            "playtype_label","playtype_api","type_grouping","side",
            "gp","poss","freq","ppp","pts","fgm","fga","efg_pct","tov_poss_pct","percentile",
        ]
        df = df[[c for c in keep if c in df.columns]].copy()
        parts.append(df)

        time.sleep(sleep_s)

    out = paths.processed / f"player_playtype_season_{season}.csv"
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(out, index=False)
    return combined


def fetch_team_playtypes_season(
    season: str,
    paths: ProjectPaths,
    playtype_dict_csv: Path | None = None,
    league: str = "NBA",
    sleep_s: float = 0.9,
) -> pd.DataFrame:
    """
    Saves: data/processed/team_playtype_season_{season}.csv
    """
    dpath = playtype_dict_csv or (paths.processed / "playtype_dict.csv")
    ptd = pd.read_csv(dpath)

    parts = []
    for i, row in ptd.iterrows():
        pt_label = row["playtype_label"]
        pt_api = row["playtype_api"]
        grouping = row["type_grouping"]
        side = row["side"]

        try:
            df = _synergy_fetch(season, "T", pt_api, grouping)
        except (KeyError, RequestsConnectionError) as e:
            raise RuntimeError(f"Synergy fetch failed for T/{grouping}/{pt_api}: {e}") from e

        df = df.rename(columns={
            "SEASON_ID": "season_id_raw",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbr",
            "TEAM_NAME": "team_name",
            "PLAY_TYPE": "playtype_api_returned",
            "TYPE_GROUPING": "type_grouping",
            "PERCENTILE": "percentile",
            "GP": "gp",
            "PPP": "ppp",
            "POSS_PCT": "freq",
            "POSS": "poss",
            "PTS": "pts",
        })

        df["league"] = league
        df["season"] = season
        df["playtype_label"] = pt_label
        df["playtype_api"] = pt_api
        df["side"] = side

        keep = [
            "league","season","season_id_raw",
            "team_id","team_abbr","team_name",
            "playtype_label","playtype_api","type_grouping","side",
            "gp","poss","freq","ppp","pts","percentile",
        ]
        df = df[[c for c in keep if c in df.columns]].copy()
        parts.append(df)

        time.sleep(sleep_s)

    out = paths.processed / f"team_playtype_season_{season}.csv"
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(out, index=False)
    return combined

