# Transfer Fit NBA — PROJECT STATE

## Goal (MVP)
Streamlit app that lets a user:
- select a team (prefill its profile)
- adjust playstyle sliders (6 dims + playtypes)
- get a shortlist of players with fit score + player card + radar/spider comparison
- frame market problem via transfer success/outcome (needs multi-season ingestion)

## Repo
- Root: transfer-fit-nba
- Python env: .venv + Jupyter kernel "transfer-fit-nba"

## Seasons in scope
- Completed and ingested: 2023-24 (Regular Season)
- Next to ingest: 2022-23 and 2024-25
- Max seasons for MVP: 3–4

## Data sources
- nba_api endpoints used:
  - LeagueGameFinder (team + player game logs)
  - ShotChartDetail (shots)
  - pbp (raw pbp saved per-game → processed pbp_event CSV)
  - SynergyPlayTypes (playtypes) — currently unstable due to read timeouts
- stats.nba.com throttling suspected (timeouts). Plan: retry later on different IP or with slower pacing.

## Directory layout
- data/raw/
  - pbp/season=2023-24/ (per-game CSVs)
  - shots/season=2023-24/ (per-game or per-team CSVs)
- data/processed/
  - teams.csv, players.csv
  - team_game_fact_2023-24_regular.csv
  - player_game_fact_2023-24_regular.csv
  - shot_event_2023-24_regular.csv (raw-ish shots)
  - player_shot_profile_2023-24.csv + percentiles
  - team_shot_profile_2023-24.csv + percentiles
  - player_creation_season_2023-24_usage.csv (USG-like + poss metrics)
  - player_assist_shotbucket_2023-24.csv
  - player_vs_defense_tier_2023-24.csv
  - player_dim.csv (positions)
  - team_strength_2023-24.csv (off/def/net + flags)
  - team_playtype_season_2023-24_enriched.csv
  - player_card_2023-24_v9_with_denominators.csv (latest player card)
  - playtype_dict.csv
  - (attempted) player_playtype_season_2023-24.csv, team_playtype_season_2023-24.csv
  - (attempted) failed_playtypes_2023-24.csv

## Key “player card” metrics included (high level)
- Shooting efficiency: ts_pct, efg_pct + position percentiles
- Shot diet: rim/mid/3/corner rates + percentiles and rates
- Creation/usage: player_poss_used, player_poss_decided, usg_pct, poss_decided_pct
- Rates: per100_min and extra denominators (planned: per100_poss_used/decided)
- Assist splits by shot bucket (per100 poss used) + percentiles
- Physicality: oreb_rate, ft_rate (+ percentiles)
- Defense proxy: stl/blk/pf rates etc (team + player where available)
- Vs strong defenses proxy: aggregates player game performance vs teams flagged strong defense

## Transfers
- Midseason transfers detected for 2023-24 (transfer_event_2023-24_midseason.csv)
- Cross-season transfer logic postponed until multiple seasons ingested

## Playtypes (tactics module)
Target tables:
- playtype_dict [playtype_label, playtype_api, type_grouping, side]
- player_playtype_season (from SynergyPlayTypes "P")
- team_playtype_season (from SynergyPlayTypes "T")
- player_playtype_vs_team (postponed; MVP proxy uses vs-strong-defense overall instead)
Current blocker:
- ReadTimeout / KeyError 'resultSet' on some playtypes; batching + caching implemented but unstable.

## Current blockers / issues
- stats.nba.com SynergyPlayTypes read timeouts; solution: slower pacing, new IP, run from terminal script, rerun failures only.
- Need to parameterize ingestion by season and loop 2022-23 + 2024-25.

## Next actions (ordered)
1) Finalize playtypes for 2023-24 (batch + cache, rerun failures).
2) Create season-parameterized ingestion flow (preferably in src/ + thin notebooks).
3) Ingest 2022-23 and 2024-25 with same pipeline.
4) Build transfers across seasons + transfer_outcome.
5) Streamlit app: team select → sliders → shortlist → player cards → compare.
