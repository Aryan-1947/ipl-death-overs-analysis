"""Death-overs metrics: batting, bowling, and a combined efficiency score."""
from __future__ import annotations

import numpy as np
import pandas as pd


def death_phase(df: pd.DataFrame, start_over: int = 17, end_over: int = 20,
                from_season: int | None = None, to_season: int | None = None) -> pd.DataFrame:
    """Filter to the death overs (1-indexed, inclusive) and optional season range."""
    out = df[(df["over_no"] >= start_over) & (df["over_no"] <= end_over)]
    if from_season is not None:
        out = out[out["season"] >= from_season]
    if to_season is not None:
        out = out[out["season"] <= to_season]
    return out.copy()


def _summary(df: pd.DataFrame, keys: list[str], runs_col: str, wkt_col: str) -> pd.DataFrame:
    out = (df.groupby(keys)
             .agg(innings=("match_id", "nunique"),
                  runs=(runs_col, "sum"),
                  balls=("legal", "sum"),
                  wickets=(wkt_col, "sum"),
                  boundaries=("is_boundary", "sum"),
                  dots=("is_dot", "sum"))
             .reset_index())
    out["rate"] = out["runs"] / out["balls"] * 6          # run rate (batting) / economy (bowling)
    out["boundary_pct"] = out["boundaries"] / out["balls"] * 100
    out["dot_pct"] = out["dots"] / out["balls"] * 100
    out["balls_per_wkt"] = out["balls"] / out["wickets"].replace(0, np.nan)
    return out


def team_efficiency(death: pd.DataFrame, by_season: bool = True) -> pd.DataFrame:
    """
    One row per team (and season). Columns prefixed bat_/bowl_.
    net_rr = batting run rate - bowling economy in the death overs (higher = better).
    """
    keys_extra = ["season"] if by_season else []
    bat = _summary(death, keys_extra + ["batting_team"], "total_runs", "is_dismissal")
    bowl = _summary(death, keys_extra + ["bowling_team"], "bowler_runs", "is_bowler_wicket")
    bat = bat.rename(columns={"batting_team": "team"}).add_prefix("bat_").rename(
        columns={"bat_team": "team", "bat_season": "season"})
    bowl = bowl.rename(columns={"bowling_team": "team"}).add_prefix("bowl_").rename(
        columns={"bowl_team": "team", "bowl_season": "season"})
    eff = bat.merge(bowl, on=keys_extra + ["team"], how="inner")
    eff["net_rr"] = eff["bat_rate"] - eff["bowl_rate"]
    eff["total_balls"] = eff["bat_balls"] + eff["bowl_balls"]
    return eff.sort_values("net_rr", ascending=False).reset_index(drop=True)


def over_by_over(death: pd.DataFrame) -> pd.DataFrame:
    """Batting run rate in each death over, per team, plus a league row."""
    team = (death.groupby(["batting_team", "over_no"])
                 .agg(runs=("total_runs", "sum"), balls=("legal", "sum")).reset_index())
    team["run_rate"] = team["runs"] / team["balls"] * 6
    league = (death.groupby("over_no").agg(runs=("total_runs", "sum"), balls=("legal", "sum")).reset_index())
    league["run_rate"] = league["runs"] / league["balls"] * 6
    league["batting_team"] = "League average"
    return pd.concat([team, league], ignore_index=True)


def bowler_table(death: pd.DataFrame, min_balls: int = 120) -> pd.DataFrame:
    """Death-overs bowler leaderboard."""
    t = _summary(death, ["bowler"], "bowler_runs", "is_bowler_wicket")
    t = t[t["balls"] >= min_balls].rename(columns={"rate": "economy"})
    t["bowling_sr"] = t["balls_per_wkt"]
    return t.sort_values("economy").reset_index(drop=True)


def context_split(death: pd.DataFrame) -> pd.DataFrame:
    """
    Split by innings.
      bat_first_rr   : batting rate when setting a target (innings 1)
      chasing_rr     : batting rate when chasing (innings 2)
      bowl_first_econ: economy when bowling first (innings 1)
      defending_econ : economy when defending a target (innings 2)
    """
    def rate(df, team_col, runs_col):
        t = (df.groupby([team_col, "innings"])
               .agg(runs=(runs_col, "sum"), balls=("legal", "sum")).reset_index())
        t["rate"] = t["runs"] / t["balls"] * 6
        p = t.pivot(index=team_col, columns="innings", values="rate")
        p.index.name = "team"
        return p.rename_axis(columns=None)

    bat = rate(death, "batting_team", "total_runs").rename(columns={1: "bat_first_rr", 2: "chasing_rr"})
    bowl = rate(death, "bowling_team", "bowler_runs").rename(columns={1: "bowl_first_econ", 2: "defending_econ"})
    return bat.join(bowl, how="inner").reset_index()
