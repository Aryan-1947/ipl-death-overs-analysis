"""
Context-adjusted death-overs ratings.

Raw numbers are confounded: a team that plays half its games at Chinnaswamy looks great with the bat,
and a team that faces weak batting lineups looks great with the ball. This module fits one model

    runs-per-over (death phase of one innings)
        = intercept + chasing + batting_team + bowling_team + venue + season

by weighted ridge regression (weight = legal balls), so each team's effect is measured
*holding venue, innings, opponent and season fixed*. Ridge shrinks thin-sample teams/venues toward
average, and a match-level bootstrap gives uncertainty intervals.

Interpretation (all in runs per over, death overs only):
  bat_effect   > 0 : scores more than an average team would in the same context
  bowl_effect  > 0 : concedes fewer than an average team would in the same context
  adj_net = bat_effect + bowl_effect
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def innings_level(death: pd.DataFrame) -> pd.DataFrame:
    g = (death.groupby(["match_id", "innings", "season", "venue", "batting_team", "bowling_team"], as_index=False)
              .agg(runs=("total_runs", "sum"), balls=("legal", "sum")))
    g = g[g["balls"] > 0].copy()
    g["rpo"] = g["runs"] / g["balls"] * 6
    g["chasing"] = (g["innings"] == 2).astype(float)
    return g.reset_index(drop=True)


def _design(g: pd.DataFrame, min_venue_innings: int):
    g = g.copy()
    counts = g["venue"].value_counts()
    g["venue_g"] = np.where(g["venue"].map(counts) >= min_venue_innings, g["venue"], "Other venues")
    teams = sorted(set(g["batting_team"]) | set(g["bowling_team"]))
    venues = sorted(g["venue_g"].unique())
    seasons = sorted(g["season"].unique())[1:]  # first season is the baseline

    n = len(g)
    cols = [np.ones(n), g["chasing"].to_numpy(float)]
    names, ridge = ["intercept", "chasing"], [0.0, 0.0]
    for t in teams:
        cols.append((g["batting_team"] == t).to_numpy(float)); names.append(f"bat::{t}"); ridge.append(1.0)
    for t in teams:
        cols.append((g["bowling_team"] == t).to_numpy(float)); names.append(f"bowl::{t}"); ridge.append(1.0)
    for v in venues:
        cols.append((g["venue_g"] == v).to_numpy(float)); names.append(f"venue::{v}"); ridge.append(1.0)
    for s in seasons:  # season = scoring-inflation control, left unpenalised
        cols.append((g["season"] == s).to_numpy(float)); names.append(f"season::{s}"); ridge.append(0.0)
    return g, np.column_stack(cols), names, np.array(ridge), teams, venues


def _solve(X, y, w, ridge, lam):
    A = X.T @ (X * w[:, None]) + np.diag(ridge * lam) + 1e-8 * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ (w * y))


def _team_effects(beta, names, teams):
    ix = {n: i for i, n in enumerate(names)}
    bat = np.array([beta[ix[f"bat::{t}"]] for t in teams])
    bowl = np.array([beta[ix[f"bowl::{t}"]] for t in teams])
    bat_eff = bat - bat.mean()
    bowl_eff = -(bowl - bowl.mean())  # flip sign: positive = runs saved
    return bat_eff, bowl_eff


def fit_adjusted(death: pd.DataFrame, n_overs: int = 4, ridge: float = 100.0,
                 min_venue_innings: int = 20, bootstrap: int = 200, seed: int = 7):
    """
    Returns (team_df, venue_df, chase_effect).
    ridge: prior strength in 'balls' (100 ~ 4 innings of evidence pulling a team to average).
    """
    g = innings_level(death)
    if g["match_id"].nunique() < 30:
        raise ValueError("Too few matches for a stable adjusted model (need ~30+).")
    g, X, names, pen, teams, venues = _design(g, min_venue_innings)
    y, w = g["rpo"].to_numpy(), g["balls"].to_numpy(float)

    beta = _solve(X, y, w, pen, ridge)
    bat_eff, bowl_eff = _team_effects(beta, names, teams)
    net = bat_eff + bowl_eff

    lo = hi = np.full(len(teams), np.nan)
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        groups = g.groupby("match_id").indices
        keys = list(groups)
        draws = np.empty((bootstrap, len(teams)))
        for b in range(bootstrap):
            pick = rng.integers(0, len(keys), len(keys))
            idx = np.concatenate([groups[keys[i]] for i in pick])
            bb = _solve(X[idx], y[idx], w[idx], pen, ridge)
            be, bw = _team_effects(bb, names, teams)
            draws[b] = be + bw
        lo, hi = np.percentile(draws, [5, 95], axis=0)

    bat_n = g.groupby("batting_team").size()
    bowl_n = g.groupby("bowling_team").size()
    team_df = pd.DataFrame({
        "team": teams, "bat_effect": bat_eff, "bowl_effect": bowl_eff, "adj_net": net,
        "adj_lo": lo, "adj_hi": hi,
        "adj_net_runs_per_match": net * n_overs,
        "bat_innings": [int(bat_n.get(t, 0)) for t in teams],
        "bowl_innings": [int(bowl_n.get(t, 0)) for t in teams],
    }).sort_values("adj_net", ascending=False).reset_index(drop=True)

    ix = {n: i for i, n in enumerate(names)}
    vcount = g["venue_g"].value_counts()
    v_raw = np.array([beta[ix[f"venue::{v}"]] for v in venues])
    v_w = np.array([vcount[v] for v in venues], float)
    venue_df = pd.DataFrame({"venue": venues, "effect": v_raw - np.average(v_raw, weights=v_w),
                             "innings": v_w.astype(int)})
    venue_df = venue_df.sort_values("effect", ascending=False).reset_index(drop=True)
    return team_df, venue_df, float(beta[ix["chasing"]])
