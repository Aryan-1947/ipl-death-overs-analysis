"""
IPL death-overs efficiency analysis (raw + context-adjusted).

Examples
--------
python run_analysis.py
python run_analysis.py --from-season 2018 --team "Chennai Super Kings"
python run_analysis.py --ridge 200 --bootstrap 500
python run_analysis.py --raw data/sample          # synthetic demo data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from death_overs import adjust, metrics, plots
from death_overs.parse import load_deliveries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", help="folder of Cricsheet JSON files")
    ap.add_argument("--out", default="outputs", help="folder for charts and CSV tables")
    ap.add_argument("--from-season", type=int, default=None)
    ap.add_argument("--to-season", type=int, default=None)
    ap.add_argument("--start-over", type=int, default=17)
    ap.add_argument("--end-over", type=int, default=20)
    ap.add_argument("--team", action="append", help="team for a detailed profile (repeatable)")
    ap.add_argument("--min-balls", type=int, default=300, help="min balls (bat+bowl) for team-level charts")
    ap.add_argument("--min-bowler-balls", type=int, default=120)
    ap.add_argument("--ridge", type=float, default=100.0, help="shrinkage strength (in balls) for the adjusted model")
    ap.add_argument("--min-venue-innings", type=int, default=20, help="venues with fewer innings are pooled as 'Other'")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples for intervals (0 = off)")
    ap.add_argument("--refresh", action="store_true", help="re-parse JSON instead of using the CSV cache")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.raw).parent / f"{Path(a.raw).name}_deliveries.csv"

    print("Loading deliveries ...")
    df = load_deliveries(a.raw, cache=cache, refresh=a.refresh)
    death = metrics.death_phase(df, a.start_over, a.end_over, a.from_season, a.to_season)
    seasons = f"{death['season'].min()}-{death['season'].max()}"
    sub = f"IPL {seasons}, overs {a.start_over}-{a.end_over}"
    n_overs = a.end_over - a.start_over + 1
    print(f"{death['match_id'].nunique()} matches, {len(death):,} deliveries in scope ({sub})")

    # ---- raw metrics ----
    eff_all = metrics.team_efficiency(death, by_season=False)
    eff_all = eff_all[eff_all["total_balls"] >= a.min_balls]
    eff_season = metrics.team_efficiency(death, by_season=True)
    obo = metrics.over_by_over(death)
    bowlers = metrics.bowler_table(death, a.min_bowler_balls)
    split = metrics.context_split(death)
    split = split[split["team"].isin(eff_all["team"])]

    eff_all.to_csv(out / "team_efficiency.csv", index=False)
    eff_season.to_csv(out / "team_efficiency_by_season.csv", index=False)
    bowlers.to_csv(out / "death_bowlers.csv", index=False)
    split.to_csv(out / "chase_vs_defend.csv", index=False)

    plots.net_rr_bar(eff_all, out / "01_net_run_rate.png", sub)
    plots.bat_vs_bowl_scatter(eff_all, out / "02_bat_vs_bowl.png", sub)
    plots.season_heatmap(eff_season, out / "03_season_heatmap.png")
    plots.over_progression(obo, eff_all["team"].head(3).tolist(), out / "04_over_progression.png")
    if len(bowlers):
        plots.bowler_scatter(bowlers, out / "05_top_bowlers.png")
    for team in a.team or []:
        if team not in set(eff_season["team"]):
            print(f"  ! '{team}' not found. Available: {sorted(set(eff_season['team']))}")
            continue
        plots.team_profile(eff_season, obo, team, out / f"06_profile_{team.replace(' ', '_')}.png")
    plots.chase_vs_defend(split, out / "08_chase_vs_defend.png")

    # ---- context-adjusted model ----
    print("Fitting context-adjusted model ...")
    adj, venues, chase = adjust.fit_adjusted(death, n_overs=n_overs, ridge=a.ridge,
                                             min_venue_innings=a.min_venue_innings, bootstrap=a.bootstrap)
    adj = adj[adj["team"].isin(eff_all["team"])].merge(eff_all[["team", "net_rr"]].rename(columns={"net_rr": "raw_net_rr"}), on="team")
    adj["raw_rank"] = adj["raw_net_rr"].rank(ascending=False).astype(int)
    adj["adj_rank"] = adj["adj_net"].rank(ascending=False).astype(int)
    adj.to_csv(out / "team_efficiency_adjusted.csv", index=False)
    venues.to_csv(out / "venue_effects.csv", index=False)
    plots.adjusted_vs_raw(adj, out / "07_adjusted_vs_raw.png", sub)
    plots.venue_effects(venues, out / "09_venue_effects.png")

    pd.set_option("display.width", 180)
    print("\nRAW team death-overs efficiency:")
    print(eff_all[["team", "bat_rate", "bowl_rate", "net_rr", "bat_boundary_pct", "bowl_dot_pct"]].round(2).to_string(index=False))
    print(f"\nCONTEXT-ADJUSTED (runs/over vs average team; chasing innings score {chase:+.2f} runs/over vs batting first):")
    cols = ["team", "bat_effect", "bowl_effect", "adj_net", "adj_lo", "adj_hi", "raw_rank", "adj_rank"]
    print(adj[cols].round(2).to_string(index=False))
    print("\nVenue effects (check for duplicate venue names here):")
    print(venues.round(2).to_string(index=False))
    print(f"\nCharts and CSVs written to {out.resolve()}")


if __name__ == "__main__":
    main()
