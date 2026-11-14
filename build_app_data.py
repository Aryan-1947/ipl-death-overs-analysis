"""
Compress the parsed Cricsheet data into one small file for the Streamlit app.

    python build_app_data.py

Creates app_data/ipl_deliveries.csv.gz (a few MB). Commit this file to GitHub so the app can be deployed
(the big raw JSON in data/ stays git-ignored). Data: Cricsheet, ODC-By.
"""
import argparse
from pathlib import Path

from death_overs.parse import load_deliveries

KEEP = ["match_id", "season", "innings", "venue", "batting_team", "bowling_team", "over_no", "bowler",
        "total_runs", "bowler_runs", "legal", "is_dot", "is_boundary", "is_dismissal", "is_bowler_wicket"]
BOOLS = ["legal", "is_dot", "is_boundary", "is_dismissal", "is_bowler_wicket"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="app_data/ipl_deliveries.csv.gz")
    ap.add_argument("--refresh", action="store_true", help="re-parse the JSON files")
    a = ap.parse_args()

    cache = Path(a.raw).parent / f"{Path(a.raw).name}_deliveries.csv"
    df = load_deliveries(a.raw, cache=cache, refresh=a.refresh)
    df = df[KEEP].copy()
    for c in BOOLS:
        df[c] = df[c].astype("int8")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, compression="gzip")
    print(f"Wrote {len(df):,} deliveries from {df['match_id'].nunique()} matches "
          f"to {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
