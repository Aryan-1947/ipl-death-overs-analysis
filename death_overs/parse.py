"""Parse Cricsheet ball-by-ball JSON into one flat, ball-level DataFrame."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Franchise renames, so a franchise is one continuous line across seasons.
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Best-effort venue clean-up (Cricsheet names vary by season). Check the printed venue table for stragglers.
VENUE_ALIASES = {
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Punjab Cricket Association Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium",
}

# Dismissals that are not credited to the bowler.
NON_BOWLER_WICKETS = {"run out", "retired hurt", "retired out", "obstructing the field"}


def norm_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def norm_venue(name: str) -> str:
    base = name.split(",")[0].strip()
    return VENUE_ALIASES.get(base, base)


def parse_match(path: Path) -> list[dict]:
    """Return one dict per delivery for a single Cricsheet match file."""
    with open(path, encoding="utf-8") as f:
        match = json.load(f)

    info = match["info"]
    # Skip rain-shortened games: "death overs" are not comparable there.
    if info.get("overs", 20) != 20:
        return []

    season = int(info["dates"][0][:4])  # calendar year of first match day (2007/08 -> 2008, 2020/21 -> 2020)
    match_id = path.stem
    venue = norm_venue(info.get("venue", "Unknown"))
    teams = info["teams"]
    rows: list[dict] = []

    for inn_no, inn in enumerate(match.get("innings", []), start=1):
        if inn.get("super_over"):
            continue
        bat_raw = inn["team"]
        bowl_raw = next((t for t in teams if t != bat_raw), None)
        if bowl_raw is None:
            continue
        batting, bowling = norm_team(bat_raw), norm_team(bowl_raw)

        for ov in inn.get("overs", []):
            for ball_idx, d in enumerate(ov["deliveries"]):
                runs = d["runs"]
                extras = d.get("extras", {})
                wide = "wides" in extras
                noball = "noballs" in extras
                legal = not (wide or noball)
                wickets = d.get("wickets", [])
                bowler_wkt = any(w["kind"] not in NON_BOWLER_WICKETS for w in wickets)
                # Runs charged to the bowler exclude byes, leg byes and penalty runs.
                bowler_runs = runs["total"] - sum(extras.get(k, 0) for k in ("byes", "legbyes", "penalty"))

                rows.append({
                    "match_id": match_id,
                    "season": season,
                    "venue": venue,
                    "innings": inn_no,
                    "batting_team": batting,
                    "bowling_team": bowling,
                    "over": ov["over"],            # 0-indexed
                    "over_no": ov["over"] + 1,     # 1-indexed (what fans call "the 18th over")
                    "ball_idx": ball_idx,
                    "batter": d["batter"],
                    "bowler": d["bowler"],
                    "batter_runs": runs["batter"],
                    "total_runs": runs["total"],
                    "bowler_runs": bowler_runs,
                    "legal": legal,
                    "is_dot": legal and runs["total"] == 0,
                    "is_boundary": legal and runs["batter"] in (4, 6),
                    "is_dismissal": len(wickets) > 0,
                    "is_bowler_wicket": bowler_wkt,
                })
    return rows


def load_deliveries(raw_dir: str | Path, cache: str | Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load every match in raw_dir (optionally cached as CSV for fast re-runs)."""
    raw_dir = Path(raw_dir)
    if cache and Path(cache).exists() and not refresh:
        cached = pd.read_csv(cache)
        if "venue" in cached.columns:  # older caches (pre-venue) are re-parsed
            return cached

    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No .json files in {raw_dir}. Run `python download_data.py` "
            f"(or `python make_sample_data.py` for a synthetic demo)."
        )
    rows: list[dict] = []
    for i, p in enumerate(files, 1):
        rows.extend(parse_match(p))
        if i % 200 == 0:
            print(f"  parsed {i}/{len(files)} matches")
    df = pd.DataFrame(rows)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    return df
