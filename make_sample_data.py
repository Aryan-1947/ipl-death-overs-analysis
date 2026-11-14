"""
Generate SYNTHETIC matches in the Cricsheet JSON schema so the pipeline can be tried offline.
These are NOT real IPL results. Use `python download_data.py` for the real data.
"""
import json
import random
from pathlib import Path

DEST = Path("data/sample")
# team -> (batting skill, bowling skill), roughly -1..1
TEAMS = {
    "Chennai Super Kings": (0.5, 0.4), "Mumbai Indians": (0.6, 0.5),
    "Royal Challengers Bangalore": (0.4, -0.5), "Kolkata Knight Riders": (0.1, 0.2),
    "Rajasthan Royals": (0.0, 0.1), "Delhi Daredevils": (-0.1, 0.0),
    "Kings XI Punjab": (-0.2, -0.4), "Sunrisers Hyderabad": (-0.3, 0.5),
}
# venue -> how much it inflates (+) or suppresses (-) death-over scoring (synthetic)
VENUES = {
    "Wankhede Stadium, Mumbai": 0.6, "MA Chidambaram Stadium, Chepauk, Chennai": -0.6,
    "M Chinnaswamy Stadium, Bengaluru": 0.9, "Eden Gardens, Kolkata": 0.1,
    "Feroz Shah Kotla": -0.2, "Punjab Cricket Association Stadium, Mohali": 0.3,
    "Rajiv Gandhi International Stadium, Uppal": -0.3, "Sawai Mansingh Stadium, Jaipur": 0.0,
}
# Teams play ~40% of their games at home, so venue is confounded with team (as in real IPL data).
HOME = {
    "Mumbai Indians": "Wankhede Stadium, Mumbai", "Chennai Super Kings": "MA Chidambaram Stadium, Chepauk, Chennai",
    "Royal Challengers Bangalore": "M Chinnaswamy Stadium, Bengaluru", "Kolkata Knight Riders": "Eden Gardens, Kolkata",
    "Delhi Daredevils": "Feroz Shah Kotla", "Kings XI Punjab": "Punjab Cricket Association Stadium, Mohali",
    "Sunrisers Hyderabad": "Rajiv Gandhi International Stadium, Uppal", "Rajasthan Royals": "Sawai Mansingh Stadium, Jaipur",
}
CHASE_BOOST = 0.3  # chasing sides score slightly faster in the death (synthetic)
OUTCOMES = [0, 1, 2, 3, 4, 6, "W"]


def probs(over: int, diff: float) -> list[float]:
    p = [0.27, 0.36, 0.08, 0.01, 0.12, 0.08, 0.08] if over >= 16 else [0.37, 0.36, 0.07, 0.01, 0.11, 0.05, 0.03]
    p[0] -= 0.03 * diff; p[5] += 0.02 * diff; p[4] += 0.01 * diff; p[6] -= 0.015 * diff
    return [max(x, 0.005) for x in p]


def innings(bat: str, bowl: str, rng: random.Random, venue_boost: float = 0.0, chase_boost: float = 0.0) -> dict:
    diff = TEAMS[bat][0] - TEAMS[bowl][1] + venue_boost + chase_boost
    batters = [f"{bat[:3]} Batter {i}" for i in range(1, 12)]
    bowlers = [f"{bowl[:3]} Bowler {i}" for i in range(1, 6)]
    wk, idx, overs = 0, 0, []
    for o in range(20):
        if wk >= 10:
            break
        bw, dels, legal = rng.choice(bowlers), [], 0
        while legal < 6 and wk < 10:
            out = rng.choices(OUTCOMES, probs(o, diff))[0]
            d = {"batter": batters[min(idx, 10)], "bowler": bw, "non_striker": batters[min(idx + 1, 10)]}
            if rng.random() < 0.04:  # wide
                d["runs"] = {"batter": 0, "extras": 1, "total": 1}; d["extras"] = {"wides": 1}
            else:
                legal += 1
                if out == "W":
                    d["runs"] = {"batter": 0, "extras": 0, "total": 0}
                    d["wickets"] = [{"player_out": d["batter"], "kind": rng.choice(["caught", "bowled", "run out", "lbw"])}]
                    wk += 1; idx += 1
                else:
                    d["runs"] = {"batter": out, "extras": 0, "total": out}
            dels.append(d)
        overs.append({"over": o, "deliveries": dels})
    return {"team": bat, "overs": overs}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    rng, n = random.Random(42), 0
    names = list(TEAMS)
    for year in range(2018, 2025):
        for _ in range(60):
            a, b = rng.sample(names, 2)
            r = rng.random()
            venue = HOME[a] if r < 0.4 else HOME[b] if r < 0.8 else rng.choice(list(VENUES))
            vb = VENUES[venue]
            n += 1
            m = {"info": {"match_type": "T20", "overs": 20, "teams": [a, b], "venue": venue,
                          "dates": [f"{year}-04-{rng.randint(1, 28):02d}"]},
                 "innings": [innings(a, b, rng, vb, 0.0), innings(b, a, rng, vb, CHASE_BOOST)]}
            (DEST / f"{1000000 + n}.json").write_text(json.dumps(m))
    print(f"Wrote {n} synthetic matches to {DEST.resolve()}")


if __name__ == "__main__":
    main()
