"""
Streamlit dashboard: IPL death-overs efficiency (raw + context-adjusted).

    python build_app_data.py      # once, creates app_data/ipl_deliveries.csv.gz
    streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from death_overs import adjust, metrics, plots

st.set_page_config(page_title="IPL death-overs efficiency", page_icon="🏏", layout="wide")

APP_DATA = Path("app_data/ipl_deliveries.csv.gz")
FALLBACK = Path("data/raw_deliveries.csv")


@st.cache_resource(show_spinner="Loading data ...")
def load_data() -> pd.DataFrame | None:
    if APP_DATA.exists():
        return pd.read_csv(APP_DATA)
    if FALLBACK.exists():
        return pd.read_csv(FALLBACK)
    return None


@st.cache_data(show_spinner=False)
def render(plot_name: str, *args, **kwargs) -> bytes:
    """Draw one of the project's matplotlib charts and return PNG bytes."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "chart.png"
        getattr(plots, plot_name)(*args, out=path, **kwargs)
        return path.read_bytes()


@st.cache_data(show_spinner="Fitting the context-adjusted model ...")
def analyse(_df: pd.DataFrame, lo: int, hi: int, start: int, end: int, ridge: float, boot: int) -> dict:
    death = metrics.death_phase(_df, start, end, lo, hi)
    n_overs = end - start + 1
    res: dict = {"death": death, "n_overs": n_overs, "error": None}
    if death.empty:
        res["error"] = "No deliveries in this selection."
        return res

    eff_all = metrics.team_efficiency(death, by_season=False)
    eff_all = eff_all[eff_all["total_balls"] >= 75 * n_overs]  # same cut-off as the CLI (300 balls for 4 overs)
    res["eff_all"] = eff_all
    res["eff_season"] = metrics.team_efficiency(death, by_season=True)
    res["obo"] = metrics.over_by_over(death)
    split = metrics.context_split(death)
    res["split"] = split[split["team"].isin(eff_all["team"])]
    try:
        adj, venues, chase = adjust.fit_adjusted(death, n_overs=n_overs, ridge=ridge, bootstrap=boot)
        adj = adj[adj["team"].isin(eff_all["team"])].merge(
            eff_all[["team", "net_rr"]].rename(columns={"net_rr": "raw_net_rr"}), on="team")
        adj["raw_rank"] = adj["raw_net_rr"].rank(ascending=False).astype(int)
        adj["adj_rank"] = adj["adj_net"].rank(ascending=False).astype(int)
        res.update(adj=adj, venues=venues, chase=chase)
    except ValueError as exc:  # too few matches for the model
        res["model_error"] = str(exc)
    return res


# ------------------------------------------------------------------ data + sidebar
df = load_data()
if df is None:
    st.error("No data found. Run `python build_app_data.py` first (after `python download_data.py`).")
    st.stop()

seasons = sorted(int(s) for s in df["season"].unique())
st.sidebar.title("🏏 Controls")
lo, hi = st.sidebar.select_slider("Seasons", options=seasons,
                                  value=(2018 if 2018 in seasons else seasons[0], seasons[-1]))
start, end = st.sidebar.slider("Overs (1-20)", 1, 20, (17, 20))
ridge = st.sidebar.slider("Shrinkage (ridge, in balls)", 20, 400, 100, step=20,
                          help="Higher = teams/venues with little data are pulled harder towards average.")
boot = st.sidebar.selectbox("Bootstrap draws (uncertainty bars)", [0, 100, 200], index=1,
                            help="More draws = smoother intervals but slower.")
st.sidebar.caption("Data: Cricsheet (cricsheet.org), Open Data Commons Attribution License. "
                   "Personal project built on public scorecard data.")

st.title("IPL death-overs efficiency")
phase = "death overs" if (start, end) == (17, 20) else f"overs {start}-{end}"
sub = f"IPL {lo}-{hi}, overs {start}-{end}"
st.caption(f"Who really scores and defends best in the {phase}, once venue, innings, opponent and season are accounted for?")

R = analyse(df, lo, hi, start, end, float(ridge), int(boot))
if R["error"]:
    st.warning(R["error"])
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Matches in selection", f"{R['death']['match_id'].nunique():,}")
c2.metric("Deliveries", f"{len(R['death']):,}")
if "chase" in R:
    c3.metric("Chasing innings vs batting first", f"{R['chase']:+.2f} runs/over")

tabs = st.tabs(["Team ratings", "Venues", "Chasing vs defending", "Team profile", "Bowlers", "Method & limits"])

# ------------------------------------------------------------------ 1. team ratings
with tabs[0]:
    if "model_error" in R:
        st.warning(f"Adjusted model unavailable: {R['model_error']} Try a wider season range.")
    else:
        adj = R["adj"]
        st.image(render("adjusted_vs_raw", adj, subtitle=sub))
        if adj["adj_lo"].notna().all():
            above = adj.loc[adj["adj_lo"] > 0, "team"].tolist()
            below = adj.loc[adj["adj_hi"] < 0, "team"].tolist()
            msg = (f"**Clearly above average:** {', '.join(above) or 'none'}.  \n"
                   f"**Clearly below average:** {', '.join(below) or 'none'}.  \n"
                   f"All other teams' intervals include zero, so they cannot be told apart from an average side.")
            st.info(msg)
        st.markdown("**Adjusted** = batting effect + bowling effect in runs per over versus an average team, "
                    "holding venue, innings, opponent and season fixed. **Raw** = batting run rate minus economy.")
        show = adj[["team", "bat_effect", "bowl_effect", "adj_net", "adj_lo", "adj_hi",
                    "raw_net_rr", "raw_rank", "adj_rank"]].round(2)
        st.dataframe(show, hide_index=True)

# ------------------------------------------------------------------ 2. venues
with tabs[1]:
    if "venues" not in R:
        st.info("Needs the adjusted model; widen the season range.")
    else:
        v = R["venues"]
        if (v["venue"] != "Other venues").sum() < 2:
            st.info("Too few venues with enough innings in this selection. Widen the seasons.")
        else:
            st.image(render("venue_effects", v))
            st.caption("Runs per over vs an average venue, after controlling for team, innings and season. "
                       "Venues with n < ~40 innings are noisy. Rarely used venues are pooled as 'Other venues'.")
            st.dataframe(v.round(2), hide_index=True)

# ------------------------------------------------------------------ 3. chase vs defend
with tabs[2]:
    if R["split"].empty:
        st.info("Not enough data in this selection.")
    else:
        st.image(render("chase_vs_defend", R["split"]))
        st.caption("Dashed lines are the average of the teams shown. Chases end early once the target is reached, "
                   "so chasing death-overs are not like-for-like with first-innings ones: read this as descriptive.")
        st.dataframe(R["split"].round(2), hide_index=True)

# ------------------------------------------------------------------ 4. team profile
with tabs[3]:
    teams = sorted(set(R["eff_all"]["team"]))
    if not teams:
        st.info("No team has enough balls in this selection.")
    else:
        team = st.selectbox("Team", teams)
        if "adj" in R and team in set(R["adj"]["team"]):
            row = R["adj"].set_index("team").loc[team]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Adjusted net", f"{row['adj_net']:+.2f}")
            m2.metric("Batting effect", f"{row['bat_effect']:+.2f}")
            m3.metric("Bowling effect", f"{row['bowl_effect']:+.2f}")
            m4.metric("Rank (adj / raw)", f"{int(row['adj_rank'])} / {int(row['raw_rank'])}")
        st.image(render("team_profile", R["eff_season"], R["obo"], team))

# ------------------------------------------------------------------ 5. bowlers
with tabs[4]:
    min_b = st.slider("Minimum balls bowled in this phase", 30, 300, 120, step=10)
    bt = metrics.bowler_table(R["death"], min_b)
    if bt.empty:
        st.info("No bowlers meet that minimum. Lower it or widen the selection.")
    else:
        st.image(render("bowler_scatter", bt))
        cols = ["bowler", "balls", "wickets", "economy", "dot_pct", "boundary_pct", "bowling_sr"]
        st.dataframe(bt[cols].round(2), hide_index=True)
    st.caption("Bowler numbers are raw: not adjusted for venue or opposition, and a bowler's team may change over time.")

# ------------------------------------------------------------------ 6. method
with tabs[5]:
    st.markdown("""
**Data.** Ball-by-ball IPL data from [Cricsheet](https://cricsheet.org) (Open Data Commons Attribution License).
Super overs and rain-shortened matches are excluded. Franchise renames are merged.

**Raw metrics.** Batting run rate, bowling economy (byes/leg byes not charged to bowlers), boundary %, dot %,
and *net run rate* = batting RR minus bowling economy.

**Adjusted model.** One weighted ridge regression (weight = legal balls) of runs per over in each innings' phase on
batting team, bowling team, venue, chasing vs setting, and season. Each team's effect is therefore measured against
an average team in the same context. Intervals are 90% bootstrap over matches.

**Limits.**
- Scorecard data only: no ball-tracking, so nothing on release points, pace or pitch maps.
- Squad changes between seasons, dew, toss and match situation (wickets in hand, required rate) are not controlled.
- Chasing innings are truncated when the target is reached.
- Effects are linear and additive; intervals reflect sampling noise, not model misspecification.
- Most differences between mid-table teams are within noise. That is a finding, not a flaw.
""")
