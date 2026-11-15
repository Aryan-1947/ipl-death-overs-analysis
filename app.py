"""
Streamlit dashboard: IPL death-overs efficiency (raw + context-adjusted), "control room" theme.

    python build_app_data.py      # once, creates app_data/ipl_deliveries.csv.gz
    python -m streamlit run app.py
"""
from __future__ import annotations

import html
import tempfile
import threading
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from death_overs import adjust, metrics, plots

st.set_page_config(page_title="IPL Death-Overs Lab", page_icon="🏏", layout="wide")

APP_DATA = Path("app_data/ipl_deliveries.csv.gz")
FALLBACK = Path("data/raw_deliveries.csv")
esc = html.escape

# ====================================================================== theme (CSS)
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600;700&display=swap');
.stApp{background:radial-gradient(1100px 560px at 72% -8%,#1d2c68 0%,rgba(7,13,31,0) 62%),linear-gradient(180deg,#0a1230 0%,#060b1c 100%);}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.4rem;max-width:1220px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c1534 0%,#070c1f 100%);border-right:1px solid rgba(245,185,66,.14);}
.brand{text-align:center;font-family:'Bebas Neue',Impact,sans-serif;font-size:31px;letter-spacing:3px;color:#f5b942;line-height:1.05;margin:6px 0 2px;}
.brand-sub{text-align:center;color:#8d99ba;font-size:11px;letter-spacing:1.5px;margin-bottom:14px;}
.fine{color:#7c88aa;font-size:11px;line-height:1.5;margin-top:18px;}
[data-testid="stSidebar"] .stButton button{background:linear-gradient(180deg,#ff9a1a,#f26d00);color:#fff;border:none;border-radius:999px;font-weight:700;padding:.55rem 1rem;box-shadow:0 4px 16px rgba(255,122,0,.35);width:100%;}
[data-testid="stSidebar"] .stButton button:hover{filter:brightness(1.1);color:#fff;}
.ticker{display:flex;align-items:center;background:linear-gradient(90deg,#ff7a00,#ffb000);border-radius:12px;overflow:hidden;margin-bottom:20px;height:42px;}
.ticker-tag{background:#0b1330;color:#ffb000;font-family:'Bebas Neue',Impact,sans-serif;letter-spacing:2px;padding:0 16px;height:100%;display:flex;align-items:center;font-size:15px;white-space:nowrap;}
.ticker-track{overflow:hidden;flex:1;white-space:nowrap;}
.ticker-inner{display:inline-block;padding-left:100%;animation:tick 55s linear infinite;color:#1a1200;font-weight:700;letter-spacing:.4px;font-size:13px;}
@keyframes tick{to{transform:translateX(-100%);}}
[role="tablist"]{gap:8px;border-bottom:1px solid rgba(255,255,255,.08);}
[role="tab"]{background:rgba(255,255,255,.05);border-radius:10px 10px 0 0;padding:8px 18px;color:#c7d0ea;border:none;}
[role="tab"] p{color:#c7d0ea;}
[role="tab"][aria-selected="true"]{background:linear-gradient(180deg,#ff8a00,#e66a00);color:#fff;font-weight:700;}
[role="tab"][aria-selected="true"] p{color:#fff;font-weight:700;}
.react-aria-SelectionIndicator,[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none !important;}
.sec{font-family:'Bebas Neue',Impact,sans-serif;font-size:36px;letter-spacing:1.5px;color:#fff;margin:14px 0 14px;line-height:1.1;}
.sec small{display:block;font-family:Inter,sans-serif;font-size:13px;color:#9aa6c4;letter-spacing:0;margin:4px 0 0;}
.chips{display:flex;gap:12px;margin:0 0 18px;flex-wrap:wrap;}
.chip{padding:10px 18px;border-radius:12px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.09);min-width:150px;}
.chip b{display:block;font-family:'Bebas Neue',Impact,sans-serif;font-size:30px;color:#f5b942;font-weight:400;line-height:1.1;letter-spacing:.5px;}
.chip span{font-size:10px;letter-spacing:1.2px;color:#8d99ba;text-transform:uppercase;}
.shield{width:78px;height:90px;box-sizing:border-box;padding:3px;margin:0 auto;background:linear-gradient(160deg,#ffe08a,#9b6a14);clip-path:polygon(50% 0,100% 11%,100% 60%,50% 100%,0 60%,0 11%);}
.shield-in{width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-family:'Bebas Neue',Impact,sans-serif;font-size:27px;letter-spacing:1px;clip-path:polygon(50% 0,100% 11%,100% 60%,50% 100%,0 60%,0 11%);}
.shield.sm{width:38px;height:44px;padding:2px;}
.shield.sm .shield-in{font-size:14px;letter-spacing:.5px;}
.podium{display:flex;gap:24px;align-items:flex-end;justify-content:center;margin:26px 0 28px;}
.p-card{flex:1;max-width:330px;text-align:center;padding:26px 14px 20px;border-radius:18px;position:relative;background:linear-gradient(160deg,var(--tint),rgba(10,16,40,.92));border:1px solid rgba(255,255,255,.12);}
.p-first{max-width:390px;padding:34px 16px 26px;border:1px solid rgba(245,185,66,.6);box-shadow:0 0 42px rgba(245,185,66,.14);}
.p-first .shield{width:96px;height:111px;}
.p-first .shield-in{font-size:34px;}
.medal{position:absolute;top:-14px;left:50%;transform:translateX(-50%);width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:13px;color:#1a1200;box-shadow:0 2px 8px rgba(0,0,0,.4);}
.m1{background:#f5b942;} .m2{background:#c9d1e0;} .m3{background:#cd8a52;}
.p-name{font-weight:700;color:#fff;margin:12px 0 2px;font-size:16px;}
.p-num{font-family:'Bebas Neue',Impact,sans-serif;font-size:54px;line-height:1.05;color:#fff;letter-spacing:1px;}
.p-first .p-num{font-size:72px;}
.p-sub{font-size:12px;color:#a6b1cf;}
.lb-row{display:grid;grid-template-columns:34px 44px minmax(160px,1.5fr) 3fr 92px 64px 64px 64px;gap:12px;align-items:center;padding:10px 16px;margin-bottom:8px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);}
.lb-rank{font-family:'Bebas Neue',Impact,sans-serif;font-size:27px;color:#c7d0ea;text-align:center;}
.lb-name{font-weight:700;color:#fff;font-size:15px;line-height:1.25;}
.lb-name span{display:block;font-weight:400;font-size:11px;color:#8d99ba;}
.lb-bar{position:relative;height:16px;background:rgba(255,255,255,.07);border-radius:8px;}
.zero{position:absolute;top:-4px;bottom:-4px;width:2px;background:rgba(255,255,255,.5);}
.fill{position:absolute;top:0;bottom:0;border-radius:8px;}
.fill.pos{background:linear-gradient(90deg,#b27a12,#f5b942);}
.fill.neg{background:linear-gradient(90deg,#d6334a,#8a1a2c);}
.whisk{position:absolute;top:50%;height:2px;background:#fff;transform:translateY(-50%);}
.cap{position:absolute;top:1px;bottom:1px;width:2px;background:#fff;}
.lb-big{font-family:'Bebas Neue',Impact,sans-serif;font-size:31px;text-align:center;color:#fff;line-height:1;}
.lb-big span,.lb-stat span{display:block;font-family:Inter,sans-serif;font-size:9px;letter-spacing:1px;color:#8d99ba;}
.lb-stat{text-align:center;font-weight:700;color:#fff;font-size:15px;}
.tag{display:inline-block;font-size:9px;letter-spacing:.8px;padding:1px 7px;border-radius:20px;margin-left:6px;font-weight:700;}
.t-up{background:rgba(46,204,113,.18);color:#5fe39a;}
.t-dn{background:rgba(255,80,100,.18);color:#ff7b8c;}
.t-na{background:rgba(255,255,255,.08);color:#9aa6c4;}
.team-head{display:flex;align-items:center;gap:18px;margin:6px 0 14px;}
.team-head .shield{margin:0;}
.team-head .nm{font-family:'Bebas Neue',Impact,sans-serif;font-size:40px;letter-spacing:1.5px;color:#fff;line-height:1;}
[data-testid="stImage"] img{border-radius:16px;border:1px solid rgba(245,185,66,.2);background:rgba(255,255,255,.035);padding:10px;}
[data-testid="stExpander"]{border:1px solid rgba(255,255,255,.08);border-radius:12px;background:rgba(255,255,255,.025);}
@media (max-width:900px){.lb-row{grid-template-columns:30px 40px 1fr 80px;}.lb-bar,.lb-stat{display:none;}.podium{flex-direction:column;align-items:center;}.p-card,.p-first{max-width:100%;width:100%;}.podium .p-first{order:-1;}}
</style>
"""

# ====================================================================== helpers
ABBR = {
    "Mumbai Indians": "MI", "Chennai Super Kings": "CSK", "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR", "Rajasthan Royals": "RR", "Delhi Capitals": "DC",
    "Punjab Kings": "PBKS", "Sunrisers Hyderabad": "SRH", "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG", "Deccan Chargers": "DCH", "Rising Pune Supergiant": "RPS",
    "Gujarat Lions": "GL", "Kochi Tuskers Kerala": "KTK", "Pune Warriors": "PWI",
}
FALLBACK_COLOR = "#5b6b94"


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _lum(h: str) -> float:
    r, g, b = _rgb(h)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _darken(h: str, f: float) -> str:
    r, g, b = (int(c * f) for c in _rgb(h))
    return f"#{r:02x}{g:02x}{b:02x}"


def _rgba(h: str, a: float) -> str:
    r, g, b = _rgb(h)
    return f"rgba({r},{g},{b},{a})"


def team_color(team: str) -> str:
    return plots.TEAM_COLORS.get(team, FALLBACK_COLOR)


def abbr(name: str) -> str:
    if name in ABBR:
        return ABBR[name]
    parts = [p for p in name.replace(".", " ").split() if p]
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else name[:3].upper()


def shield(label: str, color: str, small: bool = False) -> str:
    txt = "#161616" if _lum(color) > 0.62 else "#ffffff"
    return (f'<div class="shield{" sm" if small else ""}"><div class="shield-in" '
            f'style="background:linear-gradient(160deg,{color},{_darken(color, 0.5)});color:{txt}">{esc(label)}</div></div>')


def sec(title: str, small: str = "") -> None:
    st.markdown(f'<div class="sec">{esc(title)}<small>{esc(small)}</small></div>', unsafe_allow_html=True)


def chips(items: list[tuple[str, str, str]]) -> None:
    body = "".join(f'<div class="chip"><b style="color:{c}">{esc(v)}</b><span>{esc(lab)}</span></div>' for v, lab, c in items)
    st.markdown(f'<div class="chips">{body}</div>', unsafe_allow_html=True)


def podium(cards: list[dict]) -> str:
    order = {3: [1, 0, 2], 2: [1, 0]}.get(len(cards), list(range(len(cards))))
    out = []
    for i in order:
        c = cards[i]
        out.append(
            f'<div class="p-card{" p-first" if i == 0 else ""}" style="--tint:{c["tint"]}">'
            f'<div class="medal m{i + 1}">{i + 1}</div>{c["shield"]}'
            f'<div class="p-name">{esc(c["name"])}</div><div class="p-num">{c["big"]}</div>'
            f'<div class="p-sub">{c["sub"]}</div></div>')
    return '<div class="podium">' + "".join(out) + "</div>"


def leaderboard(adj: pd.DataFrame) -> str:
    a = adj.sort_values("adj_net", ascending=False).reset_index(drop=True)
    has_ci = a["adj_lo"].notna().all()
    lows = a["adj_lo"] if has_ci else a["adj_net"]
    highs = a["adj_hi"] if has_ci else a["adj_net"]
    lo_s, hi_s = min(0.0, float(lows.min())), max(0.0, float(highs.max()))
    pad = (hi_s - lo_s) * 0.06 or 0.5
    lo_s, hi_s = lo_s - pad, hi_s + pad

    def pos(x: float) -> float:
        return (x - lo_s) / (hi_s - lo_s) * 100

    rows = []
    for i, r in a.iterrows():
        v = float(r["adj_net"])
        c = team_color(r["team"])
        if has_ci and r["adj_lo"] > 0:
            tag = '<span class="tag t-up">CLEARLY ABOVE AVG</span>'
        elif has_ci and r["adj_hi"] < 0:
            tag = '<span class="tag t-dn">CLEARLY BELOW AVG</span>'
        else:
            tag = '<span class="tag t-na">WITHIN NOISE</span>' if has_ci else ""
        left, width = pos(min(0, v)), abs(pos(v) - pos(0))
        bar = (f'<div class="fill {"pos" if v >= 0 else "neg"}" style="left:{left:.2f}%;width:{width:.2f}%"></div>'
               f'<div class="zero" style="left:{pos(0):.2f}%"></div>')
        if has_ci:
            l, h = pos(float(r["adj_lo"])), pos(float(r["adj_hi"]))
            bar += (f'<div class="whisk" style="left:{l:.2f}%;width:{h - l:.2f}%"></div>'
                    f'<div class="cap" style="left:{l:.2f}%"></div><div class="cap" style="left:{h:.2f}%"></div>')
        rows.append(
            f'<div class="lb-row"><div class="lb-rank">{i + 1}</div>{shield(abbr(r["team"]), c, small=True)}'
            f'<div class="lb-name">{esc(r["team"])}<span>raw #{int(r["raw_rank"])} → adjusted #{int(r["adj_rank"])} {tag}</span></div>'
            f'<div class="lb-bar">{bar}</div>'
            f'<div class="lb-big">{v:+.2f}<span>ADJ NET</span></div>'
            f'<div class="lb-stat">{r["bat_effect"]:+.2f}<span>BAT</span></div>'
            f'<div class="lb-stat">{r["bowl_effect"]:+.2f}<span>BOWL</span></div>'
            f'<div class="lb-stat">{r["raw_net_rr"]:+.2f}<span>RAW NET</span></div></div>')
    return "".join(rows)


# ====================================================================== dark matplotlib charts
LIGHT, MUTED, EDGE = "#e8ecf7", "#9aa6c4", "#3a4770"
DARK_RC = {
    "text.color": LIGHT, "axes.labelcolor": LIGHT, "axes.edgecolor": EDGE, "axes.titlecolor": LIGHT,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#ffffff", "grid.alpha": 0.08,
    "legend.facecolor": "#0f1a38", "legend.edgecolor": EDGE, "legend.labelcolor": LIGHT,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "savefig.transparent": True,
}
_LOCK = threading.Lock()


def _dark_save(fig, path):
    """Drop-in for plots._save: recolour hard-coded black lines, save transparent."""
    for ax in fig.axes:
        for ln in ax.lines:
            try:
                if mcolors.to_hex(ln.get_color()) == "#000000":
                    ln.set_color(LIGHT)
            except (ValueError, TypeError):
                pass
        leg = ax.get_legend()
        if leg is not None:
            for h in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
                for getter, setter in (("get_color", "set_color"), ("get_markerfacecolor", "set_markerfacecolor"),
                                       ("get_markeredgecolor", "set_markeredgecolor")):
                    try:
                        if mcolors.to_hex(getattr(h, getter)()) == "#000000":
                            getattr(h, setter)(LIGHT)
                    except (ValueError, TypeError, AttributeError):
                        pass
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return path


@st.cache_data(show_spinner=False)
def render(plot_name: str, *args, **kwargs) -> bytes:
    """Draw one of the project's charts in the dark theme and return PNG bytes."""
    with _LOCK, tempfile.TemporaryDirectory() as d, matplotlib.rc_context(DARK_RC):
        path = Path(d) / "chart.png"
        original = plots._save
        plots._save = _dark_save
        try:
            getattr(plots, plot_name)(*args, out=path, **kwargs)
        finally:
            plots._save = original
        return path.read_bytes()


# ====================================================================== data + model
@st.cache_resource(show_spinner="Loading data ...")
def load_data() -> pd.DataFrame | None:
    if APP_DATA.exists():
        return pd.read_csv(APP_DATA)
    if FALLBACK.exists():
        return pd.read_csv(FALLBACK)
    return None


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


# ====================================================================== page: sidebar
st.markdown(CSS, unsafe_allow_html=True)

df = load_data()
if df is None:
    st.error("No data found. Run `python build_app_data.py` first (after `python download_data.py`).")
    st.stop()

seasons = sorted(int(s) for s in df["season"].unique())
st.sidebar.markdown('<div class="brand">🏏 DEATH-OVERS LAB</div><div class="brand-sub">IPL · CONTROL ROOM</div>',
                    unsafe_allow_html=True)
lo, hi = st.sidebar.select_slider("Seasons", options=seasons,
                                  value=(2018 if 2018 in seasons else seasons[0], seasons[-1]))
start, end = st.sidebar.slider("Overs (1-20)", 1, 20, (17, 20))
ridge = st.sidebar.slider("Shrinkage (ridge, in balls)", 20, 400, 100, step=20,
                          help="Higher = teams/venues with little data are pulled harder towards average.")
prec = st.sidebar.select_slider("Interval precision", options=["Off", "Fast", "Precise"], value="Fast",
                                help="Bootstrap draws behind the uncertainty bars: Off = none, Fast = 100, Precise = 200.")
boot = {"Off": 0, "Fast": 100, "Precise": 200}[prec]
if st.sidebar.button("↻  Recompute everything"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.markdown(
    '<div class="fine">Data: Cricsheet (cricsheet.org), Open Data Commons Attribution License.<br><br>'
    'Unofficial fan project built on public scorecard data. Not affiliated with the BCCI, the IPL or any franchise. '
    'Team badges are text abbreviations, not official logos.</div>', unsafe_allow_html=True)

R = analyse(df, lo, hi, start, end, float(ridge), int(boot))
if R["error"]:
    st.warning(R["error"])
    st.stop()

sub = f"IPL {lo}-{hi}, overs {start}-{end}"
n_matches = R["death"]["match_id"].nunique()
ticker = (f"ADJUSTED FOR VENUE, INNINGS, OPPONENT AND SEASON  ·  IPL {lo}-{hi}, OVERS {start}-{end}  ·  {n_matches:,} MATCHES  ·  "
          "90% BOOTSTRAP INTERVALS  ·  MOST MID-TABLE GAPS ARE WITHIN NOISE  ·  SCORECARD DATA ONLY, NO TRACKING DATA  ·  "
          "ANALYSIS, NOT PREDICTION")
st.markdown(f'<div class="ticker"><div class="ticker-tag">READ THIS</div><div class="ticker-track">'
            f'<div class="ticker-inner">{esc(ticker)}</div></div></div>', unsafe_allow_html=True)

tabs = st.tabs(["🏆 Team ratings", "🏟 Venues", "⚔ Chasing vs defending", "👕 Team profile", "🎯 Bowlers", "🧪 Method & limits"])

# ---------------------------------------------------------------------- 1. team ratings
with tabs[0]:
    sec("WHO OWNS THE DEATH OVERS?", "adjusted net runs per over vs an average side")
    if "model_error" in R:
        st.warning(f"Adjusted model unavailable: {R['model_error']} Try a wider season range.")
    else:
        adj = R["adj"]
        top = adj.sort_values("adj_net", ascending=False).reset_index(drop=True)
        has_ci = top["adj_lo"].notna().all()
        cards = []
        for _, r in top.head(3).iterrows():
            c = team_color(r["team"])
            rng = f" · range {r['adj_lo']:+.2f} to {r['adj_hi']:+.2f}" if has_ci else ""
            cards.append(dict(name=r["team"], shield=shield(abbr(r["team"]), c), big=f"{r['adj_net']:+.2f}",
                              sub=f"runs/over vs average{rng}", tint=_rgba(c, 0.38)))
        st.markdown(podium(cards), unsafe_allow_html=True)
        st.markdown(leaderboard(adj), unsafe_allow_html=True)
        if has_ci:
            above = adj.loc[adj["adj_lo"] > 0, "team"].tolist()
            below = adj.loc[adj["adj_hi"] < 0, "team"].tolist()
            st.info(f"**Clearly above average:** {', '.join(above) or 'none'}.  \n"
                    f"**Clearly below average:** {', '.join(below) or 'none'}.  \n"
                    "Every other team's interval includes zero, so it cannot be told apart from an average side. "
                    "Bars show the adjusted rating; white whiskers are the 90% bootstrap interval.")
        sec("RAW VS ADJUSTED", "does context change the picture?")
        st.image(render("adjusted_vs_raw", adj, subtitle=sub))
        with st.expander("Full table"):
            show = adj[["team", "bat_effect", "bowl_effect", "adj_net", "adj_lo", "adj_hi",
                        "raw_net_rr", "raw_rank", "adj_rank"]].round(2)
            st.dataframe(show, hide_index=True)

# ---------------------------------------------------------------------- 2. venues
with tabs[1]:
    sec("HITTERS' GROUNDS & BOWLERS' GROUNDS", "death-over runs per over vs an average venue")
    if "venues" not in R:
        st.info("Needs the adjusted model; widen the season range.")
    else:
        v = R["venues"]
        real = v[v["venue"] != "Other venues"]
        if len(real) < 2:
            st.info("Too few venues with enough innings in this selection. Widen the seasons.")
        else:
            solid = real[real["innings"] >= 40]
            solid = solid if len(solid) >= 2 else real
            hot, cold = solid.iloc[0], solid.iloc[-1]
            chips([(f"{hot['effect']:+.2f}", f"Most inflating: {hot['venue']} (n={hot['innings']})", "#5fe39a"),
                   (f"{cold['effect']:+.2f}", f"Most suppressing: {cold['venue']} (n={cold['innings']})", "#ff7b8c")])
            st.image(render("venue_effects", v))
            st.caption("After controlling for team, innings and season. Venues with n < ~40 innings are noisy; "
                       "rarely used venues are pooled as 'Other venues'.")
            with st.expander("Venue table"):
                st.dataframe(v.round(2), hide_index=True)

# ---------------------------------------------------------------------- 3. chase vs defend
with tabs[2]:
    sec("SETTING VS CHASING, BOWLING FIRST VS DEFENDING")
    if R["split"].empty:
        st.info("Not enough data in this selection.")
    else:
        if "chase" in R:
            chips([(f"{R['chase']:+.2f}", "Chasing innings vs batting first · runs/over", "#f5b942")])
        st.image(render("chase_vs_defend", R["split"]))
        st.caption("Dashed lines are the average of the teams shown. Chases end early once the target is reached, "
                   "so chasing death-overs are not like-for-like with first-innings ones: read this as descriptive.")
        with st.expander("Table"):
            st.dataframe(R["split"].round(2), hide_index=True)

# ---------------------------------------------------------------------- 4. team profile
with tabs[3]:
    teams = sorted(set(R["eff_all"]["team"]))
    if not teams:
        st.info("No team has enough balls in this selection.")
    else:
        team = st.selectbox("Pick a team", teams)
        c = team_color(team)
        st.markdown(f'<div class="team-head">{shield(abbr(team), c)}<div class="nm">{esc(team)}</div></div>',
                    unsafe_allow_html=True)
        if "adj" in R and team in set(R["adj"]["team"]):
            row = R["adj"].set_index("team").loc[team]
            chips([(f"{row['adj_net']:+.2f}", "Adjusted net", "#f5b942"),
                   (f"{row['bat_effect']:+.2f}", "Batting effect", "#f5b942"),
                   (f"{row['bowl_effect']:+.2f}", "Bowling effect", "#f5b942"),
                   (f"#{int(row['adj_rank'])}  /  #{int(row['raw_rank'])}", "Adjusted / raw rank", "#ffffff")])
        st.image(render("team_profile", R["eff_season"], R["obo"], team))

# ---------------------------------------------------------------------- 5. bowlers
with tabs[4]:
    sec("DEATH-OVERS BOWLERS", "best economy, minimum balls bowled")
    min_b = st.slider("Minimum balls bowled in this phase", 30, 300, 120, step=10)
    bt = metrics.bowler_table(R["death"], min_b)
    if bt.empty:
        st.info("No bowlers meet that minimum. Lower it or widen the selection.")
    else:
        cards = []
        for _, r in bt.head(3).iterrows():
            parts = r["bowler"].split()
            ini = (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else r["bowler"][:2].upper()
            cards.append(dict(name=r["bowler"], shield=shield(ini, "#2b4a9b"), big=f"{r['economy']:.2f}",
                              sub=f"economy · {r['dot_pct']:.0f}% dots · {int(r['wickets'])} wkts", tint="rgba(60,90,180,.30)"))
        st.markdown(podium(cards), unsafe_allow_html=True)
        st.image(render("bowler_scatter", bt))
        with st.expander("Leaderboard table"):
            cols = ["bowler", "balls", "wickets", "economy", "dot_pct", "boundary_pct", "bowling_sr"]
            st.dataframe(bt[cols].round(2), hide_index=True)
    st.caption("Bowler numbers are raw: not adjusted for venue or opposition, and a bowler's team may change over time.")

# ---------------------------------------------------------------------- 6. method
with tabs[5]:
    sec("HOW IT WORKS")
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
