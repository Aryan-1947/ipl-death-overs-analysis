"""Matplotlib charts for the death-overs analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

TEAM_COLORS = {
    "Chennai Super Kings": "#F9CD05", "Mumbai Indians": "#004BA0",
    "Royal Challengers Bengaluru": "#D1171B", "Kolkata Knight Riders": "#3A225D",
    "Rajasthan Royals": "#EA1A85", "Delhi Capitals": "#17449B",
    "Punjab Kings": "#DD1F2D", "Sunrisers Hyderabad": "#F7A721",
    "Gujarat Titans": "#1C2C5B", "Lucknow Super Giants": "#00A3E0",
}
GREY = "#8c8c8c"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def _color(team: str) -> str:
    return TEAM_COLORS.get(team, GREY)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def net_rr_bar(eff: pd.DataFrame, out: Path, subtitle: str = "") -> Path:
    d = eff.sort_values("net_rr")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.5))
    ax.barh(d["team"], d["net_rr"], color=[_color(t) for t in d["team"]], edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    for y, v in enumerate(d["net_rr"]):
        ax.text(v + (0.05 if v >= 0 else -0.05), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlabel("Net death-over run rate  (batting RR - bowling economy)")
    ax.set_title(f"Death-overs efficiency by team\n{subtitle}", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    return _save(fig, out)


def bat_vs_bowl_scatter(eff: pd.DataFrame, out: Path, subtitle: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(eff["bowl_rate"], eff["bat_rate"], s=eff["total_balls"] / eff["total_balls"].max() * 600 + 60,
               c=[_color(t) for t in eff["team"]], edgecolor="black", linewidth=0.7, alpha=0.9, zorder=3)
    for _, r in eff.iterrows():
        ax.annotate(r["team"], (r["bowl_rate"], r["bat_rate"]), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.axvline(eff["bowl_rate"].mean(), color=GREY, ls="--", lw=0.8)
    ax.axhline(eff["bat_rate"].mean(), color=GREY, ls="--", lw=0.8)
    ax.margins(x=0.12, y=0.12)
    ax.invert_xaxis()  # better bowling (lower economy) to the right
    ax.set_xlabel("Bowling economy in death overs  (lower = better, axis reversed)")
    ax.set_ylabel("Batting run rate in death overs")
    ax.text(0.98, 0.97, "Elite both ways", transform=ax.transAxes, ha="right", va="top", color="green", fontsize=9)
    ax.text(0.02, 0.03, "Struggles both ways", transform=ax.transAxes, ha="left", va="bottom", color="firebrick", fontsize=9)
    ax.set_title(f"Hitting vs defending the death overs\n{subtitle}", loc="left", fontweight="bold")
    return _save(fig, out)


def season_heatmap(eff_season: pd.DataFrame, out: Path, min_balls: int = 150) -> Path:
    d = eff_season[eff_season["total_balls"] >= min_balls]
    pv = d.pivot(index="team", columns="season", values="net_rr")
    pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]
    m = max(abs(np.nanmin(pv.values)), abs(np.nanmax(pv.values)), 0.5)
    fig, ax = plt.subplots(figsize=(1 + 0.9 * pv.shape[1], 0.55 * pv.shape[0] + 1.5))
    im = ax.imshow(pv.values, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-m, vcenter=0, vmax=m), aspect="auto")
    ax.set_xticks(range(pv.shape[1]), pv.columns.astype(int), rotation=45)
    ax.set_yticks(range(pv.shape[0]), pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Net run rate")
    ax.set_title("Net death-over run rate, team x season", loc="left", fontweight="bold")
    return _save(fig, out)


def over_progression(obo: pd.DataFrame, teams: list[str], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    lg = obo[obo["batting_team"] == "League average"]
    ax.plot(lg["over_no"], lg["run_rate"], color="black", ls="--", lw=2, marker="o", label="League average")
    for t in teams:
        d = obo[obo["batting_team"] == t]
        ax.plot(d["over_no"], d["run_rate"], color=_color(t), lw=2.2, marker="o", label=t)
    ax.set_xticks(sorted(obo["over_no"].unique()))
    ax.set_xlabel("Over")
    ax.set_ylabel("Batting run rate (runs/over)")
    ax.set_title("How teams accelerate through the death", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    return _save(fig, out)


def bowler_scatter(bt: pd.DataFrame, out: Path, top_n: int = 15) -> Path:
    d = bt.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(d["economy"], d["dot_pct"], s=d["wickets"] * 12 + 40, c=d["boundary_pct"],
                    cmap="RdYlGn_r", edgecolor="black", linewidth=0.6, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["bowler"], (r["economy"], r["dot_pct"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    fig.colorbar(sc, label="Boundary % conceded")
    ax.set_xlabel("Economy (death overs)")
    ax.set_ylabel("Dot-ball %")
    ax.set_title(f"Best death bowlers (top {len(d)} by economy)\nbubble size = wickets", loc="left", fontweight="bold")
    return _save(fig, out)


def team_profile(eff_season: pd.DataFrame, obo: pd.DataFrame, team: str, out: Path) -> Path:
    d = eff_season[eff_season["team"] == team].sort_values("season")
    c = _color(team)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    ax = axs[0, 0]
    ax.plot(d["season"], d["bat_rate"], marker="o", color=c, lw=2.2, label="Batting RR")
    ax.plot(d["season"], d["bowl_rate"], marker="s", color="black", lw=1.8, ls="--", label="Bowling economy")
    ax.set_title("Run rate scored vs conceded"); ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.bar(d["season"], d["net_rr"], color=[c if v >= 0 else "#bbbbbb" for v in d["net_rr"]], edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Net death-over run rate")

    ax = axs[1, 0]
    t = obo[obo["batting_team"] == team]
    lg = obo[obo["batting_team"] == "League average"]
    ax.plot(lg["over_no"], lg["run_rate"], color="black", ls="--", marker="o", label="League")
    ax.plot(t["over_no"], t["run_rate"], color=c, lw=2.2, marker="o", label=team)
    ax.set_xticks(sorted(obo["over_no"].unique())); ax.set_title("Batting run rate by over"); ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(d["season"], d["bat_dot_pct"], marker="o", color=c, lw=2.2, label="Dot % faced (batting)")
    ax.plot(d["season"], d["bowl_dot_pct"], marker="s", color="black", ls="--", label="Dot % bowled")
    ax.set_title("Dot-ball control"); ax.legend(fontsize=8)

    for a in axs.flat:
        a.set_xlabel("Season" if a in axs[1] else "")
    fig.suptitle(f"{team}: death-overs profile (overs 17-20)", fontweight="bold", x=0.01, ha="left")
    return _save(fig, out)


def adjusted_vs_raw(adj: pd.DataFrame, out: Path, subtitle: str = "") -> Path:
    """Raw net RR (hollow) vs context-adjusted net (filled, 90% bootstrap interval)."""
    d = adj.sort_values("adj_net").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(d) + 1.8))
    y = np.arange(len(d))
    if d["adj_lo"].notna().all():
        ax.errorbar(d["adj_net"], y, xerr=[d["adj_net"] - d["adj_lo"], d["adj_hi"] - d["adj_net"]],
                    fmt="none", ecolor=GREY, elinewidth=2, capsize=3, zorder=2)
    for i, r in d.iterrows():
        ax.plot([r["raw_net_rr"], r["adj_net"]], [i, i], color="#cccccc", lw=1, zorder=1)
    ax.scatter(d["raw_net_rr"], y, s=70, facecolor="white", edgecolor="black", zorder=3, label="Raw net RR")
    ax.scatter(d["adj_net"], y, s=90, c=[_color(t) for t in d["team"]], edgecolor="black", zorder=4,
               label="Adjusted (venue, innings, opponent, season)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, d["team"])
    ax.set_xlabel("Runs per over vs an average team  (bars = 90% bootstrap interval)")
    ax.set_title(f"Does context change the ranking?\n{subtitle}", loc="left", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    return _save(fig, out)


def chase_vs_defend(split: pd.DataFrame, out: Path) -> Path:
    d = split.sort_values("chasing_rr").reset_index(drop=True)
    y = np.arange(len(d)); h = 0.38
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.5 * len(d) + 2), sharey=True)
    ax = axs[0]
    ax.barh(y + h / 2, d["bat_first_rr"], h, color="#4c78a8", label="Setting a target")
    ax.barh(y - h / 2, d["chasing_rr"], h, color="#f58518", label="Chasing")
    ax.set_title("Batting run rate", loc="left", fontweight="bold"); ax.legend(fontsize=8)
    ax.set_yticks(y, d["team"]); ax.set_xlim(left=min(d[["bat_first_rr", "chasing_rr"]].min()) - 1)
    ax = axs[1]
    ax.barh(y + h / 2, d["bowl_first_econ"], h, color="#4c78a8", label="Bowling first")
    ax.barh(y - h / 2, d["defending_econ"], h, color="#f58518", label="Defending a target")
    ax.set_title("Bowling economy (lower = better)", loc="left", fontweight="bold"); ax.legend(fontsize=8)
    ax.set_xlim(left=min(d[["bowl_first_econ", "defending_econ"]].min()) - 1)
    for a in axs:
        a.grid(axis="y", visible=False); a.set_xlabel("Runs per over")
    fig.suptitle("Death overs: setting vs chasing, bowling first vs defending", x=0.01, ha="left", fontweight="bold")
    return _save(fig, out)


def venue_effects(venues: pd.DataFrame, out: Path, top_n: int = 14) -> Path:
    d = venues[venues["venue"] != "Other venues"]
    d = pd.concat([d.head(top_n // 2), d.tail(top_n // 2)]).drop_duplicates().sort_values("effect")
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(d) + 1.6))
    ax.barh(d["venue"], d["effect"], color=["#2e7d32" if v > 0 else "#c62828" for v in d["effect"]],
            edgecolor="black", linewidth=0.5)
    for yy, (v, n) in enumerate(zip(d["effect"], d["innings"])):
        ax.text(v + (0.03 if v >= 0 else -0.03), yy, f"n={n}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Death-over runs per over vs average venue (after team/innings/season controls)")
    ax.set_title("Which grounds inflate or suppress death-over scoring?", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    return _save(fig, out)




def venue_effects(venues: pd.DataFrame, out: Path, top_n: int | None = None) -> Path:
    d = venues[venues["venue"] != "Other venues"]
    if top_n:
        d = pd.concat([d.head(top_n // 2), d.tail(top_n // 2)]).drop_duplicates()
    d = d.sort_values("effect")
    labels = [f"{v}  (n={n})" for v, n in zip(d["venue"], d["innings"])]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(d) + 1.8))
    ax.barh(labels, d["effect"], color=["#2e7d32" if v > 0 else "#c62828" for v in d["effect"]],
            edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Death-over runs per over vs average venue (after team/innings/season controls)\n"
                  "n = innings played; venues with n < ~40 are noisy")
    ax.set_title("Which grounds inflate or suppress death-over scoring?", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    return _save(fig, out)


def chase_vs_defend(split: pd.DataFrame, out: Path) -> Path:
    d = split.sort_values("chasing_rr").reset_index(drop=True)
    y = np.arange(len(d))
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.5 * len(d) + 2), sharey=True)
    panels = [
        (axs[0], "bat_first_rr", "chasing_rr", "Setting a target", "Chasing", "Batting run rate"),
        (axs[1], "bowl_first_econ", "defending_econ", "Bowling first", "Defending a target",
         "Bowling economy (lower = better)"),
    ]
    for ax, c1, c2, l1, l2, title in panels:
        for i, r in d.iterrows():
            ax.plot([r[c1], r[c2]], [i, i], color="#cccccc", lw=2, zorder=1)
        ax.scatter(d[c1], y, s=70, color="#4c78a8", edgecolor="black", zorder=3, label=l1)
        ax.scatter(d[c2], y, s=70, color="#f58518", edgecolor="black", zorder=3, label=l2)
        ax.axvline(d[c1].mean(), color="#4c78a8", ls="--", lw=1, alpha=0.7)
        ax.axvline(d[c2].mean(), color="#f58518", ls="--", lw=1, alpha=0.7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("Runs per over  (dashed lines = average of the 10 teams)")
    axs[0].set_yticks(y, d["team"])
    for a in axs:
        a.grid(axis="y", visible=False)
    fig.suptitle("Death overs: setting vs chasing, bowling first vs defending", x=0.01, ha="left", fontweight="bold")
    return _save(fig, out)