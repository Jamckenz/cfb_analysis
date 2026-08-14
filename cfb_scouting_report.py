from __future__ import annotations
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import cfb_data as D
from cfb_model import (
    Game, load_schedule, load_results, live_adjusted_scores, preseason_adjusted_score,
    season_projection, preseason_win_prob_for_game, data_power_score, HOME_FIELD_ADV,
)
from cfb_analysis import pca_clustering, project_onto_pca
from cfb_report import current_week

TEAM_COLORS = D.TEAM_COLORS
TEAM_ACCENT = D.TEAM_ACCENT


def compute_baseline_and_weekly(games: list[Game], team: str, through_week: int) -> tuple[dict | None, list[dict]]:
    team_games = sorted([g for g in games if g.team == team and g.played and g.week <= through_week],
                         key=lambda g: g.week)
    if not team_games:
        return None, []

    baseline_off = sum(g.team_score for g in team_games) / len(team_games)
    baseline_def = sum(g.opp_score for g in team_games) / len(team_games)
    yardage_games = [g for g in team_games if g.team_yards is not None and g.opp_yards is not None]
    baseline_off_yds = (sum(g.team_yards for g in yardage_games) / len(yardage_games)
                         if yardage_games else None)
    baseline_def_yds = (sum(g.opp_yards for g in yardage_games) / len(yardage_games)
                         if yardage_games else None)

    rows = []
    for g in team_games:
        off_dev = ((g.team_score - baseline_off) / baseline_off * 100) if baseline_off else 0.0
        def_dev = ((g.opp_score - baseline_def) / baseline_def * 100) if baseline_def else 0.0
        rows.append({
            "week": g.week, "opponent": g.opponent_display, "result": "W" if g.team_score > g.opp_score
            else ("L" if g.team_score < g.opp_score else "T"),
            "off_pts": g.team_score, "def_pts": g.opp_score,
            "off_dev_pct": off_dev, "def_dev_pct": def_dev,
        })

    baseline = {
        "off_ppg": baseline_off, "def_ppg": baseline_def,
        "off_ypg": baseline_off_yds, "def_ypg": baseline_def_yds,
        "games": len(team_games), "record": f"{sum(1 for r in rows if r['result']=='W')}-"
                                             f"{sum(1 for r in rows if r['result']=='L')}",
    }
    return baseline, rows


def team_logo_path(team: str) -> str | None:
    safe_name = team.replace(" ", "_").lower()
    for ext in ("png", "PNG", "jpg", "jpeg"):
        p = os.path.join(os.path.dirname(__file__), "logos", f"{safe_name}.{ext}")
        if os.path.exists(p):
            return p
    return None


def draw_team_badge(fig, team: str, color: str, rect: list[float]):
    logo_path = team_logo_path(team)
    badge_ax = fig.add_axes(rect)
    badge_ax.axis("off")
    if logo_path:
        img = plt.imread(logo_path)
        badge_ax.imshow(img)
        badge_ax.set_aspect("equal")
    else:
        badge_ax.set_xlim(0, 1)
        badge_ax.set_ylim(0, 1)
        badge_ax.set_aspect("equal")
        badge_ax.add_patch(patches.Circle((0.5, 0.5), 0.46, facecolor="white",
                                           edgecolor=color, linewidth=2.5, zorder=1))
        initials = "".join(w[0] for w in team.split())[:3].upper()
        badge_ax.text(0.5, 0.5, initials, ha="center", va="center", fontsize=13,
                       fontweight="bold", color=color, zorder=2)
    return badge_ax


def build_team_report(team: str, games: list[Game], week: int | None = None):
    if week is None:
        week = current_week(games)
    color = TEAM_COLORS[team]
    baseline, weekly_rows = compute_baseline_and_weekly(games, team, week)

    live_scores = live_adjusted_scores(games)
    preseason_score = preseason_adjusted_score(team)
    live_score = live_scores[team]
    proj = next(r for r in season_projection(games) if r["team"] == team)

    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(nrows=100, ncols=100)

    header_ax = fig.add_axes([0, 0.90, 1, 0.10])
    header_ax.set_facecolor(color)
    header_ax.axis("off")
    header_ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=header_ax.transAxes,
                                           facecolor=color, zorder=0))
    header_ax.text(0.14, 0.62, team.upper(), color="white", fontsize=26, fontweight="bold",
                    transform=header_ax.transAxes, va="center")
    header_ax.text(0.14, 0.20, f"SCOUTING REPORT — WEEK {week}", color="white", fontsize=12,
                    transform=header_ax.transAxes, va="center", alpha=0.9, family="monospace")
    header_ax.text(0.96, 0.5, f"{proj['actual_record']}", color="white", fontsize=22,
                    fontweight="bold", transform=header_ax.transAxes, va="center", ha="right")
    header_ax.text(0.96, 0.14, "RECORD", color="white", fontsize=9,
                    transform=header_ax.transAxes, va="center", ha="right", alpha=0.85,
                    family="monospace")
    header_ax.text(0.5, 0.06, "PAGE 1 OF 4", color="white", fontsize=8,
                    transform=header_ax.transAxes, va="center", ha="center", alpha=0.7,
                    family="monospace")

    draw_team_badge(fig, team, color, [0.035, 0.915, 0.075, 0.062])

    baseline_ax = fig.add_axes([0.04, 0.68, 0.92, 0.19])
    baseline_ax.axis("off")
    baseline_ax.text(0, 1.05, f"SEASON BASELINE (through Week {week})", fontsize=12,
                      fontweight="bold", color="#333333")

    if baseline is None:
        baseline_ax.text(0.5, 0.5, "No games played yet — baseline will populate once\n"
                                    "results are logged in weekly_results.csv",
                          ha="center", va="center", fontsize=11, color="#888888", style="italic")
    else:
        cards = [
            ("OFF PPG", f"{baseline['off_ppg']:.1f}"),
            ("DEF PPG", f"{baseline['def_ppg']:.1f}"),
            ("OFF YPG", f"{baseline['off_ypg']:.0f}" if baseline["off_ypg"] else "—"),
            ("DEF YPG", f"{baseline['def_ypg']:.0f}" if baseline["def_ypg"] else "—"),
            ("LIVE RATING", f"{live_score:.1f}"),
            ("PROJECTED", proj["projected_record"]),
        ]
        card_w = 1.0 / len(cards)
        for i, (label, value) in enumerate(cards):
            x0 = i * card_w
            baseline_ax.add_patch(patches.FancyBboxPatch(
                (x0 + card_w * 0.06, 0.05), card_w * 0.88, 0.85,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                facecolor="#F2F2F2", edgecolor="#CCCCCC", linewidth=0.8, transform=baseline_ax.transAxes))
            baseline_ax.text(x0 + card_w / 2, 0.58, value, ha="center", va="center",
                              fontsize=15, fontweight="bold", color=color, transform=baseline_ax.transAxes)
            baseline_ax.text(x0 + card_w / 2, 0.20, label, ha="center", va="center",
                              fontsize=7.5, color="#555555", family="monospace", transform=baseline_ax.transAxes)
        baseline_ax.set_xlim(0, 1)
        baseline_ax.set_ylim(0, 1)

    trend_ax = fig.add_axes([0.08, 0.42, 0.84, 0.22])
    trend_ax.set_title("WEEK-OVER-WEEK PERFORMANCE vs. SEASON BASELINE", fontsize=11,
                        fontweight="bold", color="#333333", loc="left")
    if not weekly_rows:
        trend_ax.axis("off")
        trend_ax.text(0.5, 0.5, "No played games to trend yet", ha="center", va="center",
                       fontsize=10, color="#888888", style="italic", transform=trend_ax.transAxes)
    else:
        weeks = [r["week"] for r in weekly_rows]
        off_devs = [r["off_dev_pct"] for r in weekly_rows]
        def_devs = [-r["def_dev_pct"] for r in weekly_rows]
        width = 0.38
        x = np.arange(len(weeks))
        bars1 = trend_ax.bar(x - width / 2, off_devs, width, label="Offense (pts vs. baseline)",
                              color=[("#2E7D32" if v >= 0 else "#C62828") for v in off_devs])
        bars2 = trend_ax.bar(x + width / 2, def_devs, width, label="Defense (fewer pts allowed = better)",
                              color=[("#1565C0" if v >= 0 else "#EF6C00") for v in def_devs])
        trend_ax.axhline(0, color="black", linewidth=0.8)
        trend_ax.set_xticks(x)
        def label_for(opp):
            return opp if opp.lower().startswith("at ") else f"vs {opp}"
        trend_ax.set_xticklabels([f"Wk{w}\n{label_for(r['opponent'])}" for w, r in zip(weeks, weekly_rows)],
                                  fontsize=6.5)
        trend_ax.set_ylabel("% deviation from\nseason baseline", fontsize=8)
        trend_ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        trend_ax.grid(axis="y", alpha=0.25)
        for spine in ["top", "right"]:
            trend_ax.spines[spine].set_visible(False)

    pca_ax = fig.add_axes([0.08, 0.06, 0.84, 0.30])
    pca_out = pca_clustering()
    res = pca_out["results"]
    scaler, pca = pca_out["scaler"], pca_out["pca"]
    k = pca_out["k"]
    cmap = plt.get_cmap("Greys")
    for c in range(1, k + 1):
        members = [t for t in res if res[t]["cluster"] == c]
        xs = [res[t]["pc1"] for t in members]
        ys = [res[t]["pc2"] for t in members]
        pca_ax.scatter(xs, ys, s=25, color=cmap(0.35 + 0.1 * c), alpha=0.5, zorder=1)

    pca_ax.scatter([res[team]["pc1"]], [res[team]["pc2"]], s=180, facecolors="none",
                    edgecolors=color, linewidths=2, zorder=4)
    pca_ax.annotate(f"{team} (2025)", (res[team]["pc1"], res[team]["pc2"]), textcoords="offset points",
                     xytext=(8, 8), fontsize=8, color=color, alpha=0.6, fontweight="bold")

    traj_x, traj_y = [], []
    for wk in weeks_played_list(games, team) if weekly_rows else []:
        from cfb_model import season_to_date_stats
        stats = season_to_date_stats(games, team, through_week=wk)
        if stats is None or stats["off_ypg"] is None or stats["def_ypg"] is None:
            continue
        pc1, pc2, _ = project_onto_pca(stats["off_ppg"], stats["off_ypg"], stats["def_ppg"],
                                        stats["def_ypg"], stats["sos"], scaler, pca)
        traj_x.append(pc1)
        traj_y.append(pc2)
    if traj_x:
        pca_ax.plot([res[team]["pc1"]] + traj_x, [res[team]["pc2"]] + traj_y, color=color,
                     linewidth=2, zorder=3, alpha=0.85)
        pca_ax.scatter([traj_x[-1]], [traj_y[-1]], s=260, color=color, zorder=5,
                        edgecolors="black", linewidths=1.5)
        pca_ax.annotate(f"Week {week}", (traj_x[-1], traj_y[-1]), textcoords="offset points",
                         xytext=(18, 18), fontsize=9, fontweight="bold", color=color,
                         arrowprops=dict(arrowstyle="-", color=color, alpha=0.6, linewidth=1))

    pca_ax.axhline(0, color="grey", linewidth=0.5)
    pca_ax.axvline(0, color="grey", linewidth=0.5)
    pca_ax.set_xlabel("PC1 — overall statistical dominance", fontsize=8)
    pca_ax.set_ylabel("PC2 — defense/SOS-lean vs. offense-volume-lean", fontsize=8)
    pca_ax.set_title("PERFORMANCE CLUSTER POSITION (vs. Power 4 + Notre Dame, 2025 field)",
                      fontsize=11, fontweight="bold", color="#333333", loc="left")
    pca_ax.tick_params(labelsize=7)
    pca_ax.grid(alpha=0.15)

    fig.text(0.5, 0.015, "Generated by the CFB 2026 Tracker — cfb_scouting_report.py", ha="center",
              fontsize=7, color="#AAAAAA")

    return fig


def weeks_played_list(games: list[Game], team: str) -> list[int]:
    return sorted({g.week for g in games if g.team == team and g.played})


def build_methodology_page(team: str):
    color = TEAM_COLORS[team]
    fig = plt.figure(figsize=(8.5, 11))

    header_ax = fig.add_axes([0, 0.94, 1, 0.06])
    header_ax.axis("off")
    header_ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=header_ax.transAxes,
                                           facecolor=color, zorder=0))
    header_ax.text(0.10, 0.5, "HOW THIS REPORT IS BUILT", color="white", fontsize=15,
                    fontweight="bold", transform=header_ax.transAxes, va="center")
    header_ax.text(0.96, 0.5, "PAGE 2 OF 4", color="white", fontsize=9,
                    transform=header_ax.transAxes, va="center", ha="right", family="monospace")
    draw_team_badge(fig, team, color, [0.025, 0.948, 0.045, 0.037])

    body_ax = fig.add_axes([0.06, 0.03, 0.88, 0.89])
    body_ax.axis("off")
    body_ax.set_xlim(0, 1)
    body_ax.set_ylim(0, 1)

    y = 1.0

    def heading(text, size=11.5):
        nonlocal y
        body_ax.text(0, y, text, fontsize=size, fontweight="bold", color=color, va="top")
        y -= 0.028

    def para(text, size=8.7, gap=0.022, color_="#222222"):
        nonlocal y
        import textwrap
        wrapped = textwrap.wrap(text, width=98)
        for line in wrapped:
            body_ax.text(0, y, line, fontsize=size, color=color_, va="top", family="sans-serif")
            y -= gap
        y -= (gap * 0.6)

    def bullet(label, text, size=8.7):
        nonlocal y
        import textwrap
        wrapped = textwrap.wrap(text, width=90)
        body_ax.text(0.015, y, "•", fontsize=size, color=color, va="top", fontweight="bold")
        body_ax.text(0.035, y, label, fontsize=size, color="#111111", va="top", fontweight="bold")
        y -= 0.022
        for line in wrapped:
            body_ax.text(0.035, y, line, fontsize=size, color="#333333", va="top")
            y -= 0.020
        y -= 0.010

    heading("This is not a black box")
    para("Nothing in this report comes from a trained machine-learning model, a neural network, or "
         "anything you have to \"trust\" without being able to check it. Every number below is a "
         "formula you could reproduce by hand with a calculator — real 2025 game stats, run through "
         "arithmetic anyone can inspect. If a number looks wrong, you can trace exactly where it came "
         "from and why.")

    heading("How a team's Power Score is built")
    bullet("1. Start with real performance.",
           "Each team's 2025 scoring margin (points scored minus points allowed, per game) and "
           "yardage efficiency, blended together. Final score alone is noisy — garbage time, a lucky "
           "bounce — so it's balanced against how many yards a team actually gained and allowed.")
    bullet("2. Adjust for schedule strength.",
           "A big margin against weak opponents counts for less than the same margin against a tough "
           "schedule. This adjustment is based on how good a team's actual opponents were that season.")
    bullet("3. Adjust for returning production.",
           "How much of a team's key personnel is coming back in 2026, weighted by position — a "
           "returning quarterback matters far more than a returning backup. This nudges the score up "
           "for continuity, down for heavy turnover.")

    heading("How a Win Probability is calculated")
    para("Take the gap between two teams' Power Scores (plus a small bonus for whoever's playing at "
         "home). Feed that gap into a standard statistical formula that converts \"how big is the "
         "gap\" into \"what percent chance does each team have to win.\" This is the same category of "
         "formula real sportsbooks use to set their own lines — it isn't unique or mysterious math.")

    heading("What the cluster chart (bottom of page 1) actually shows")
    para("That scatter plot uses a technique called PCA (Principal Component Analysis) — which sounds "
         "complicated but does something simple: it takes several different stats (scoring, yardage, "
         "schedule strength, offense, defense) and boils them down into just two numbers per team, so "
         "every team can be plotted as a single dot. Teams that performed similarly end up near each "
         "other; teams that performed very differently end up far apart. The horizontal axis (PC1) is "
         "roughly \"how statistically dominant was this team, overall.\" The vertical axis (PC2) "
         "separates teams that leaned on defense and a tougher schedule (up) from teams that put up "
         "big offensive numbers against a lighter schedule (down). This team's colored line traces "
         "where they've actually moved during the current season.")

    heading("How we know this isn't just guessing")
    para("The exact same method used in this report was tested against 10,370 real games from the "
         "2021-2024 seasons — using only the prior year's data to predict each season, the same way "
         "this report uses 2025 data to inform 2026. The result: a Brier score of 0.19 (0 is a perfect "
         "prediction, 0.25 is a coin flip — professional sportsbook lines typically score in the same "
         "0.18-0.19 range). When this method says a team has about a 70% chance to win, that team "
         "actually wins right around 70% of the time. That's a checkable, falsifiable claim, not a "
         "sales pitch.")

    fig.text(0.5, 0.012, "Generated by the CFB 2026 Tracker — cfb_scouting_report.py", ha="center",
              fontsize=7, color="#AAAAAA")
    return fig


def build_definitions_page(team: str):
    color = TEAM_COLORS[team]
    fig = plt.figure(figsize=(8.5, 11))

    header_ax = fig.add_axes([0, 0.94, 1, 0.06])
    header_ax.axis("off")
    header_ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=header_ax.transAxes,
                                           facecolor=color, zorder=0))
    header_ax.text(0.10, 0.5, "PLAIN-LANGUAGE DEFINITIONS", color="white", fontsize=15,
                    fontweight="bold", transform=header_ax.transAxes, va="center")
    header_ax.text(0.96, 0.5, "PAGE 3 OF 4", color="white", fontsize=9,
                    transform=header_ax.transAxes, va="center", ha="right", family="monospace")
    draw_team_badge(fig, team, color, [0.025, 0.948, 0.045, 0.037])

    body_ax = fig.add_axes([0.06, 0.03, 0.88, 0.89])
    body_ax.axis("off")
    body_ax.set_xlim(0, 1)
    body_ax.set_ylim(0, 1)

    y = 0.97

    def bullet(label, text, size=8.8):
        nonlocal y
        import textwrap
        wrapped = textwrap.wrap(text, width=92)
        body_ax.text(0.015, y, "•", fontsize=size, color=color, va="top", fontweight="bold")
        body_ax.text(0.035, y, label, fontsize=size, color="#111111", va="top", fontweight="bold")
        y -= 0.026
        for line in wrapped:
            body_ax.text(0.035, y, line, fontsize=size, color="#333333", va="top")
            y -= 0.0225
        y -= 0.016

    definitions = [
        ("Power Score / Live Rating", "A single number representing team strength. Only meaningful "
         "relative to another team's number — like a credit score, not a percentage. \"Live\" means "
         "it updates as actual 2026 results come in, on top of the preseason number."),
        ("SOS (Strength of Schedule)", "How good a team's opponents actually were, based on those "
         "opponents' real win-loss records — not a guess or a reputation, a calculation."),
        ("EPA / PPA (points added per play)", "A measure of how much a single play actually helped a "
         "team, accounting for down, distance, and field position — better than raw yards because a "
         "3rd-and-1 conversion and a garbage-time checkdown don't count the same, even though both "
         "might gain 4 yards."),
        ("Success Rate", "The percentage of plays that meaningfully gained the yardage needed for that "
         "down. A cleaner measure of week-to-week consistency than a single big play inflating a "
         "team's average."),
        ("Brier Score", "The standard scientific measure of how well-calibrated a prediction is. Lower "
         "is better: 0 means perfect predictions, 0.25 is what you'd get by always guessing a coin "
         "flip. This report's method scores 0.19 across four real seasons."),
        ("Continuity Index", "A measure of how much proven, returning talent a team has coming back, "
         "weighted by position — quarterback continuity counts far more than backup running back "
         "continuity, because that's what the data shows actually predicts next season."),
        ("Cluster", "A group of teams whose full statistical profiles are similar to each other, found "
         "mathematically by the PCA method described on the previous page — not a subjective tier "
         "list anyone typed in by hand."),
        ("Vig / Posted Odds", "The bookmaker's built-in markup on a betting line. This report shows both "
         "the model's raw, \"fair\" probability and what a sportsbook-style posted line would look "
         "like after that markup is applied."),
    ]
    for label, text in definitions:
        bullet(label + ":", text)

    fig.text(0.5, 0.012, "Generated by the CFB 2026 Tracker — cfb_scouting_report.py", ha="center",
              fontsize=7, color="#AAAAAA")
    return fig


def compute_weekly_accuracy(games: list[Game], team: str, through_week: int) -> list[dict]:
    team_games = sorted([g for g in games if g.team == team and g.played and g.week <= through_week],
                         key=lambda g: g.week)
    rows = []
    for g in team_games:
        pred_prob = preseason_win_prob_for_game(g)
        team_preseason_score = preseason_adjusted_score(g.team)
        opp_score = data_power_score(g.opponent_lookup)
        hfa = {"Home": HOME_FIELD_ADV, "Away": -HOME_FIELD_ADV, "Neutral": 0.0}[g.site]
        pred_margin = ((team_preseason_score + hfa) - opp_score) / 3.0
        actual_margin = g.team_score - g.opp_score
        hit = (g.actual_result > 0.5) == (pred_prob > 0.5)
        rows.append({
            "week": g.week, "opponent": g.opponent_display,
            "pred_prob": pred_prob, "actual_result": g.actual_result, "hit": hit,
            "pred_margin": pred_margin, "actual_margin": actual_margin,
            "margin_error": abs(actual_margin - pred_margin),
            "brier_term": (pred_prob - g.actual_result) ** 2,
        })
    return rows


def build_model_performance_page(team: str, games: list[Game], week: int):
    color = TEAM_COLORS[team]
    rows = compute_weekly_accuracy(games, team, week)

    fig = plt.figure(figsize=(8.5, 11))

    header_ax = fig.add_axes([0, 0.94, 1, 0.06])
    header_ax.axis("off")
    header_ax.add_patch(patches.Rectangle((0, 0), 1, 1, transform=header_ax.transAxes,
                                           facecolor=color, zorder=0))
    header_ax.text(0.10, 0.5, "MODEL PERFORMANCE TRACKER", color="white", fontsize=15,
                    fontweight="bold", transform=header_ax.transAxes, va="center")
    header_ax.text(0.96, 0.5, "PAGE 4 OF 4", color="white", fontsize=9,
                    transform=header_ax.transAxes, va="center", ha="right", family="monospace")
    draw_team_badge(fig, team, color, [0.025, 0.948, 0.045, 0.037])

    body_ax = fig.add_axes([0.06, 0.06, 0.88, 0.86])
    body_ax.axis("off")
    body_ax.set_xlim(0, 1)
    body_ax.set_ylim(0, 1)

    y = 0.98
    body_ax.text(0, y, f"How well has this model actually called {team}'s games this season?",
                 fontsize=11.5, fontweight="bold", color=color, va="top")
    y -= 0.035

    if not rows:
        body_ax.text(0, y, "No games played yet — this page fills in once results are logged in "
                            "weekly_results.csv.", fontsize=9.5, color="#888888", style="italic", va="top")
        fig.text(0.5, 0.02, "Generated by the CFB 2026 Tracker — cfb_scouting_report.py", ha="center",
                  fontsize=7, color="#AAAAAA")
        return fig

    body_ax.text(0, y, "Every prediction below was locked in BEFORE that week's game was played — "
                        "using the model's preseason rating, not anything adjusted after the fact. "
                        "This is the model grading its own homework, game by game.",
                 fontsize=8.7, color="#333333", va="top", wrap=True)
    y -= 0.055

    col_x = [0.0, 0.09, 0.40, 0.55, 0.65, 0.76, 0.90]
    headers = ["Wk", "Opponent", "Predicted", "Actual", "Result", "Pred Mgn", "Mgn Error"]
    for cx, h in zip(col_x, headers):
        body_ax.text(cx, y, h, fontsize=8, fontweight="bold", color="#555555", va="top")
    y -= 0.008
    body_ax.plot([0, 1], [y, y], color="#CCCCCC", linewidth=0.8, transform=body_ax.transAxes)
    y -= 0.026

    for r in rows:
        hit_color = "#2E7D32" if r["hit"] else "#C62828"
        hit_label = "HIT" if r["hit"] else "MISS"
        body_ax.text(col_x[0], y, f"{r['week']}", fontsize=8.3, color="#222222", va="top")
        body_ax.text(col_x[1], y, r["opponent"][:22], fontsize=8.3, color="#222222", va="top")
        body_ax.text(col_x[2], y, f"{r['pred_prob']*100:.0f}%", fontsize=8.3, color="#222222", va="top")
        body_ax.text(col_x[3], y, "W" if r["actual_result"] > 0.5 else "L",
                      fontsize=8.3, color="#222222", va="top")
        body_ax.text(col_x[4], y, hit_label, fontsize=8.3, color=hit_color, fontweight="bold", va="top")
        body_ax.text(col_x[5], y, f"{r['pred_margin']:+.1f}", fontsize=8.3, color="#222222", va="top")
        body_ax.text(col_x[6], y, f"{r['margin_error']:.1f}", fontsize=8.3, color="#222222", va="top")
        y -= 0.026

    y -= 0.02
    body_ax.plot([0, 1], [y, y], color="#CCCCCC", linewidth=0.8, transform=body_ax.transAxes)
    y -= 0.05

    n = len(rows)
    accuracy = sum(1 for r in rows if r["hit"]) / n
    brier = sum(r["brier_term"] for r in rows) / n
    avg_margin_err = sum(r["margin_error"] for r in rows) / n

    body_ax.text(0, y, "SEASON-TO-DATE SUMMARY", fontsize=11, fontweight="bold", color=color, va="top")
    y -= 0.04

    cards = [
        ("ACCURACY", f"{accuracy*100:.0f}%", f"{sum(1 for r in rows if r['hit'])}/{n} correct calls"),
        ("BRIER SCORE", f"{brier:.3f}", "0 = perfect, 0.25 = coin flip"),
        ("AVG MARGIN ERROR", f"{avg_margin_err:.1f} pts", "predicted vs. actual point margin"),
    ]
    card_w = 1.0 / len(cards)
    for i, (label, value, sub) in enumerate(cards):
        x0 = i * card_w
        body_ax.add_patch(patches.FancyBboxPatch(
            (x0 + card_w * 0.05, y - 0.16), card_w * 0.90, 0.15,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor="#F2F2F2", edgecolor="#CCCCCC", linewidth=0.8, transform=body_ax.transAxes))
        body_ax.text(x0 + card_w / 2, y - 0.055, value, ha="center", va="center",
                      fontsize=16, fontweight="bold", color=color, transform=body_ax.transAxes)
        body_ax.text(x0 + card_w / 2, y - 0.10, label, ha="center", va="center",
                      fontsize=7, color="#555555", family="monospace", transform=body_ax.transAxes)
        body_ax.text(x0 + card_w / 2, y - 0.135, sub, ha="center", va="center",
                      fontsize=6, color="#888888", transform=body_ax.transAxes)
    y -= 0.22

    body_ax.text(0, y, "FOR CONTEXT", fontsize=10, fontweight="bold", color=color, va="top")
    y -= 0.03
    import textwrap
    context_text = (f"The model's own multi-season validation (2021-2024, 10,370 real games across "
                     f"every FBS team, not just {team}) scored a Brier score of 0.19. {team}'s "
                     f"{n}-game sample this season is small enough that some swing above or below that "
                     f"number is normal and expected — a handful of games can't prove or disprove the "
                     f"model the way thousands can. Treat this page as a spot-check on this specific "
                     f"team, not a substitute for that larger validation.")
    for line in textwrap.wrap(context_text, width=98):
        body_ax.text(0, y, line, fontsize=8.5, color="#333333", va="top")
        y -= 0.024

    fig.text(0.5, 0.02, "Generated by the CFB 2026 Tracker — cfb_scouting_report.py", ha="center",
              fontsize=7, color="#AAAAAA")
    return fig


def save_scouting_reports(week: int | None = None, teams: list[str] | None = None) -> list[str]:
    from matplotlib.backends.backend_pdf import PdfPages

    games = load_schedule()
    load_results(games)
    if week is None:
        week = current_week(games)
    teams = teams or D.FOCUS_TEAMS

    saved = []
    for team in teams:
        page1 = build_team_report(team, games, week=week)
        page2 = build_methodology_page(team)
        page3 = build_definitions_page(team)
        page4 = build_model_performance_page(team, games, week)
        safe_name = team.replace(" ", "_")
        path = os.path.join(os.path.dirname(__file__), f"Scouting_Report_{safe_name}_Week{week}.pdf")
        with PdfPages(path) as pdf:
            pdf.savefig(page1)
            pdf.savefig(page2)
            pdf.savefig(page3)
            pdf.savefig(page4)
        plt.close(page1)
        plt.close(page2)
        plt.close(page3)
        plt.close(page4)
        print(f"Saved {path}")
        saved.append(path)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--team", type=str, default=None,
                         help="Just one team (e.g. 'Illinois'); default is all 4 tracked teams")
    args = parser.parse_args()
    save_scouting_reports(week=args.week, teams=[args.team] if args.team else None)
