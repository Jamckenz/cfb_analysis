from __future__ import annotations
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import cfb_data as D
from cfb_model import (
    Game, load_schedule, load_results, preseason_adjusted_score,
    preseason_win_prob_for_game, IN_SEASON_K, game_predictions, season_projection,
    season_to_date_stats,
)
from cfb_analysis import monte_carlo, pca_clustering, project_onto_pca

TEAM_COLORS = D.TEAM_COLORS
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def live_score_trajectory(games: list[Game], team: str) -> list[tuple[int, float]]:
    team_games = sorted([g for g in games if g.team == team], key=lambda g: g.week)
    preseason = preseason_adjusted_score(team)
    traj = [(0, preseason)]
    cum_delta = 0.0
    for g in team_games:
        if g.played:
            cum_delta += IN_SEASON_K * (g.actual_result - preseason_win_prob_for_game(g))
            traj.append((g.week, preseason + cum_delta))
    return traj


def plot_live_scores(games: list[Game]):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for team in D.FOCUS_TEAMS:
        traj = live_score_trajectory(games, team)
        weeks, scores = zip(*traj)
        ax.plot(weeks, scores, marker="o", label=team, color=TEAM_COLORS[team], linewidth=2)
        ax.annotate(f"{scores[-1]:.1f}", (weeks[-1], scores[-1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9, color=TEAM_COLORS[team])
    ax.set_xlabel("Week (0 = preseason)")
    ax.set_ylabel("Live Adjusted Score")
    ax.set_title("Live Power Rating Through the Season")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def snapshot_games(games: list[Game], through_week: int) -> list[Game]:
    snap = []
    for g in games:
        g2 = Game(g.team, g.week, g.date, g.opponent_display, g.opponent_lookup, g.site)
        if g.week <= through_week and g.played:
            g2.team_score, g2.opp_score = g.team_score, g.opp_score
        snap.append(g2)
    return snap


def projection_trajectory(games: list[Game]) -> dict[str, list[tuple[int, float]]]:
    weeks = sorted({g.week for g in games})
    out = {team: [] for team in D.FOCUS_TEAMS}
    for wk in [0] + weeks:
        snap = snapshot_games(games, wk)
        for row in season_projection(snap):
            out[row["team"]].append((wk, row["expected_wins"]))
    return out


def plot_projection_evolution(games: list[Game]):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    traj = projection_trajectory(games)
    for team, points in traj.items():
        weeks, wins = zip(*points)
        ax.plot(weeks, wins, marker="o", label=team, color=TEAM_COLORS[team], linewidth=2)
    ax.set_xlabel("Week")
    ax.set_ylabel("Projected Final Wins")
    ax.set_title("Season Projection — How It's Evolved")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_schedule_win_probabilities(games: list[Game]):
    preds = game_predictions(games)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, team in zip(axes.flat, D.FOCUS_TEAMS):
        rows = [r for r in preds if r["team"] == team]
        weeks = [r["week"] for r in rows]
        probs = [r["win_prob"] * 100 for r in rows]
        colors = ["#2E75B6" if not r["played"] else "#BFBFBF" for r in rows]
        bars = ax.bar(weeks, probs, color=colors)
        for r, b in zip(rows, bars):
            if r["played"]:
                b.set_color("#548235" if r["hit"] else "#C00000")
        ax.axhline(50, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(team, color=TEAM_COLORS[team], fontweight="bold")
        ax.set_xlabel("Week")
        ax.set_ylim(0, 100)
        for r, wk in zip(rows, weeks):
            ax.annotate(r["opponent"], (wk, 2), rotation=90, fontsize=6.5, ha="center", va="bottom")
    axes[0, 0].set_ylabel("Win Probability (%)")
    axes[1, 0].set_ylabel("Win Probability (%)")
    fig.suptitle("Win Probability by Game  (blue=upcoming, green=correct call, red=upset)", y=1.0)
    fig.tight_layout()
    return fig


def plot_monte_carlo(games: list[Game], trials: int = 10_000, seed: int = 42):
    results = monte_carlo(trials=trials, seed=seed)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, team in zip(axes.flat, D.FOCUS_TEAMS):
        dist = results[team]["distribution"]
        wins = sorted(dist.keys())
        probs = [dist[w] * 100 for w in wins]
        ax.bar(wins, probs, color=TEAM_COLORS[team])
        ax.axvline(results[team]["mean_wins"], color="black", linestyle="--", linewidth=1,
                   label=f"mean {results[team]['mean_wins']:.1f}")
        ax.set_title(team, color=TEAM_COLORS[team], fontweight="bold")
        ax.set_xlabel("Season Wins")
        ax.set_ylabel("Probability (%)")
        ax.legend(fontsize=8)
    fig.suptitle(f"Monte Carlo Season Simulation ({trials:,} trials)", y=1.0)
    fig.tight_layout()
    return fig


def plot_pca_clusters():
    out = pca_clustering()
    res = out["results"]
    cmap = plt.get_cmap("RdYlBu_r")
    k = out["k"]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    for c in range(1, k + 1):
        members = [t for t in res if res[t]["cluster"] == c]
        xs = [res[t]["pc1"] for t in members]
        ys = [res[t]["pc2"] for t in members]
        ax.scatter(xs, ys, s=60, color=cmap((c - 1) / max(k - 1, 1)), label=f"Cluster {c}", alpha=0.85)

    label_offsets = {
        "Ole Miss": (10, -14), "Notre Dame": (10, 10),
        "Indiana": (10, 10), "Illinois": (10, -14),
    }
    for team in D.FOCUS_TEAMS:
        t = res[team]
        color = TEAM_COLORS[team]
        ax.scatter([t["pc1"]], [t["pc2"]], s=280, facecolors="none", edgecolors=color,
                   linewidths=2.5, zorder=5)
        ax.annotate(team, (t["pc1"], t["pc2"]), textcoords="offset points",
                    xytext=label_offsets[team], fontsize=11, fontweight="bold", color=color)

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("PC1 — overall statistical dominance")
    ax.set_ylabel("PC2 — defense/SOS-leaning (+) vs. offense-volume-leaning (−)")
    ax.set_title("Power 4 + Notre Dame — 2025 Performance Clusters (tracked teams circled)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_pca_progression(games: list[Game]):
    out = pca_clustering()
    res = out["results"]
    scaler, pca = out["scaler"], out["pca"]
    cmap = plt.get_cmap("RdYlBu_r")
    k = out["k"]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    for c in range(1, k + 1):
        members = [t for t in res if res[t]["cluster"] == c]
        xs = [res[t]["pc1"] for t in members]
        ys = [res[t]["pc2"] for t in members]
        ax.scatter(xs, ys, s=45, color=cmap((c - 1) / max(k - 1, 1)), alpha=0.35, zorder=1)

    any_progression = False
    for team in D.FOCUS_TEAMS:
        color = TEAM_COLORS[team]
        weeks_played = sorted({g.week for g in games if g.team == team and g.played})
        traj = []
        for wk in weeks_played:
            stats = season_to_date_stats(games, team, through_week=wk)
            if stats is None or stats["off_ypg"] is None or stats["def_ypg"] is None:
                continue
            pc1, pc2, _ = project_onto_pca(stats["off_ppg"], stats["off_ypg"],
                                            stats["def_ppg"], stats["def_ypg"],
                                            stats["sos"], scaler, pca)
            traj.append((wk, pc1, pc2))

        preseason_pc1, preseason_pc2 = res[team]["pc1"], res[team]["pc2"]
        if traj:
            any_progression = True
            xs = [preseason_pc1] + [p[1] for p in traj]
            ys = [preseason_pc2] + [p[2] for p in traj]
            ax.plot(xs, ys, color=color, linewidth=2, zorder=4, alpha=0.8)
            ax.scatter(xs[:-1], ys[:-1], color=color, s=50, zorder=4, alpha=0.5)
            ax.scatter([xs[-1]], [ys[-1]], color=color, s=220, zorder=6,
                       edgecolors="black", linewidths=1.5)
            ax.annotate(f"{team} (Wk {traj[-1][0]})", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(10, 8), fontsize=10, fontweight="bold", color=color)
        else:
            ax.scatter([preseason_pc1], [preseason_pc2], s=220, facecolors="none",
                       edgecolors=color, linewidths=2, zorder=5)
            ax.annotate(f"{team} (2025)", (preseason_pc1, preseason_pc2), textcoords="offset points",
                        xytext=(10, 8), fontsize=10, fontweight="bold", color=color, alpha=0.6)

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("PC1 — overall statistical dominance")
    ax.set_ylabel("PC2 — defense/SOS-leaning (+) vs. offense-volume-leaning (−)")
    title = "2026 In-Season Movement Through the 2025 Power 4 Landscape"
    if not any_progression:
        title += " (no games with yardage logged yet — showing 2025 positions)"
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def save_all_plots(show: bool = False) -> list[str]:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    games = load_schedule()
    load_results(games)

    figs = {
        "live_scores.png": plot_live_scores(games),
        "projection_evolution.png": plot_projection_evolution(games),
        "schedule_win_probabilities.png": plot_schedule_win_probabilities(games),
        "monte_carlo.png": plot_monte_carlo(games),
        "pca_clusters.png": plot_pca_clusters(),
        "pca_progression.png": plot_pca_progression(games),
    }
    saved = []
    for name, fig in figs.items():
        path = os.path.join(PLOTS_DIR, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
        saved.append(path)

    if show:
        plt.show()
    return saved


if __name__ == "__main__":
    save_all_plots(show="--show" in sys.argv)

