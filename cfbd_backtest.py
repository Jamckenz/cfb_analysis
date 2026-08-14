from __future__ import annotations
import argparse
import os
import sys
import json
import math

import requests
import numpy as np

from cfb_model import (
    SCORING_WEIGHT, EFFICIENCY_WEIGHT, SOS_SCALE, PPA_PLAYS_PER_GAME,
    MODEL_SCALE, HOME_FIELD_ADV, win_probability,
)
from cfbd_fetch import get_api_key, fetch_advanced_stats, API_BASE

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cfbd_backtest_cache")


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def fetch_games(year: int, api_key: str) -> list[dict]:
    url = f"{API_BASE}/games"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers,
                         params={"year": year, "seasonType": "regular"}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def cached_fetch(year: int, api_key: str) -> tuple[list[dict], list[dict]]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    games_path = os.path.join(CACHE_DIR, f"games_{year}.json")
    adv_path = os.path.join(CACHE_DIR, f"advanced_{year}.json")

    if os.path.exists(games_path):
        with open(games_path) as f:
            games = json.load(f)
    else:
        print(f"  Fetching games for {year}...")
        games = fetch_games(year, api_key)
        with open(games_path, "w") as f:
            json.dump(games, f)

    if os.path.exists(adv_path):
        with open(adv_path) as f:
            advanced = json.load(f)
    else:
        print(f"  Fetching advanced stats for {year}...")
        advanced = fetch_advanced_stats(year, api_key)
        with open(adv_path, "w") as f:
            json.dump(advanced, f)

    return games, advanced


def compute_scoring_margin_and_sos(games: list[dict]) -> tuple[dict, dict]:
    team_games = {}
    for g in games:
        if not _first(g, "completed"):
            continue
        home = _first(g, "homeTeam", "home_team")
        away = _first(g, "awayTeam", "away_team")
        hp = _first(g, "homePoints", "home_points")
        ap = _first(g, "awayPoints", "away_points")
        if home is None or away is None or hp is None or ap is None:
            continue
        team_games.setdefault(home, []).append((away, hp, ap))
        team_games.setdefault(away, []).append((home, ap, hp))

    scoring_margin = {}
    win_pct = {}
    for team, glist in team_games.items():
        margins = [pf - pa for _, pf, pa in glist]
        wins = sum(1 for _, pf, pa in glist if pf > pa)
        scoring_margin[team] = sum(margins) / len(margins)
        win_pct[team] = wins / len(glist)

    sos = {}
    for team, glist in team_games.items():
        opp_win_pcts = [win_pct.get(opp, 0.5) for opp, _, _ in glist]
        sos[team] = sum(opp_win_pcts) / len(opp_win_pcts) if opp_win_pcts else 0.5

    return scoring_margin, sos


def compute_prior_year_scores(scoring_margin: dict, sos: dict, advanced: list[dict]) -> dict:
    ppa_by_team = {}
    for row in advanced:
        team = row.get("team")
        offense = row.get("offense", {}) or {}
        defense = row.get("defense", {}) or {}
        off_ppa = offense.get("ppa")
        def_ppa = defense.get("ppa")
        if team and off_ppa is not None and def_ppa is not None:
            ppa_by_team[team] = (off_ppa, def_ppa)

    scores = {}
    for team, margin in scoring_margin.items():
        s = sos.get(team, 0.5)
        if team in ppa_by_team:
            off_ppa, def_ppa = ppa_by_team[team]
            eff_margin = (off_ppa - def_ppa) * PPA_PLAYS_PER_GAME
            blended = SCORING_WEIGHT * margin + EFFICIENCY_WEIGHT * eff_margin
        else:
            blended = margin
        scores[team] = blended + (s - 0.5) * SOS_SCALE
    return scores


def compute_score_diffs(games: list[dict], prior_scores: dict,
                         home_field_adv: float = HOME_FIELD_ADV) -> list[dict]:
    diffs = []
    for g in games:
        if not _first(g, "completed"):
            continue
        home = _first(g, "homeTeam", "home_team")
        away = _first(g, "awayTeam", "away_team")
        hp = _first(g, "homePoints", "home_points")
        ap = _first(g, "awayPoints", "away_points")
        neutral = bool(_first(g, "neutralSite", "neutral_site")) or False
        if home not in prior_scores or away not in prior_scores:
            continue
        if hp is None or ap is None or hp == ap:
            continue

        hfa = 0.0 if neutral else home_field_adv
        diff = (prior_scores[home] + hfa) - prior_scores[away]
        diffs.append({
            "home": home, "away": away, "diff": diff,
            "actual_home_win": 1.0 if hp > ap else 0.0,
        })
    return diffs


def prob_from_diff(diff: float, scale: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / scale))


def calibration_from_diffs(diffs: list[dict], scale: float) -> tuple[float, float]:
    probs = np.array([prob_from_diff(d["diff"], scale) for d in diffs])
    actuals = np.array([d["actual_home_win"] for d in diffs])
    brier = float(np.mean((probs - actuals) ** 2))
    eps = 1e-9
    log_loss = float(-np.mean(actuals * np.log(probs + eps) + (1 - actuals) * np.log(1 - probs + eps)))
    return brier, log_loss


def predict_and_score_games(games: list[dict], prior_scores: dict) -> list[dict]:
    diffs = compute_score_diffs(games, prior_scores)
    predictions = []
    for d in diffs:
        predictions.append({
            "home": d["home"], "away": d["away"],
            "pred_prob_home": prob_from_diff(d["diff"], MODEL_SCALE),
            "actual_home_win": d["actual_home_win"],
        })
    return predictions


def calibration_report(predictions: list[dict], n_bins: int = 10):
    probs = np.array([p["pred_prob_home"] for p in predictions])
    actuals = np.array([p["actual_home_win"] for p in predictions])

    brier = float(np.mean((probs - actuals) ** 2))
    eps = 1e-9
    log_loss = float(-np.mean(actuals * np.log(probs + eps) + (1 - actuals) * np.log(1 - probs + eps)))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_stats = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        n = int(mask.sum())
        if n == 0:
            bin_stats.append((lo, hi, n, None, None))
            continue
        avg_pred = float(probs[mask].mean())
        avg_actual = float(actuals[mask].mean())
        bin_stats.append((lo, hi, n, avg_pred, avg_actual))

    return brier, log_loss, bin_stats


def plot_calibration(bin_stats, out_path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    xs = [b[3] for b in bin_stats if b[3] is not None]
    ys = [b[4] for b in bin_stats if b[3] is not None]
    sizes = [max(20, b[2] / 5) for b in bin_stats if b[3] is not None]
    ax.scatter(xs, ys, s=sizes, color="#C00000", zorder=5, label="Model bins (size = sample count)")
    ax.plot(xs, ys, color="#C00000", alpha=0.5)
    ax.set_xlabel("Model's predicted home win probability")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Calibration: predicted vs. actual, across all backtested games")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def sweep_model_scale(all_diffs: list[dict], scales: list[float]) -> list[tuple[float, float, float]]:
    results = []
    for scale in scales:
        brier, log_loss = calibration_from_diffs(all_diffs, scale)
        results.append((scale, brier, log_loss))
    return results


def run(start_year: int, end_year: int, sweep: bool = False,
        scale_min: float = 8.0, scale_max: float = 60.0, scale_step: float = 2.0):
    api_key = get_api_key()
    all_predictions = []
    all_diffs = []
    yearly_summary = []

    for test_year in range(start_year, end_year + 1):
        prior_year = test_year - 1
        print(f"\n{test_year} (using {prior_year} as prior-season input)...")
        try:
            prior_games, prior_advanced = cached_fetch(prior_year, api_key)
            test_games, _ = cached_fetch(test_year, api_key)
        except requests.HTTPError as e:
            print(f"  Skipping {test_year}: {e}")
            continue

        scoring_margin, sos = compute_scoring_margin_and_sos(prior_games)
        prior_scores = compute_prior_year_scores(scoring_margin, sos, prior_advanced)
        predictions = predict_and_score_games(test_games, prior_scores)
        diffs = compute_score_diffs(test_games, prior_scores)

        if not predictions:
            print(f"  No usable games for {test_year} (missing prior-year data for most teams).")
            continue

        brier, log_loss, _ = calibration_report(predictions)
        print(f"  {len(predictions)} games, Brier score {brier:.4f}, log-loss {log_loss:.4f}")
        yearly_summary.append((test_year, len(predictions), brier, log_loss))
        all_predictions.extend(predictions)
        all_diffs.extend(diffs)

    if not all_predictions:
        print("\nNo predictions generated — check your year range and API key.")
        return

    print("\n" + "=" * 78)
    print(f"OVERALL: {len(all_predictions)} games across {len(yearly_summary)} seasons")
    print("=" * 78)
    brier, log_loss, bin_stats = calibration_report(all_predictions)
    print(f"Brier score: {brier:.4f}  (0 = perfect, 0.25 = coin flip, closer to 0 is better)")
    print(f"Log-loss:    {log_loss:.4f}  (lower is better)")
    print(f"[Using cfb_model.py's current MODEL_SCALE = {MODEL_SCALE}]")

    print(f"\n{'Predicted range':<20}{'N':>8}{'Avg predicted':>16}{'Avg actual':>14}")
    for lo, hi, n, avg_pred, avg_actual in bin_stats:
        if n == 0:
            print(f"{lo:.1f}-{hi:.1f}{'':<12}{n:>8}{'':>16}{'':>14}")
        else:
            print(f"{lo:.1f}-{hi:.1f}{'':<12}{n:>8}{avg_pred*100:>15.1f}%{avg_actual*100:>13.1f}%")

    print(f"\n{'Year':<8}{'Games':>8}{'Brier':>10}{'Log-loss':>12}")
    for year, n, b, ll in yearly_summary:
        print(f"{year:<8}{n:>8}{b:>10.4f}{ll:>12.4f}")

    plot_path = os.path.join(os.path.dirname(__file__), "calibration.png")
    plot_calibration(bin_stats, plot_path)

    if sweep:
        print("\n" + "=" * 78)
        print(f"MODEL_SCALE SWEEP  ({len(all_diffs)} games, reusing cached data — no new API calls)")
        print("=" * 78)
        scales = list(np.arange(scale_min, scale_max + scale_step, scale_step))
        results = sweep_model_scale(all_diffs, scales)

        print(f"{'Scale':>8}{'Brier':>12}{'Log-loss':>12}")
        for scale, b, ll in results:
            marker = ""
            print(f"{scale:>8.1f}{b:>12.4f}{ll:>12.4f}{marker}")

        best_brier = min(results, key=lambda r: r[1])
        best_logloss = min(results, key=lambda r: r[2])
        print(f"\nBest by Brier score:  MODEL_SCALE = {best_brier[0]:.1f}  (Brier {best_brier[1]:.4f}, "
              f"vs. {brier:.4f} at current MODEL_SCALE={MODEL_SCALE})")
        print(f"Best by log-loss:     MODEL_SCALE = {best_logloss[0]:.1f}  (log-loss {best_logloss[2]:.4f}, "
              f"vs. {log_loss:.4f} at current MODEL_SCALE={MODEL_SCALE})")
        print(f"\nTo apply: change MODEL_SCALE = {MODEL_SCALE} to MODEL_SCALE = {best_brier[0]:.1f} "
              f"near the top of cfb_model.py")

        sweep_plot_path = os.path.join(os.path.dirname(__file__), "model_scale_sweep.png")
        plot_scale_sweep(results, MODEL_SCALE, sweep_plot_path)


def plot_scale_sweep(results: list[tuple[float, float, float]], current_scale: float, out_path: str):
    import matplotlib.pyplot as plt
    scales = [r[0] for r in results]
    briers = [r[1] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(scales, briers, color="#C00000", linewidth=2, marker="o", markersize=4)
    ax.axvline(current_scale, color="grey", linestyle="--", label=f"Current MODEL_SCALE ({current_scale})")
    best = min(results, key=lambda r: r[1])
    ax.scatter([best[0]], [best[1]], color="black", s=100, zorder=5,
               label=f"Best: {best[0]:.1f} (Brier {best[1]:.4f})")
    ax.set_xlabel("MODEL_SCALE")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Finding the empirically optimal MODEL_SCALE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2016,
                         help="First season to TEST (uses start_year-1 as its prior-season input)")
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--sweep-scale", action="store_true",
                         help="Also search MODEL_SCALE values to find the one that minimizes Brier score")
    parser.add_argument("--scale-min", type=float, default=8.0)
    parser.add_argument("--scale-max", type=float, default=60.0)
    parser.add_argument("--scale-step", type=float, default=2.0)
    args = parser.parse_args()
    run(args.start_year, args.end_year, sweep=args.sweep_scale,
        scale_min=args.scale_min, scale_max=args.scale_max, scale_step=args.scale_step)
