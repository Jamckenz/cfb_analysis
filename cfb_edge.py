from __future__ import annotations
import csv
import os

import cfb_data as D
from cfb_model import (
    load_schedule, load_results, game_predictions, win_probability, MODEL_SCALE,
    prob_to_american_odds,
)
from config import EDGE_THRESHOLD, ENSEMBLE_MODEL_WEIGHT

VEGAS_CSV = os.path.join(os.path.dirname(__file__), "vegas_lines.csv")


def load_vegas_lines() -> dict[tuple[str, int], dict]:
    if not os.path.exists(VEGAS_CSV):
        with open(VEGAS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["team", "week", "vegas_spread", "vegas_moneyline"])
        return {}
    lines = {}
    with open(VEGAS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("vegas_spread") in (None, ""):
                continue
            lines[(row["team"], int(row["week"]))] = {
                "spread": float(row["vegas_spread"]),
                "moneyline": float(row["vegas_moneyline"]) if row.get("vegas_moneyline") else None,
            }
    return lines


def spread_to_win_prob(spread: float) -> float:
    pred_margin = -spread
    score_diff = pred_margin * 3
    return win_probability(score_diff, 0, "Neutral")


def moneyline_to_win_prob(ml: float) -> float:
    if ml < 0:
        return -ml / (-ml + 100)
    return 100 / (ml + 100)


def ensemble_probability(model_prob: float, market_prob: float,
                          model_weight: float = ENSEMBLE_MODEL_WEIGHT) -> float:
    return model_weight * model_prob + (1 - model_weight) * market_prob


def find_edges():
    games = load_schedule()
    load_results(games)
    vegas = load_vegas_lines()
    preds = game_predictions(games)

    print("=" * 100)
    print("MODEL vs. MARKET vs. ENSEMBLE")
    print("=" * 100)
    if not vegas:
        print(f"No lines logged yet. Add rows to {VEGAS_CSV}:")
        print("  team,week,vegas_spread,vegas_moneyline")
        print("  Ole Miss,1,-3.5,-175")
        return

    print(f"Ensemble = {ENSEMBLE_MODEL_WEIGHT*100:.0f}% model + {(1-ENSEMBLE_MODEL_WEIGHT)*100:.0f}% market "
          f"(edit ENSEMBLE_MODEL_WEIGHT in config.py to change the blend)\n")

    found_any = False
    for r in preds:
        key = (r["team"], r["week"])
        if key not in vegas or r["played"]:
            continue
        v = vegas[key]
        market_prob = spread_to_win_prob(v["spread"])
        model_prob = r["win_prob"]
        ensemble_prob = ensemble_probability(model_prob, market_prob)
        ensemble_ml = prob_to_american_odds(ensemble_prob)
        edge = model_prob - market_prob
        flag = " <-- EDGE" if abs(edge) >= EDGE_THRESHOLD else ""
        if flag:
            found_any = True
        ml_str = f"+{ensemble_ml}" if ensemble_ml is not None and ensemble_ml > 0 else str(ensemble_ml)
        print(f"{r['team']:<12} Wk{r['week']:<3} vs {r['opponent']:<18} "
              f"model {model_prob*100:5.1f}%  market {market_prob*100:5.1f}%  "
              f"ensemble {ensemble_prob*100:5.1f}% (ML {ml_str:>6})  "
              f"edge {edge*100:+5.1f}pp{flag}")

    if not found_any:
        print(f"\nNo games clear the {EDGE_THRESHOLD*100:.0f}pp edge threshold this week.")


if __name__ == "__main__":
    find_edges()
