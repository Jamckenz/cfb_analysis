from __future__ import annotations
import argparse
import random

import cfb_data as D
from cfb_model import load_schedule, load_results, game_predictions, season_projection
from cfb_analysis import pca_clustering
from cfb_report import current_week

ND_SOS_2025 = 0.559
CLUSTER1_TEAMS = ["Oregon", "Indiana", "Ohio State", "Miami", "Texas Tech"]
ND_NEAREST_NEIGHBORS = ["Utah", "Florida State", "Ole Miss", "Texas A&M"]

JOKE_POOL = [
    lambda: (f"Notre Dame's 2025 strength of schedule was .559 — worse than every Cluster 1 team "
              f"({', '.join(CLUSTER1_TEAMS)}), whose average sits in the .600s. That's not an opinion, "
              f"that's the same math that just told you who's favored this week."),
    lambda: (f"The PCA model — an algorithm that has never seen a Notre Dame highlight reel and does not "
              f"know what a subway alumnus is — still wouldn't put them in the elite cluster. Make of that "
              f"what you will."),
    lambda: (f"Notre Dame's closest statistical comp in 2025 was Ole Miss. A team from Oxford, Mississippi. "
              f"If your program's nearest neighbor is a 3-loss SEC team, maybe don't build a schedule soft "
              f"enough to hide it."),
    lambda: (f"Reminder: our own backtest — using nothing but 2024 trailing performance, before either team "
              f"had played a 2025 down — had Notre Dame ranked #1 of these four teams. They finished 3rd and "
              f"missed the Playoff. Indiana, ranked #2 by that same model, won the national title. The model "
              f"was more generous to Notre Dame than reality was."),
    lambda: (f"Notre Dame plays as an Independent, which is Latin for 'doesn't have a conference schedule to "
              f"hide behind, but built one just as soft anyway.'"),
]


def biggest_mover(games) -> str | None:
    from cfb_model import live_adjusted_scores, preseason_adjusted_score
    live = live_adjusted_scores(games)
    deltas = {t: live[t] - preseason_adjusted_score(t) for t in D.FOCUS_TEAMS}
    if all(abs(d) < 0.05 for d in deltas.values()):
        return None
    team = max(deltas, key=lambda t: abs(deltas[t]))
    direction = "up" if deltas[team] > 0 else "down"
    return f"{team}'s live rating moved the most this week, {direction} {abs(deltas[team]):.1f} points."


def generate_recap(week: int | None = None) -> str:
    games = load_schedule()
    load_results(games)
    if week is None:
        week = current_week(games)

    preds = game_predictions(games)
    this_week = [r for r in preds if r["week"] == week and r["played"]]
    proj = season_projection(games)

    lines = [f"FOOTBALL UPDATE — WEEK {week}", "=" * 40, ""]

    if this_week:
        lines.append("This week's results:")
        for r in this_week:
            result = "W" if r["actual_result"] > 0.5 else ("L" if r["actual_result"] < 0.5 else "T")
            hit_note = "(model called it)" if r["hit"] else "(model missed this one)"
            lines.append(f"  {r['team']} {result} vs {r['opponent']} ({r['actual_score']}) {hit_note}")
        lines.append("")

    mover = biggest_mover(games)
    if mover:
        lines.append(mover)
        lines.append("")

    lines.append("Season projections:")
    for row in proj:
        acc = f"{row['prediction_accuracy']*100:.0f}%" if row["prediction_accuracy"] is not None else "n/a"
        lines.append(f"  {row['team']:<12} {row['actual_record']:>6} so far -> "
                      f"projected {row['projected_record']} (model accuracy so far: {acc})")
    lines.append("")

    rng = random.Random(week)
    joke = rng.choice(JOKE_POOL)()
    lines.append("Notre Dame Corner:")
    lines.append(f"  {joke}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()
    print(generate_recap(week=args.week))
