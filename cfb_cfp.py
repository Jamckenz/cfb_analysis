from __future__ import annotations
import argparse

import cfb_data as D
from cfb_analysis import monte_carlo

WIN_TOTAL_TO_CFP_PROB = {
    16: 0.99, 15: 0.99, 14: 0.98, 13: 0.97, 12: 0.95,
    11: 0.82, 10: 0.50, 9: 0.18, 8: 0.05, 7: 0.01,
    6: 0.0, 5: 0.0, 4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0, 0: 0.0,
}


def cfp_lookup(wins: int) -> float:
    if wins in WIN_TOTAL_TO_CFP_PROB:
        return WIN_TOTAL_TO_CFP_PROB[wins]
    if wins > max(WIN_TOTAL_TO_CFP_PROB):
        return WIN_TOTAL_TO_CFP_PROB[max(WIN_TOTAL_TO_CFP_PROB)]
    return 0.0


def cfp_probabilities(trials: int = 10_000, seed: int = 42) -> dict:
    results = monte_carlo(trials=trials, seed=seed)
    out = {}
    for team in D.FOCUS_TEAMS:
        dist = results[team]["distribution"]
        prob = sum(p * cfp_lookup(int(round(w))) for w, p in dist.items())
        out[team] = {
            "cfp_probability": prob,
            "mean_wins": results[team]["mean_wins"],
            "distribution": dist,
        }
    return out


def print_cfp_probabilities(trials: int = 10_000):
    probs = cfp_probabilities(trials=trials)
    print("=" * 78)
    print("12-TEAM PLAYOFF PROBABILITY (simplified estimate)")
    print("=" * 78)
    print("Not a full bracket simulation -- this maps each team's simulated win total\n"
          "to a rough historical rate of making the field at that record, based on how\n"
          "the 12-team Playoff has selected teams so far (conference champs + at-large\n"
          "by committee ranking). It doesn't model specific opponents, conference title\n"
          "games, or what other teams around the bubble do. Treat it as directional.\n")
    for team in D.FOCUS_TEAMS:
        p = probs[team]
        print(f"  {team:<14} {p['cfp_probability']*100:5.1f}%  "
              f"(mean projected record: {p['mean_wins']:.1f} wins)")

    print()
    print("Win-total breakdown (probability of finishing with exactly N wins x "
          "P(CFP | N wins)):")
    for team in D.FOCUS_TEAMS:
        dist = probs[team]["distribution"]
        top_wins = sorted(dist.keys(), key=lambda w: -dist[w])[:4]
        parts = [f"{w}W: {dist[w]*100:.0f}% (CFP odds at {w}W: {cfp_lookup(w)*100:.0f}%)"
                 for w in sorted(top_wins)]
        print(f"  {team}: " + ", ".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10_000)
    args = parser.parse_args()
    print_cfp_probabilities(trials=args.trials)
