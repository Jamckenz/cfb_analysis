from __future__ import annotations

SCORING_MARGIN_2024 = {
    "Ole Miss": 24.2, "Notre Dame": 20.6, "Indiana": 25.7, "Illinois": 6.6,
}
SOS_OWP_2024 = {
    "Ole Miss": 0.530, "Notre Dame": 0.650, "Indiana": 0.510, "Illinois": 0.545,
}

ACTUAL_2025 = {
    "Ole Miss":   {"record": "13-2", "win_pct": 13/15, "note": "CFP quarterfinalist (Sugar Bowl W vs Georgia), lost Fiesta Bowl to Miami"},
    "Notre Dame": {"record": "10-2", "win_pct": 10/12, "note": "Missed the CFP; no bowl (withdrew from Pop-Tarts Bowl)"},
    "Indiana":    {"record": "16-0", "win_pct": 16/16, "note": "Undefeated national champion"},
    "Illinois":   {"record": "9-4",  "win_pct": 9/13,  "note": "Music City Bowl champion (beat Tennessee 30-28)"},
}

SOS_SCALE = 40.0


def trailing_power_score(team: str) -> float:
    margin = SCORING_MARGIN_2024[team]
    sos = SOS_OWP_2024[team]
    return margin + (sos - 0.5) * SOS_SCALE


def run_backtest():
    teams = list(SCORING_MARGIN_2024.keys())
    scores = {t: trailing_power_score(t) for t in teams}

    predicted_rank = sorted(teams, key=lambda t: -scores[t])
    actual_rank = sorted(teams, key=lambda t: -ACTUAL_2025[t]["win_pct"])

    print("=" * 78)
    print("BACKTEST: 2024 trailing data -> would it have called 2025 correctly?")
    print("=" * 78)
    print(f"{'Team':<14}{'2024-based Score':>18}{'Predicted Rank':>16}{'Actual 2025':>14}{'Actual Rank':>13}")
    for t in teams:
        print(f"{t:<14}{scores[t]:>18.1f}{predicted_rank.index(t)+1:>16}"
              f"{ACTUAL_2025[t]['record']:>14}{actual_rank.index(t)+1:>13}")

    print("\nPredicted order (best to worst, using only 2024 data):", " > ".join(predicted_rank))
    print("Actual order (best to worst, by 2025 win %):          ", " > ".join(actual_rank))

    hits = sum(1 for t in teams if predicted_rank.index(t) == actual_rank.index(t))
    print(f"\nExact rank matches: {hits}/{len(teams)}")

    print("\nNotes:")
    for t in teams:
        print(f"  {t}: {ACTUAL_2025[t]['note']}")

    biggest_miss = max(teams, key=lambda t: abs(predicted_rank.index(t) - actual_rank.index(t)))
    print(f"\nBiggest miss: {biggest_miss} — predicted rank {predicted_rank.index(biggest_miss)+1}, "
          f"actual rank {actual_rank.index(biggest_miss)+1}. Trailing-year box scores can't see a coaching "
          f"change, a portal haul, or a team simply putting it together — that's exactly why the live "
          f"in-season model (cfb_model.py) re-weights every week instead of trusting a preseason number "
          f"all year.")


if __name__ == "__main__":
    run_backtest()
