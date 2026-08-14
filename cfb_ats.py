from __future__ import annotations

import cfb_data as D
from cfb_model import load_schedule, load_results
from cfb_edge import load_vegas_lines


def compute_ats(games) -> dict[str, dict]:
    vegas = load_vegas_lines()
    records = {t: {"wins": 0, "losses": 0, "pushes": 0, "games": []} for t in D.FOCUS_TEAMS}

    for g in games:
        if not g.played:
            continue
        key = (g.team, g.week)
        if key not in vegas:
            continue
        spread = vegas[key]["spread"]
        actual_margin = g.team_score - g.opp_score
        cover_margin = actual_margin + spread

        if abs(cover_margin) < 1e-9:
            result = "Push"
            records[g.team]["pushes"] += 1
        elif cover_margin > 0:
            result = "Cover"
            records[g.team]["wins"] += 1
        else:
            result = "No Cover"
            records[g.team]["losses"] += 1

        records[g.team]["games"].append({
            "week": g.week, "opponent": g.opponent_display,
            "actual_score": f"{g.team_score}-{g.opp_score}",
            "spread": spread, "result": result,
        })

    return records


def print_ats_report():
    games = load_schedule()
    load_results(games)
    records = compute_ats(games)

    print("=" * 80)
    print("AGAINST THE SPREAD (ATS) RECORD")
    print("=" * 80)

    any_data = False
    for team in D.FOCUS_TEAMS:
        r = records[team]
        total = r["wins"] + r["losses"] + r["pushes"]
        if total == 0:
            continue
        any_data = True
        print(f"\n{team}: {r['wins']}-{r['losses']}-{r['pushes']} ATS")
        for g in r["games"]:
            print(f"  Wk{g['week']:<3} vs {g['opponent']:<18} {g['actual_score']:<8} "
                  f"(line {g['spread']:+.1f})  {g['result']}")

    if not any_data:
        print("\nNo games with both a logged Vegas line and a played result yet.")
        print("Log lines in vegas_lines.csv and results in weekly_results.csv, then rerun.")


if __name__ == "__main__":
    print_ats_report()
