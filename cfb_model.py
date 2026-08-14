from __future__ import annotations
import csv
import json
import math
import os
from dataclasses import dataclass, field

import cfb_data as D
from config import (
    SOS_SCALE, HOME_FIELD_ADV, MODEL_SCALE, CONTINUITY_BASELINE, CONTINUITY_WEIGHT,
    IN_SEASON_K, SCORING_WEIGHT, EFFICIENCY_WEIGHT, YARDS_PER_POINT, VIG, SPREAD_JUICE,
    OFFENSE_WEIGHTS, DEFENSE_WEIGHTS, CFBD_CACHE_YEAR, PPA_PLAYS_PER_GAME, GAME_MARGIN_STD,
)

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "weekly_results.csv")

_cfbd_cache: dict | None = None


def _load_cfbd_cache() -> dict:
    global _cfbd_cache
    if _cfbd_cache is not None:
        return _cfbd_cache
    path = os.path.join(os.path.dirname(__file__), f"cfbd_efficiency_{CFBD_CACHE_YEAR}.json")
    if os.path.exists(path):
        with open(path) as f:
            _cfbd_cache = json.load(f)
    else:
        _cfbd_cache = {}
    return _cfbd_cache


def blended_margin(team_lookup_name: str) -> float:
    key = D.NAME_ALIAS.get(team_lookup_name, team_lookup_name)
    margin = D.SCORING_MARGIN.get(key)
    if margin is None:
        raise KeyError(f"No 2025 scoring margin for '{team_lookup_name}' (alias tried '{key}').")

    cfbd = _load_cfbd_cache()
    cfbd_entry = cfbd.get(team_lookup_name) or cfbd.get(key)
    if cfbd_entry and cfbd_entry.get("off_ppa") is not None and cfbd_entry.get("def_ppa") is not None:
        ppa_margin_pts = (cfbd_entry["off_ppa"] - cfbd_entry["def_ppa"]) * PPA_PLAYS_PER_GAME
        return SCORING_WEIGHT * margin + EFFICIENCY_WEIGHT * ppa_margin_pts

    eff = D.EFFICIENCY_STATS.get(team_lookup_name) or D.EFFICIENCY_STATS.get(key)
    if eff is None:
        return margin
    off_ppg, off_ypg, def_ppg, def_ypg = eff
    yardage_margin_pts = (off_ypg - def_ypg) / YARDS_PER_POINT
    return SCORING_WEIGHT * margin + EFFICIENCY_WEIGHT * yardage_margin_pts


def data_power_score(team_lookup_name: str) -> float:
    if team_lookup_name in D.PLACEHOLDER_TEAMS:
        margin, sos = D.PLACEHOLDER_TEAMS[team_lookup_name]
        return margin + (sos - 0.5) * SOS_SCALE
    key = D.NAME_ALIAS.get(team_lookup_name, team_lookup_name)
    sos = D.SOS_OWP.get(key)
    if sos is None:
        raise KeyError(f"No 2025 SOS data for '{team_lookup_name}' (alias tried '{key}').")
    return blended_margin(team_lookup_name) + (sos - 0.5) * SOS_SCALE


def continuity_index(team: str) -> tuple[float, float, float]:
    snaps = D.SNAP_RETENTION[team]
    off_idx = sum(OFFENSE_WEIGHTS[p] * snaps[p] for p in OFFENSE_WEIGHTS)
    def_idx = sum(DEFENSE_WEIGHTS[p] * snaps[p] for p in DEFENSE_WEIGHTS)
    overall = (off_idx + def_idx) / 2
    return off_idx, def_idx, overall


def preseason_adjusted_score(team: str) -> float:
    base = data_power_score(team)
    _, _, overall_idx = continuity_index(team)
    continuity_adj = (overall_idx - CONTINUITY_BASELINE) * CONTINUITY_WEIGHT
    return base + continuity_adj


def win_probability(team_score: float, opp_score: float, site: str) -> float:
    hfa = {"Home": HOME_FIELD_ADV, "Away": -HOME_FIELD_ADV, "Neutral": 0.0}[site]
    score_diff = (team_score + hfa) - opp_score
    return win_probability_from_diff(score_diff)


def win_probability_from_diff(score_diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-score_diff / MODEL_SCALE))


_KNOWN_2025_GAMES_PLAYED = {
    "Ole Miss": 15, "Notre Dame": 12, "Indiana": 16, "Illinois": 13,
}
DEFAULT_GAMES_PLAYED = 12


def games_played_2025(team_lookup_name: str) -> int:
    key = D.NAME_ALIAS.get(team_lookup_name, team_lookup_name)
    return _KNOWN_2025_GAMES_PLAYED.get(team_lookup_name,
           _KNOWN_2025_GAMES_PLAYED.get(key, DEFAULT_GAMES_PLAYED))


def score_standard_error(team_lookup_name: str) -> float:
    n = games_played_2025(team_lookup_name)
    return GAME_MARGIN_STD / (n ** 0.5)


def score_confidence_interval(team: str, z: float = 1.645) -> tuple[float, float, float]:
    point = preseason_adjusted_score(team)
    se = score_standard_error(team)
    return point - z * se, point, point + z * se


def win_probability_band(team_score: float, team_lookup: str, opp_score: float,
                          opp_lookup: str, site: str, z: float = 1.645) -> tuple[float, float, float]:
    hfa = {"Home": HOME_FIELD_ADV, "Away": -HOME_FIELD_ADV, "Neutral": 0.0}[site]
    diff = (team_score + hfa) - opp_score
    combined_se = (score_standard_error(team_lookup) ** 2 + score_standard_error(opp_lookup) ** 2) ** 0.5
    p_low = win_probability_from_diff(diff - z * combined_se)
    p_point = win_probability_from_diff(diff)
    p_high = win_probability_from_diff(diff + z * combined_se)
    return min(p_low, p_high), p_point, max(p_low, p_high)


def prob_to_american_odds(p: float) -> int | None:
    if p is None or p <= 0.0 or p >= 1.0:
        return None
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def apply_vig(p_team: float, vig: float = VIG) -> tuple[float, float]:
    p_opp = 1 - p_team
    return min(p_team * (1 + vig), 0.99), min(p_opp * (1 + vig), 0.99)


def posted_odds(win_prob: float, pred_margin: float) -> dict:
    p_team_book, p_opp_book = apply_vig(win_prob)
    spread_line = round(pred_margin * 2) / 2
    return {
        "moneyline_team": prob_to_american_odds(p_team_book),
        "moneyline_opp": prob_to_american_odds(p_opp_book),
        "spread_team": f"{-spread_line:+.1f}",
        "spread_opp": f"{spread_line:+.1f}",
        "spread_juice": SPREAD_JUICE,
        "implied_overround": round((p_team_book + p_opp_book - 1) * 100, 1),
    }


@dataclass
class Game:
    team: str
    week: int
    date: str
    opponent_display: str
    opponent_lookup: str
    site: str
    team_score: int | None = None
    opp_score: int | None = None
    team_yards: int | None = None
    opp_yards: int | None = None

    @property
    def played(self) -> bool:
        return self.team_score is not None and self.opp_score is not None

    @property
    def actual_result(self) -> float | None:
        if not self.played:
            return None
        if self.team_score > self.opp_score:
            return 1.0
        if self.team_score == self.opp_score:
            return 0.5
        return 0.0


def load_schedule() -> list[Game]:
    return [Game(*row) for row in D.SCHEDULE]


def load_results(games: list[Game]) -> None:
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["team", "week", "team_score", "opp_score", "team_yards", "opp_yards"])
        return
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        entries = {(row["team"], int(row["week"])): row for row in reader
                   if row["team_score"] not in (None, "") and row["opp_score"] not in (None, "")}
    for g in games:
        key = (g.team, g.week)
        if key in entries:
            row = entries[key]
            g.team_score = int(float(row["team_score"]))
            g.opp_score = int(float(row["opp_score"]))
            ty = row.get("team_yards")
            oy = row.get("opp_yards")
            g.team_yards = int(float(ty)) if ty not in (None, "") else None
            g.opp_yards = int(float(oy)) if oy not in (None, "") else None


def preseason_win_prob_for_game(g: Game) -> float:
    team_score = preseason_adjusted_score(g.team)
    opp_score = data_power_score(g.opponent_lookup)
    return win_probability(team_score, opp_score, g.site)


def season_to_date_stats(games: list[Game], team: str, through_week: int | None = None) -> dict | None:
    team_games = [g for g in games if g.team == team and g.played
                  and (through_week is None or g.week <= through_week)]
    if not team_games:
        return None

    off_pts = [g.team_score for g in team_games]
    def_pts = [g.opp_score for g in team_games]
    yardage_games = [g for g in team_games if g.team_yards is not None and g.opp_yards is not None]

    opp_sos = []
    for g in team_games:
        key = D.NAME_ALIAS.get(g.opponent_lookup, g.opponent_lookup)
        s = D.SOS_OWP.get(key)
        if s is not None:
            opp_sos.append(s)

    return {
        "games": len(team_games),
        "off_ppg": sum(off_pts) / len(off_pts),
        "def_ppg": sum(def_pts) / len(def_pts),
        "off_ypg": (sum(g.team_yards for g in yardage_games) / len(yardage_games)
                    if yardage_games else None),
        "def_ypg": (sum(g.opp_yards for g in yardage_games) / len(yardage_games)
                    if yardage_games else None),
        "sos": sum(opp_sos) / len(opp_sos) if opp_sos else 0.55,
    }


def live_adjusted_scores(games: list[Game]) -> dict[str, float]:
    live = {}
    for team in D.FOCUS_TEAMS:
        team_games = [g for g in games if g.team == team]
        delta = 0.0
        for g in team_games:
            if g.played:
                delta += IN_SEASON_K * (g.actual_result - preseason_win_prob_for_game(g))
        live[team] = preseason_adjusted_score(team) + delta
    return live


def game_predictions(games: list[Game]) -> list[dict]:
    live_scores = live_adjusted_scores(games)
    rows = []
    for g in games:
        team_score = live_scores[g.team]
        opp_score = data_power_score(g.opponent_lookup)
        win_prob = win_probability(team_score, opp_score, g.site)
        win_prob_low, _, win_prob_high = win_probability_band(
            team_score, g.team, opp_score, g.opponent_lookup, g.site)
        hfa = {"Home": HOME_FIELD_ADV, "Away": -HOME_FIELD_ADV, "Neutral": 0.0}[g.site]
        pred_margin = ((team_score + hfa) - opp_score) / 3.0
        odds = posted_odds(win_prob, pred_margin)
        row = {
            "team": g.team, "week": g.week, "date": g.date,
            "opponent": g.opponent_display, "site": g.site,
            "live_team_score": round(team_score, 1),
            "opp_power_score": round(opp_score, 1),
            "win_prob": round(win_prob, 4),
            "win_prob_low": round(win_prob_low, 4),
            "win_prob_high": round(win_prob_high, 4),
            "pred_margin": round(pred_margin, 1),
            "played": g.played,
            "moneyline": odds["moneyline_team"],
            "moneyline_opp": odds["moneyline_opp"],
            "spread": odds["spread_team"],
            "spread_juice": odds["spread_juice"],
        }
        if g.played:
            row["actual_score"] = f"{g.team_score}-{g.opp_score}"
            row["actual_result"] = g.actual_result
            preseason_prob = preseason_win_prob_for_game(g)
            row["hit"] = (g.actual_result > 0.5) == (preseason_prob > 0.5)
        rows.append(row)
    return rows


def season_projection(games: list[Game]) -> list[dict]:
    preds = game_predictions(games)
    out = []
    for team in D.FOCUS_TEAMS:
        team_rows = [r for r in preds if r["team"] == team]
        played = [r for r in team_rows if r["played"]]
        unplayed = [r for r in team_rows if not r["played"]]
        actual_wins = sum(r["actual_result"] for r in played)
        proj_wins = actual_wins + sum(r["win_prob"] for r in unplayed)
        variance = sum(r["win_prob"] * (1 - r["win_prob"]) for r in unplayed)
        sd = math.sqrt(variance)
        accuracy = (sum(1 for r in played if r["hit"]) / len(played)) if played else None
        out.append({
            "team": team, "games": len(team_rows), "played": len(played),
            "actual_record": f"{actual_wins:g}-{len(played) - actual_wins:g}",
            "prediction_accuracy": round(accuracy, 3) if accuracy is not None else None,
            "expected_wins": round(proj_wins, 2),
            "std_dev": round(sd, 2),
            "range": f"{max(0, proj_wins - sd):.1f}-{min(len(team_rows), proj_wins + sd):.1f}",
            "projected_record": f"{round(proj_wins)}-{len(team_rows) - round(proj_wins)}",
        })
    return out


def cfbd_data_status() -> dict:
    path = os.path.join(os.path.dirname(__file__), f"cfbd_efficiency_{CFBD_CACHE_YEAR}.json")
    exists = os.path.exists(path)
    n_teams = 0
    if exists:
        cache = _load_cfbd_cache()
        n_teams = sum(1 for v in cache.values() if v.get("off_ppa") is not None)
    return {"year": CFBD_CACHE_YEAR, "path": path, "exists": exists, "teams_with_ppa": n_teams}


def print_report():
    games = load_schedule()
    load_results(games)

    status = cfbd_data_status()
    print("=" * 70)
    if status["exists"] and status["teams_with_ppa"] > 0:
        print(f"DATA MODE: real play-by-play EPA (CFBD, {status['teams_with_ppa']} teams, "
              f"{status['year']}) blended with box-score scoring margin")
    else:
        print(f"DATA MODE: box-score yardage margin (no cfbd_efficiency_{status['year']}.json found "
              f"-- run 'python cfbd_fetch.py --year {status['year']}' to upgrade) blended with scoring margin")

    print("=" * 70)
    print("LIVE ADJUSTED SCORES")
    print("=" * 70)
    for team, score in live_adjusted_scores(games).items():
        low, _, high = score_confidence_interval(team)
        print(f"  {team:<14} {score:6.1f}  (preseason: {preseason_adjusted_score(team):.1f}, "
              f"~90% CI: {low:.1f} to {high:.1f})")

    print()
    print("=" * 70)
    print("SEASON PROJECTION")
    print("=" * 70)
    for row in season_projection(games):
        acc = f"{row['prediction_accuracy']*100:.0f}%" if row['prediction_accuracy'] is not None else "-"
        print(f"  {row['team']:<14} played {row['played']:>2}/{row['games']}  "
              f"actual {row['actual_record']:>6}  accuracy {acc:>4}  "
              f"projected {row['projected_record']:>6}  "
              f"(expected wins {row['expected_wins']}, range {row['range']})")

    print()
    print("=" * 70)
    print("UPCOMING GAMES (next unplayed game per team)")
    print("=" * 70)
    preds = game_predictions(games)
    for team in D.FOCUS_TEAMS:
        next_game = next((r for r in preds if r["team"] == team and not r["played"]), None)
        if next_game:
            ml = next_game["moneyline"]
            ml_str = f"+{ml}" if ml is not None and ml > 0 else str(ml)
            print(f"  {team:<14} Wk{next_game['week']:>2} vs {next_game['opponent']:<20} "
                  f"({next_game['site']:<7})  win prob {next_game['win_prob']*100:5.1f}% "
                  f"({next_game['win_prob_low']*100:.0f}-{next_game['win_prob_high']*100:.0f}%, ~90% CI)  "
                  f"ML {ml_str:>6}  spread {next_game['spread']:>6} ({next_game['spread_juice']})")
        else:
            print(f"  {team:<14} season complete")


if __name__ == "__main__":
    print_report()
