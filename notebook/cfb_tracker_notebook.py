
from __future__ import annotations
import math
import numpy as np


FOCUS_TEAMS = ["Ole Miss", "Notre Dame", "Indiana", "Illinois"]

SCHEDULE = [
    ("Ole Miss", 1, "9/5/26", "Louisville", "Louisville", "Neutral"),
    ("Ole Miss", 2, "9/12/26", "Charlotte", "Charlotte", "Home"),
    ("Ole Miss", 3, "9/19/26", "LSU", "LSU", "Home"),
    ("Ole Miss", 4, "9/26/26", "at Florida", "Florida", "Away"),
    ("Ole Miss", 6, "10/10/26", "at Vanderbilt", "Vanderbilt", "Away"),
    ("Ole Miss", 7, "10/17/26", "Missouri", "Missouri", "Home"),
    ("Ole Miss", 8, "10/24/26", "at Texas", "Texas", "Away"),
    ("Ole Miss", 9, "10/31/26", "Auburn", "Auburn", "Home"),
    ("Ole Miss", 10, "11/7/26", "Georgia", "Georgia", "Home"),
    ("Ole Miss", 11, "11/14/26", "at Oklahoma", "Oklahoma", "Away"),
    ("Ole Miss", 12, "11/21/26", "Wofford", "Wofford (FCS)", "Home"),
    ("Ole Miss", 13, "11/28/26", "Mississippi State", "Mississippi State", "Home"),

    ("Notre Dame", 1, "9/6/26", "Wisconsin", "Wisconsin", "Neutral"),
    ("Notre Dame", 2, "9/12/26", "Rice", "Rice", "Home"),
    ("Notre Dame", 3, "9/19/26", "Michigan State", "Michigan State", "Home"),
    ("Notre Dame", 4, "9/26/26", "at Purdue", "Purdue", "Away"),
    ("Notre Dame", 5, "10/3/26", "at North Carolina", "North Carolina", "Away"),
    ("Notre Dame", 6, "10/10/26", "Stanford", "Stanford", "Home"),
    ("Notre Dame", 7, "10/17/26", "at BYU", "BYU", "Away"),
    ("Notre Dame", 9, "10/31/26", "at Navy", "Navy", "Neutral"),
    ("Notre Dame", 10, "11/7/26", "Miami", "Miami", "Home"),
    ("Notre Dame", 11, "11/14/26", "Boston College", "Boston College", "Home"),
    ("Notre Dame", 12, "11/21/26", "SMU", "SMU", "Home"),
    ("Notre Dame", 13, "11/28/26", "at Syracuse", "Syracuse", "Away"),

    ("Indiana", 1, "9/5/26", "North Texas", "North Texas", "Home"),
    ("Indiana", 2, "9/12/26", "Howard", "Howard (FCS)", "Home"),
    ("Indiana", 3, "9/19/26", "Western Kentucky", "Western Kentucky", "Home"),
    ("Indiana", 4, "9/26/26", "Northwestern", "Northwestern", "Home"),
    ("Indiana", 5, "10/3/26", "at Rutgers", "Rutgers", "Away"),
    ("Indiana", 6, "10/10/26", "at Nebraska", "Nebraska", "Away"),
    ("Indiana", 7, "10/17/26", "Ohio State", "Ohio State", "Home"),
    ("Indiana", 8, "10/24/26", "at Michigan", "Michigan", "Away"),
    ("Indiana", 9, "10/31/26", "Minnesota", "Minnesota", "Home"),
    ("Indiana", 11, "11/14/26", "USC", "USC", "Home"),
    ("Indiana", 12, "11/21/26", "at Washington", "Washington", "Away"),
    ("Indiana", 13, "11/28/26", "Purdue", "Purdue", "Home"),

    ("Illinois", 1, "9/5/26", "UAB", "UAB", "Home"),
    ("Illinois", 2, "9/12/26", "Duke", "Duke", "Home"),
    ("Illinois", 3, "9/19/26", "Southern Illinois", "Southern Illinois (FCS)", "Home"),
    ("Illinois", 4, "9/26/26", "at Ohio State", "Ohio State", "Away"),
    ("Illinois", 5, "10/3/26", "Purdue", "Purdue", "Home"),
    ("Illinois", 6, "10/10/26", "at Michigan State", "Michigan State", "Away"),
    ("Illinois", 8, "10/24/26", "Oregon", "Oregon", "Home"),
    ("Illinois", 9, "10/31/26", "at Maryland", "Maryland", "Away"),
    ("Illinois", 10, "11/7/26", "Nebraska", "Nebraska", "Home"),
    ("Illinois", 11, "11/14/26", "at UCLA", "UCLA", "Away"),
    ("Illinois", 12, "11/21/26", "Iowa", "Iowa", "Home"),
    ("Illinois", 13, "11/28/26", "at Northwestern", "Northwestern", "Away"),
]

SCORING_MARGIN = {
"Indiana":29.9,"Texas Tech":28.1,"Notre Dame":24.4,"Ohio State":24.1,"Utah":22.4,"Oregon":19.1,
"James Madison":18.7,"North Texas":18.6,"Toledo":17.5,"South Florida":17.1,"Miami (FL)":16.1,
"Ole Miss":15.9,"Vanderbilt":15.6,"Washington":15.4,"Georgia":14.5,"Old Dominion":13.5,"Iowa":13.2,
"Texas A&M":12.8,"USC":12.8,"East Carolina":12.6,"BYU":12.3,"Arizona":12.2,"SMU":11.7,"Missouri":11.3,
"Virginia":11.2,"Florida State":11.0,"San Diego State":11.0,"Tennessee":10.9,"Oklahoma":10.8,
"Penn State":10.5,"Alabama":10.3,"Texas":10.2,"Memphis":9.3,"Pittsburgh":8.9,"Louisville":8.8,
"Connecticut":8.3,"Western Michigan":7.5,"Texas State":7.5,"Iowa State":7.2,"Michigan":7.2,
"Georgia Tech":7.2,"Fresno State":7.0,"Clemson":6.7,"UTSA":6.7,"Western Kentucky":6.7,
"Louisiana Tech":6.5,"Navy":6.5,"Houston":6.2,"Auburn":6.1,"UNLV":6.1,"Wake Forest":6.0,"Ohio":5.8,
"Boise State":5.8,"Illinois":5.8,"TCU":5.4,"Duke":5.1,"Hawaii":4.9,"New Mexico":4.8,"Cincinnati":4.7,
"Nebraska":4.1,"Tulane":3.8,"Northwestern":3.5,"North Carolina State":3.1,"LSU":3.0,
"Jacksonville State":2.9,"Kansas State":2.8,"Washington State":2.3,"Army":2.2,"Utah State":2.2,
"Miami (OH)":1.8,"Southern Miss":1.4,"Kansas":1.3,"Arizona State":1.3,"Kennesaw State":1.2,
"UCF":0.8,"South Carolina":0.6,"Buffalo":0.5,"Marshall":0.5,"Mississippi State":0.2,"Minnesota":0.1,
"Troy":-0.4,"Air Force":-0.5,"Liberty":-0.9,"Arkansas":-0.9,"Central Michigan":-1.0,"FIU":-1.5,
"Baylor":-1.5,"Temple":-1.8,"California":-1.8,"Delaware":-2.0,"Arkansas State":-2.2,"Florida":-2.4,
"Maryland":-3.0,"Missouri State":-3.2,"Rutgers":-3.2,"Louisiana":-3.2,"Kentucky":-3.4,
"Bowling Green":-3.8,"South Alabama":-3.9,"Georgia Southern":-4.4,"Wyoming":-4.5,"Akron":-5.2,
"North Carolina":-5.2,"Michigan State":-5.3,"Appalachian State":-5.6,"Eastern Michigan":-5.7,
"Tulsa":-5.8,"New Mexico State":-6.0,"Florida Atlantic":-6.8,"UTEP":-7.1,"Northern Illinois":-7.2,
"Boston College":-7.3,"Middle Tennessee":-8.5,"Virginia Tech":-8.8,"Wisconsin":-8.8,
"West Virginia":-9.1,"Colorado":-9.6,"Nevada":-9.9,"Stanford":-10.3,"Coastal Carolina":-10.9,
"San Jose State":-11.1,"Oregon State":-11.8,"UAB":-11.8,"Colorado State":-12.3,"Kent State":-12.3,
"Purdue":-13.1,"Rice":-13.8,"Ball State":-14.0,"Syracuse":-14.8,"UCLA":-15.2,"ULM":-15.3,
"Georgia State":-18.1,"Oklahoma State":-19.2,"Sam Houston State":-20.1,"Charlotte":-22.0,"UMass":-27.5,
}

SOS_OWP = {
"Oregon":.686,"Wisconsin":.675,"Miami (FL)":.663,"Purdue":.657,"Florida":.653,"UCLA":.653,
"Oklahoma":.650,"LSU":.650,"Indiana":.649,"Ohio State":.641,"Duke":.639,"South Carolina":.634,
"Virginia Tech":.634,"Alabama":.634,"Kentucky":.634,"Colorado":.624,"Georgia":.620,"Penn State":.619,
"Arizona State":.614,"BYU":.611,"West Virginia":.610,"North Carolina State":.609,"Illinois":.605,
"Arkansas":.599,"USC":.597,"South Florida":.596,"Auburn":.596,"TCU":.592,"Mississippi State":.591,
"Ole Miss":.591,"Stanford":.590,"UAB":.590,"Michigan":.588,"Texas Tech":.588,"Washington State":.587,
"Tulane":.587,"Charlotte":.583,"Syracuse":.583,"Boise State":.583,"Army":.582,"Michigan State":.579,
"North Texas":.578,"Rice":.578,"Oregon State":.577,"Louisville":.577,"Iowa":.574,"Iowa State":.573,
"Rutgers":.569,"Oklahoma State":.564,"Kansas":.561,"Kennesaw State":.560,"Notre Dame":.559,
"Virginia":.558,"Colorado State":.556,"Pittsburgh":.556,"Georgia Southern":.556,"Northwestern":.555,
"Arizona":.553,"Temple":.553,"Texas":.553,"UTSA":.552,"Wake Forest":.552,"Florida Atlantic":.550,
"Kansas State":.550,"Georgia Tech":.549,"Cincinnati":.547,"Utah State":.546,"Liberty":.545,
"Washington":.545,"Florida State":.545,"Western Michigan":.542,"Old Dominion":.539,"SMU":.539,
"James Madison":.539,"Georgia State":.538,"Air Force":.535,"Nevada":.535,"Tennessee":.533,
"Baylor":.532,"Minnesota":.523,"Texas A&M":.523,"Miami (OH)":.522,"California":.520,"Nebraska":.520,
"San Diego State":.520,"Vanderbilt":.520,"Southern Miss":.519,"Marshall":.517,"Navy":.517,
"Utah":.517,"East Carolina":.516,"North Carolina":.511,"Houston":.510,"Missouri State":.510,
"Missouri":.510,"Troy":.509,"Maryland":.507,"Coastal Carolina":.507,"FIU":.507,"Boston College":.504,
"Clemson":.500,"Memphis":.500,"UCF":.500,"Louisiana Tech":.497,"San Jose State":.496,
"Sam Houston State":.493,"Louisiana":.487,"Texas State":.484,"Toledo":.480,"Ball State":.479,
"Tulsa":.479,"Delaware":.477,"New Mexico":.473,"South Alabama":.471,"Arkansas State":.471,
"Ohio":.470,"UTEP":.468,"Jacksonville State":.466,"UNLV":.466,"Eastern Michigan":.464,
"Middle Tennessee":.464,"Appalachian State":.464,"ULM":.462,"Bowling Green":.460,
"Western Kentucky":.457,"Wyoming":.457,"Central Michigan":.450,"Hawaii":.440,"UMass":.438,
"New Mexico State":.435,"Kent State":.434,"Connecticut":.430,"Buffalo":.424,"Fresno State":.424,
"Northern Illinois":.421,"Akron":.397,
}

NAME_ALIAS = {
    "Miami": "Miami (FL)", "Miami (Ohio)": "Miami (OH)", "NC State": "North Carolina State",
    "Sam Houston": "Sam Houston State", "UConn": "Connecticut", "Pitt": "Pittsburgh",
}

PLACEHOLDER_TEAMS = {
    "Wofford (FCS)": (-30.0, 0.45),
    "Howard (FCS)": (-30.0, 0.45),
    "Southern Illinois (FCS)": (-30.0, 0.45),
}

RETURNING_STARTERS = {
    "Ole Miss": {"off": 5, "def": 4},
    "Notre Dame": {"off": 7, "def": 8},
    "Indiana": {"off": 4, "def": 6},
    "Illinois": {"off": 3, "def": 4},
}
SNAP_RETENTION = {
    "Ole Miss":   {"QB": 0.84, "RB": 0.79, "WR": 0.22, "TE": 0.24, "OL": 0.56,
                   "DL": 0.73, "LB": 0.21, "DB": 0.34},
    "Notre Dame": {"QB": 0.88, "RB": 0.10, "WR": 0.48, "TE": 0.37, "OL": 0.71,
                   "DL": 0.60, "LB": 0.76, "DB": 0.80},
    "Indiana":    {"QB": 0.01, "RB": 0.16, "WR": 0.21, "TE": 0.005, "OL": 0.60,
                   "DL": 0.35, "LB": 0.70, "DB": 0.47},
    "Illinois":   {"QB": 0.00, "RB": 1.00, "WR": 0.52, "TE": 0.30, "OL": 0.25,
                   "DL": 0.11, "LB": 0.26, "DB": 0.50},
}

PCA_STATS = {
    "USC": ("Big Ten", 35.8, 458.5, 23.0, 335.1),
    "Oregon": ("Big Ten", 36.9, 442.9, 17.9, 260.3),
    "Indiana": ("Big Ten", 41.6, 442.3, 11.7, 247.9),
    "Ohio State": ("Big Ten", 33.4, 414.7, 9.3, 202.7),
    "Washington": ("Big Ten", 34.1, 393.7, 18.7, 299.5),
    "Rutgers": ("Big Ten", 28.7, 389.2, 31.8, 427.3),
    "Michigan": ("Big Ten", 27.5, 387.7, 20.4, 306.2),
    "Maryland": ("Big Ten", 23.8, 387.5, 28.1, 421.9),
    "Northwestern": ("Big Ten", 21.8, 364.0, 20.9, 344.9),
    "Illinois": ("Big Ten", 29.4, 356.4, 23.6, 334.4),
    "Nebraska": ("Big Ten", 28.7, 353.5, 24.6, 318.2),
    "Penn State": ("Big Ten", 31.0, 352.0, 20.5, 310.7),
    "Purdue": ("Big Ten", 18.8, 333.7, 31.8, 411.7),
    "Michigan State": ("Big Ten", 24.6, 322.1, 29.9, 369.7),
    "Iowa": ("Big Ten", 29.3, 312.7, 16.1, 266.5),
    "UCLA": ("Big Ten", 18.2, 306.6, 33.4, 380.8),
    "Minnesota": ("Big Ten", 23.0, 282.1, 22.9, 310.4),
    "Wisconsin": ("Big Ten", 12.8, 234.7, 21.6, 306.0),
    "Florida State": ("ACC", 33.0, 462.3, 22.0, 311.6),
    "Georgia Tech": ("ACC", 32.2, 456.5, 25.0, 385.5),
    "Duke": ("ACC", 34.6, 409.4, 29.4, 412.8),
    "Virginia": ("ACC", 30.8, 408.9, 19.6, 295.8),
    "SMU": ("ACC", 32.2, 402.2, 20.5, 377.0),
    "Miami": ("ACC", 30.9, 396.6, 14.8, 273.2),
    "Wake Forest": ("ACC", 28.1, 385.2, 22.1, 314.2),
    "Clemson": ("ACC", 27.2, 380.4, 20.5, 341.8),
    "NC State": ("ACC", 30.2, 379.5, 27.2, 402.2),
    "Louisville": ("ACC", 29.9, 370.5, 21.2, 287.5),
    "Pittsburgh": ("ACC", 33.7, 367.4, 24.8, 310.0),
    "Boston College": ("ACC", 25.4, 366.2, 32.8, 424.4),
    "California": ("ACC", 25.3, 339.2, 27.2, 351.2),
    "Virginia Tech": ("ACC", 21.4, 331.7, 30.2, 365.6),
    "Syracuse": ("ACC", 20.2, 320.7, 34.9, 417.0),
    "Stanford": ("ACC", 18.8, 279.6, 29.2, 394.0),
    "North Carolina": ("ACC", 19.2, 279.2, 24.5, 317.0),
    "Utah": ("Big 12", 41.3, 477.2, 18.9, 340.5),
    "Texas Tech": ("Big 12", 39.4, 446.5, 11.8, 237.7),
    "Baylor": ("Big 12", 31.1, 439.0, 32.6, 386.5),
    "TCU": ("Big 12", 30.7, 409.7, 25.3, 368.4),
    "Cincinnati": ("Big 12", 30.3, 404.2, 25.6, 390.1),
    "Arizona State": ("Big 12", 25.8, 403.9, 24.5, 339.2),
    "BYU": ("Big 12", 31.4, 393.4, 19.1, 318.4),
    "Arizona": ("Big 12", 31.5, 389.7, 19.3, 298.5),
    "Houston": ("Big 12", 29.1, 386.8, 22.8, 333.9),
    "Iowa State": ("Big 12", 27.4, 383.9, 20.2, 350.4),
    "Kansas": ("Big 12", 28.1, 376.9, 26.8, 375.7),
    "UCF": ("Big 12", 24.3, 370.2, 23.6, 322.7),
    "Kansas State": ("Big 12", 29.4, 364.8, 26.7, 368.4),
    "West Virginia": ("Big 12", 21.8, 334.9, 30.8, 395.0),
    "Colorado": ("Big 12", 20.9, 304.8, 30.5, 419.2),
    "Oklahoma State": ("Big 12", 14.2, 280.7, 33.3, 408.3),
    "Ole Miss": ("SEC", 36.9, 484.4, 21.1, 335.3),
    "Tennessee": ("SEC", 39.8, 454.4, 28.8, 379.5),
    "Vanderbilt": ("SEC", 38.5, 452.6, 22.8, 340.4),
    "Texas A&M": ("SEC", 33.8, 438.2, 21.0, 282.9),
    "Arkansas": ("SEC", 32.9, 435.2, 33.8, 413.4),
    "Missouri": ("SEC", 30.2, 406.4, 18.9, 257.5),
    "Georgia": ("SEC", 32.1, 397.1, 17.6, 289.4),
    "Texas": ("SEC", 30.5, 377.8, 20.3, 319.9),
    "Mississippi State": ("SEC", 30.4, 377.2, 30.2, 399.8),
    "Auburn": ("SEC", 26.8, 353.1, 20.7, 314.7),
    "Alabama": ("SEC", 29.5, 350.6, 19.2, 281.7),
    "Oklahoma": ("SEC", 26.2, 340.5, 15.5, 249.9),
    "Florida": ("SEC", 21.6, 332.3, 24.0, 365.3),
    "Kentucky": ("SEC", 23.0, 325.3, 26.4, 359.8),
    "LSU": ("SEC", 22.8, 316.5, 19.8, 312.6),
    "South Carolina": ("SEC", 22.7, 307.5, 22.1, 342.2),
    "Notre Dame": ("Independent", 42.0, 451.0, 17.6, 292.9),
}

EFFICIENCY_STATS = {t: (v[1], v[2], v[3], v[4]) for t, v in PCA_STATS.items()}
EFFICIENCY_STATS.update({
    "Charlotte": (14.3, 271.5, 36.3, 463.4),
    "Rice": (19.1, 292.1, 32.9, 392.2),
    "Navy": (31.5, 411.2, 25.0, 358.1),
    "North Texas": (45.1, 501.4, 26.5, 371.4),
    "UAB": (26.4, 386.5, 38.2, 417.7),
    "Western Kentucky": (29.5, 395.2, 22.8, 389.0),
})


SOS_SCALE = 40.0
HOME_FIELD_ADV = 2.5
MODEL_SCALE = 20.0
CONTINUITY_BASELINE = 0.45
CONTINUITY_WEIGHT = 15.0
IN_SEASON_K = 3.0
OFFENSE_WEIGHTS = {"QB": 0.40, "RB": 0.10, "WR": 0.20, "TE": 0.05, "OL": 0.25}
DEFENSE_WEIGHTS = {"DL": 0.35, "LB": 0.30, "DB": 0.35}

SCORING_WEIGHT = 0.60
EFFICIENCY_WEIGHT = 0.40
YARDS_PER_POINT = 15.0

VIG = 0.045
SPREAD_JUICE = -110


def blended_margin(team_lookup_name):
    key = NAME_ALIAS.get(team_lookup_name, team_lookup_name)
    margin = SCORING_MARGIN.get(key)
    if margin is None:
        raise KeyError(f"No 2025 scoring margin for '{team_lookup_name}' (alias tried '{key}').")
    eff = EFFICIENCY_STATS.get(team_lookup_name) or EFFICIENCY_STATS.get(key)
    if eff is None:
        return margin
    off_ppg, off_ypg, def_ppg, def_ypg = eff
    yardage_margin_pts = (off_ypg - def_ypg) / YARDS_PER_POINT
    return SCORING_WEIGHT * margin + EFFICIENCY_WEIGHT * yardage_margin_pts


def data_power_score(team_lookup_name):
    if team_lookup_name in PLACEHOLDER_TEAMS:
        margin, sos = PLACEHOLDER_TEAMS[team_lookup_name]
        return margin + (sos - 0.5) * SOS_SCALE
    key = NAME_ALIAS.get(team_lookup_name, team_lookup_name)
    sos = SOS_OWP.get(key)
    if sos is None:
        raise KeyError(f"No 2025 SOS data for '{team_lookup_name}' (alias tried '{key}').")
    return blended_margin(team_lookup_name) + (sos - 0.5) * SOS_SCALE


def prob_to_american_odds(p):
    if p is None or p <= 0.0 or p >= 1.0:
        return None
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def apply_vig(p_team, vig=VIG):
    p_opp = 1 - p_team
    return min(p_team * (1 + vig), 0.99), min(p_opp * (1 + vig), 0.99)


def posted_odds(win_prob, pred_margin):
    p_team_book, p_opp_book = apply_vig(win_prob)
    spread_line = round(pred_margin * 2) / 2
    return {
        "moneyline_team": prob_to_american_odds(p_team_book),
        "moneyline_opp": prob_to_american_odds(p_opp_book),
        "spread_team": f"{-spread_line:+.1f}",
        "spread_opp": f"{spread_line:+.1f}",
        "spread_juice": SPREAD_JUICE,
    }


def continuity_index(team):
    snaps = SNAP_RETENTION[team]
    off_idx = sum(OFFENSE_WEIGHTS[p] * snaps[p] for p in OFFENSE_WEIGHTS)
    def_idx = sum(DEFENSE_WEIGHTS[p] * snaps[p] for p in DEFENSE_WEIGHTS)
    return off_idx, def_idx, (off_idx + def_idx) / 2


def preseason_adjusted_score(team):
    _, _, overall_idx = continuity_index(team)
    return data_power_score(team) + (overall_idx - CONTINUITY_BASELINE) * CONTINUITY_WEIGHT


def win_probability(team_score, opp_score, site):
    hfa = {"Home": HOME_FIELD_ADV, "Away": -HOME_FIELD_ADV, "Neutral": 0.0}[site]
    return 1.0 / (1.0 + 10 ** (-((team_score + hfa) - opp_score) / MODEL_SCALE))


class Game:
    __slots__ = ("team", "week", "date", "opponent_display", "opponent_lookup",
                 "site", "team_score", "opp_score")

    def __init__(self, team, week, date, opponent_display, opponent_lookup, site):
        self.team, self.week, self.date = team, week, date
        self.opponent_display, self.opponent_lookup, self.site = opponent_display, opponent_lookup, site
        self.team_score = self.opp_score = None

    @property
    def played(self):
        return self.team_score is not None and self.opp_score is not None

    @property
    def actual_result(self):
        if not self.played:
            return None
        if self.team_score > self.opp_score:
            return 1.0
        if self.team_score == self.opp_score:
            return 0.5
        return 0.0


def load_schedule():
    return [Game(*row) for row in SCHEDULE]


def apply_results(games, results):
    lookup = {(t, w): (ts, os_) for t, w, ts, os_ in results}
    for g in games:
        if (g.team, g.week) in lookup:
            g.team_score, g.opp_score = lookup[(g.team, g.week)]


def preseason_win_prob_for_game(g):
    return win_probability(preseason_adjusted_score(g.team), data_power_score(g.opponent_lookup), g.site)


def live_adjusted_scores(games):
    live = {}
    for team in FOCUS_TEAMS:
        delta = sum(IN_SEASON_K * (g.actual_result - preseason_win_prob_for_game(g))
                    for g in games if g.team == team and g.played)
        live[team] = preseason_adjusted_score(team) + delta
    return live


def game_predictions(games):
    live_scores = live_adjusted_scores(games)
    rows = []
    for g in games:
        team_score, opp_score = live_scores[g.team], data_power_score(g.opponent_lookup)
        win_prob = win_probability(team_score, opp_score, g.site)
        hfa = {"Home": HOME_FIELD_ADV, "Away": -HOME_FIELD_ADV, "Neutral": 0.0}[g.site]
        pred_margin = round(((team_score + hfa) - opp_score) / 3.0, 1)
        odds = posted_odds(win_prob, pred_margin)
        row = {
            "team": g.team, "week": g.week, "date": g.date, "opponent": g.opponent_display,
            "site": g.site, "live_team_score": round(team_score, 1), "opp_power_score": round(opp_score, 1),
            "win_prob": round(win_prob, 4), "pred_margin": pred_margin, "played": g.played,
            "moneyline": odds["moneyline_team"], "spread": odds["spread_team"],
            "spread_juice": odds["spread_juice"],
        }
        if g.played:
            row["actual_score"] = f"{g.team_score}-{g.opp_score}"
            row["actual_result"] = g.actual_result
            row["hit"] = (g.actual_result > 0.5) == (preseason_win_prob_for_game(g) > 0.5)
        rows.append(row)
    return rows


def season_projection(games):
    preds = game_predictions(games)
    out = []
    for team in FOCUS_TEAMS:
        team_rows = [r for r in preds if r["team"] == team]
        played = [r for r in team_rows if r["played"]]
        unplayed = [r for r in team_rows if not r["played"]]
        actual_wins = sum(r["actual_result"] for r in played)
        proj_wins = actual_wins + sum(r["win_prob"] for r in unplayed)
        sd = math.sqrt(sum(r["win_prob"] * (1 - r["win_prob"]) for r in unplayed))
        accuracy = (sum(1 for r in played if r["hit"]) / len(played)) if played else None
        out.append({
            "team": team, "games": len(team_rows), "played": len(played),
            "actual_record": f"{actual_wins:g}-{len(played) - actual_wins:g}",
            "prediction_accuracy": round(accuracy, 3) if accuracy is not None else None,
            "expected_wins": round(proj_wins, 2), "std_dev": round(sd, 2),
            "range": f"{max(0, proj_wins - sd):.1f}-{min(len(team_rows), proj_wins + sd):.1f}",
            "projected_record": f"{round(proj_wins)}-{len(team_rows) - round(proj_wins)}",
        })
    return out


def print_report(games):
    print("=" * 70); print("LIVE ADJUSTED SCORES"); print("=" * 70)
    for team, score in live_adjusted_scores(games).items():
        print(f"  {team:<14} {score:6.1f}  (preseason: {preseason_adjusted_score(team):.1f})")
    print("\n" + "=" * 70); print("SEASON PROJECTION"); print("=" * 70)
    for row in season_projection(games):
        acc = f"{row['prediction_accuracy']*100:.0f}%" if row['prediction_accuracy'] is not None else "-"
        print(f"  {row['team']:<14} played {row['played']:>2}/{row['games']}  actual {row['actual_record']:>6}  "
              f"accuracy {acc:>4}  projected {row['projected_record']:>6}  "
              f"(expected wins {row['expected_wins']}, range {row['range']})")
    print("\n" + "=" * 70); print("UPCOMING GAMES (next unplayed game per team)"); print("=" * 70)
    preds = game_predictions(games)
    for team in FOCUS_TEAMS:
        ng = next((r for r in preds if r["team"] == team and not r["played"]), None)
        if ng:
            ml = ng["moneyline"]
            ml_str = f"+{ml}" if ml is not None and ml > 0 else str(ml)
            print(f"  {team:<14} Wk{ng['week']:>2} vs {ng['opponent']:<20} ({ng['site']:<7})  "
                  f"win prob {ng['win_prob']*100:5.1f}%  ML {ml_str:>6}  "
                  f"spread {ng['spread']:>6} ({ng['spread_juice']})")
        else:
            print(f"  {team:<14} season complete")


WEEKLY_RESULTS = [
]

games = load_schedule()
apply_results(games, WEEKLY_RESULTS)
print_report(games)


def monte_carlo(games, trials=10_000, seed=42):
    rng = np.random.default_rng(seed)
    preds = game_predictions(games)
    results = {}
    for team in FOCUS_TEAMS:
        rows = [r for r in preds if r["team"] == team]
        actual_wins = sum(r["actual_result"] for r in rows if r["played"])
        remaining = np.array([r["win_prob"] for r in rows if not r["played"]])
        sim_wins = (actual_wins + (rng.random((trials, len(remaining))) < remaining).sum(axis=1)
                    if len(remaining) else np.full(trials, actual_wins))
        n = len(rows)
        dist = {w: float(np.mean(sim_wins == w)) for w in range(n + 1)}
        results[team] = {
            "mean_wins": float(sim_wins.mean()), "distribution": dist,
            "p_undefeated": dist.get(n, 0.0),
            "p_11_plus": sum(p for w, p in dist.items() if w >= 11),
            "p_10_plus": sum(p for w, p in dist.items() if w >= 10),
            "p_bowl_eligible": sum(p for w, p in dist.items() if w >= 6),
            "p_losing_season": sum(p for w, p in dist.items() if w < 6),
        }
    return results


mc = monte_carlo(games, trials=10_000)
for team, r in mc.items():
    print(f"\n{team}  (mean {r['mean_wins']:.2f} wins)")
    print(f"  P(undefeated)     {r['p_undefeated']*100:5.1f}%")
    print(f"  P(11+ wins)       {r['p_11_plus']*100:5.1f}%")
    print(f"  P(10+ wins)       {r['p_10_plus']*100:5.1f}%")
    print(f"  P(bowl eligible)  {r['p_bowl_eligible']*100:5.1f}%")
    print(f"  P(losing season)  {r['p_losing_season']*100:5.1f}%")


def pca_clustering(k=5, random_state=42):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    teams = list(PCA_STATS.keys())
    X = np.array([[PCA_STATS[t][1], PCA_STATS[t][2], PCA_STATS[t][3], PCA_STATS[t][4],
                   SOS_OWP.get(NAME_ALIAS.get(t, t), 0.55)] for t in teams])
    X_adj = X.copy()
    X_adj[:, 2] *= -1
    X_adj[:, 3] *= -1
    Xz = StandardScaler().fit_transform(X_adj)
    pca = PCA(n_components=3)
    scores = pca.fit_transform(Xz)
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(scores[:, :2])
    cluster_pc1 = {c: scores[labels == c, 0].mean() for c in range(k)}
    order = sorted(cluster_pc1, key=lambda c: -cluster_pc1[c])
    relabel = {old: new + 1 for new, old in enumerate(order)}
    results = {t: {"conference": PCA_STATS[t][0], "pc1": float(scores[i, 0]),
                    "pc2": float(scores[i, 1]), "pc3": float(scores[i, 2]),
                    "cluster": relabel[labels[i]]} for i, t in enumerate(teams)}
    return results, pca.explained_variance_ratio_, k


pca_results, explained, k = pca_clustering()
print(f"Explained variance: PC1 {explained[0]*100:.1f}%, PC2 {explained[1]*100:.1f}%, PC3 {explained[2]*100:.1f}%\n")
for c in range(1, k + 1):
    members = sorted([t for t in pca_results if pca_results[t]["cluster"] == c],
                      key=lambda t: -pca_results[t]["pc1"])
    print(f"Cluster {c} ({len(members)} teams): {', '.join(members)}")
nd = pca_results["Notre Dame"]
c1 = [t for t in pca_results if pca_results[t]["cluster"] == 1]
print(f"\nNotre Dame: cluster {nd['cluster']}, PC1={nd['pc1']:.2f}, PC2={nd['pc2']:.2f}")
print(f"Cluster 1 (elite tier) PC1 range: {min(pca_results[t]['pc1'] for t in c1):.2f} "
      f"to {max(pca_results[t]['pc1'] for t in c1):.2f}")
