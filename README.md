# CFB 2026 Tracker (Python)

Python port of the Excel workbook — same model, same numbers, same weekly-update
mechanism. Tracks Ole Miss, Notre Dame, Indiana, and Illinois through the 2026
season: real 2025 performance data (not preseason opinion rankings), a
position-weighted returning-production model, live in-season rating updates,
Monte Carlo simulation, PCA clustering against the rest of the Power 4, and a
full weekly content pipeline (plots, PDF report, recap, ATS record, model-vs-
market, shareable graphics).

`notebook/cfb_tracker_notebook.py` is a single-file, notebook-ready version of
the same model — paste it into Jupyter/VS Code if you'd rather not run it as a
multi-file package.

## Setup

```bash
pip install -r requirements.txt
```

## The one-command weekly routine

```bash
python run_all.py
```

Runs everything in sequence: ratings/projections, weekly recap, ATS record,
model-vs-market, all 6 plots, the PDF report, a burn card, and all 4 teams'
scouting report PDFs — skipping ATS/edge gracefully with a helpful message if
`vegas_lines.csv` is still empty. Add flags for the deeper stuff:

```bash
python run_all.py --montecarlo --pca --backtest --cfp --show
```

Everything below still works standalone too, if you just want one piece.

## Upgrade to real play-by-play data (recommended)

The model currently blends real 2025 scoring margin with box-score-derived
yardage margin. Box scores are real data, but noisy — garbage time, one-off
explosive plays, and opponent quality all get baked in undifferentiated.

`cfbd_fetch.py` pulls the actual answer: opponent-adjusted EPA/play (PPA) and
success rate from CollegeFootballData.com's play-by-play data — the same
category of data SP+, FPI, and real sportsbook models are built on. One-time
setup:

```bash
# 1. Get a free key: https://collegefootballdata.com/key
# 2. Set it as an environment variable, then:
python cfbd_fetch.py                # pulls 2025 season advanced stats
python cfbd_fetch.py --year 2024    # or a different season
```

This saves `cfbd_efficiency_2025.json` next to the other files. `cfb_model.py`
picks it up automatically from then on — no other code changes needed, and
nothing breaks if you skip this step (it falls back to the box-score
approach). Delete the JSON file any time to revert.

**To check which mode you're actually running**, the top of every
`update_tracker.py` / `run_all.py` run prints it plainly:

```
DATA MODE: real play-by-play EPA (CFBD, 68 teams, 2025) blended with box-score scoring margin
```
or
```
DATA MODE: box-score yardage margin (no cfbd_efficiency_2025.json found -- run 'python cfbd_fetch.py --year 2025' to upgrade) blended with scoring margin
```

Note: I can't fetch this data myself from inside this conversation — CFBD's
API needs a personal key and isn't reachable from my sandboxed environment —
so this is genuinely a step you'll need to run yourself, once.

## Multi-year calibration backtest

```bash
python cfbd_backtest.py --start-year 2016 --end-year 2024
```

`cfb_backtest.py` (no "d") checks one season for the 4 tracked teams — useful,
but a small sample. This one tests the model's actual methodology (same
constants, same win_probability() formula, imported directly from
cfb_model.py) against every real FBS game across as many seasons as you ask
for. For each test year, it builds power scores from the *prior* year's real
games + CFBD efficiency data, predicts every game in the test year, and scores
every prediction against what actually happened — Brier score, log-loss, and
a full calibration table (does "70% favorite" actually win about 70% of the
time?), plus a reliability diagram (`calibration.png`).

Needs the same `CFBD_API_KEY` as `cfbd_fetch.py`. First run will be slow (one
API call per year for games + advanced stats); results get cached in
`cfbd_backtest_cache/` so reruns are instant.

Add `--sweep-scale` to also search for the `MODEL_SCALE` value that actually
minimizes Brier score, instead of the value in `cfb_model.py` being a guess:

```bash
python cfbd_backtest.py --start-year 2016 --end-year 2024 --sweep-scale
```

Reuses the same cached games/scores from the main backtest — no new API calls,
just resweeps the probability-conversion step across a range of `MODEL_SCALE`
values (default 8 to 60, step 2 — override with `--scale-min`/`--scale-max`/
`--scale-step`) and reports which one actually minimizes Brier score and
log-loss, plus a `model_scale_sweep.png` chart. Tells you exactly what to
change `MODEL_SCALE` to at the top of `cfb_model.py`.

## Confidence intervals

Every power score and win probability is a point estimate with real
uncertainty behind it — a team's rating is built substantially from an
average scoring margin over a season's worth of games, and that average has
a standard error just like any sample mean does. `cfb_model.py` estimates
that error (`score_standard_error()`, using an assumed ~21-point game-to-game
margin standard deviation divided by √games played — a documented
approximation, not fit to this project's specific data) and propagates it
into a ~90% confidence band on both power scores and win probabilities.

Shows up automatically in `update_tracker.py` / `run_all.py`:

```
Ole Miss    18.3  (preseason: 18.3, ~90% CI: 9.4 to 27.2)
...
Ole Miss    Wk 1 vs Louisville  win prob 63.1% (40-81%, ~90% CI)  ML -194  spread -2.5 (-110)
```

Also available in `game_predictions()` output as `win_prob_low`/`win_prob_high`
for anything downstream that wants it. This is a separate kind of
uncertainty from `season_projection()`'s existing win/loss variance — that's
outcome randomness (a 70% favorite still loses 30% of the time, even with a
perfectly known rating); this is *rating* uncertainty (how confident are we
in the 70% itself).

## Ensemble: model + market

```bash
python cfb_edge.py
```

Now blends the model's probability with the market's instead of only
comparing them — `ensemble_probability()` in `cfb_edge.py`, weighted 70%
model / 30% market by default (`ENSEMBLE_MODEL_WEIGHT` in `config.py`).
Output adds an ensemble win probability and moneyline alongside the existing
model/market comparison and edge flag:

```
Notre Dame   Wk1 vs Wisconsin   model 85.1%  market 95.4%  ensemble 88.2% (ML -746)  edge -10.3pp <-- EDGE
```

The idea: rather than treating the market purely as something to beat, treat
it as a second, independently-informed estimate worth incorporating — a
closing line has injury reports, weather, and sharp money baked in that this
model doesn't see directly.

## CFP probability tracker

```bash
python cfb_cfp.py
```

Maps each team's Monte Carlo win-total distribution to a rough probability
of making the 12-team Playoff, using a win-total → historical-selection-rate
table (`WIN_TOTAL_TO_CFP_PROB` in `cfb_cfp.py`). Explicitly a simplified
estimate, not a full bracket simulation — it doesn't model specific
opponents, conference championship games, or what other bubble teams do,
just "teams that finish 10-2 have historically made the field about half the
time." Treat it as directional. Also available via `run_all.py --cfp`.

## Files

- `cfb_data.py` — all static data: schedules, 2025 scoring margin & strength of
  schedule (real, sourced numbers — not opinion-based rankings), returning
  production snap %, and the Power 4 + Notre Dame PCA dataset.
- `cfb_model.py` — the core model: Data Power Score, position-weighted
  Continuity Index, win probability, confidence intervals, and the in-season
  live-rating update.
- `cfb_analysis.py` — Monte Carlo season simulation and PCA/K-Means clustering.
- `cfb_cfp.py` — 12-team Playoff probability estimate.
- `config.py` — every tunable parameter, in one place.
- `cfb_plots.py` — charts (see below).
- `update_tracker.py` — the script you actually run.
- `weekly_results.csv` — where you log actual scores. Starts empty. Columns:
  `team,week,team_score,opp_score,team_yards,opp_yards` — the last two
  (total yards for/against) are optional; win/loss and ratings work fine
  without them, but they're needed for `pca_progression.png` below.

## Plots

```bash
pip install matplotlib
python cfb_plots.py            # saves 6 PNGs to plots/
python cfb_plots.py --show     # also opens them interactively
```

- **`live_scores.png`** — each team's power rating from preseason through the
  current week (a step function — moves only on weeks a game was played).
- **`projection_evolution.png`** — how the season-long expected-wins call has
  moved as actual results came in. Reconstructed retroactively from
  `weekly_results.csv`, so this fills in properly no matter when you start
  plotting.
- **`schedule_win_probabilities.png`** — win probability for every game on
  each team's schedule; blue bars are upcoming, green/red mark whether a
  played game hit or missed the preseason favorite call.
- **`monte_carlo.png`** — the win-total distribution from a fresh 10,000-trial
  simulation.
- **`pca_clusters.png`** — the Power 4 + Notre Dame scatter, Notre Dame circled,
  frozen at final 2025 positions.
- **`pca_progression.png`** — the same space, but each tracked team's
  season-to-date 2026 performance is projected onto it week by week (via the
  fitted PCA's own `.transform()`, not a refit — an apples-to-apples position
  on the same axes), so you can watch a team's dot move through last year's
  landscape in real time. Needs `team_yards`/`opp_yards` logged in
  `weekly_results.csv` — falls back to showing frozen 2025 positions with a
  note in the title if none are logged yet.

Each plotting function (`plot_live_scores`, `plot_projection_evolution`, etc.)
also just returns its `Figure`, so in a notebook you can call one directly and
it'll render inline instead of saving to disk.

## Weekly PDF report

```bash
python cfb_report.py                  # auto-detects the current week
python cfb_report.py --week 5         # force a specific week number
python cfb_report.py --out report.pdf # custom filename
```

Builds a 5-page PDF, one chart per page, each headed **"Football Update —
Week N."** Same charts as `cfb_plots.py`, just packaged to send out. The week
number auto-detects from the highest played week in `weekly_results.csv`
(preseason = Week 0) unless you override it with `--week`.

## Backtest: would this have called 2025 correctly?

```bash
python cfb_backtest.py
```

Strips the model down to only what it would have known *before* the 2025
season — 2024 trailing scoring margin + SOS, no continuity data — and checks
whether that ranking matches what actually happened. Honest scope: this is a
4-team, season-level directional check (real 2024 inputs, real 2025 final
records), not a full per-game backtest — that would need a complete 2025 game
log for every opponent, which isn't gathered here. Still finds something real:
the trailing-data model had **Notre Dame ranked #1** of the four teams
entering 2025. They finished 3rd and missed the Playoff; Indiana (ranked #2)
won the national title.

## Model vs. market

```bash
python cfb_edge.py
```

Reads `vegas_lines.csv` (starts empty — add `team,week,vegas_spread,vegas_moneyline`
rows as you see lines during the week) and flags games where the model's win
probability disagrees with the market's by more than 7 percentage points. The
market spread is converted to a probability using the same functional form as
the model's own margin↔probability conversion, so the comparison is
apples-to-apples.

## Against the spread (ATS)

```bash
python cfb_ats.py
```

A team can be 3-1 straight up and 1-3 ATS — this is usually the funnier (and
more honest) stat. Uses the same `vegas_lines.csv` as `cfb_edge.py`, cross-
referenced with actual scores from `weekly_results.csv`, to give each team's
cover/no-cover/push record and a game-by-game breakdown. Needs both a logged
line and a played result for a given team/week to count.

## Weekly recap (auto-generated)

```bash
python cfb_recap.py                   # auto-detects current week
python cfb_recap.py --week 5
```

A few lines of real recap — this week's results, biggest rating mover, season
projections — plus a "Notre Dame Corner" that rotates through data-backed
one-liners (SOS gap, PCA cluster miss, nearest-neighbor comp, the backtest
finding above) so you're not sending the same joke every week. The rotation
is deterministic per week (seeded by week number), not random each run.

## Burn card (shareable graphic)

```bash
python cfb_burncard.py                # saves burn_card.png
python cfb_burncard.py --week 5
python cfb_burncard.py --seed 0       # force a specific card instead of rotating
```

A single square, dark-mode, stat-card graphic — the same rotating angles as
the recap, sized to screenshot straight into a group chat.

## Scouting reports (per-team PDF)

```bash
python cfb_scouting_report.py                    # all 4 tracked teams
python cfb_scouting_report.py --team Illinois     # just one
python cfb_scouting_report.py --week 5            # force a specific week
```

One 4-page PDF per team (`Scouting_Report_<Team>_Week<N>.pdf`), styled like a
real scouting report:

**Page 1 — the data:**
- **Header** — team colors, record, week number.
- **Season baseline** — Off/Def PPG, Off/Def YPG, live rating, and projected
  final record, all computed through the current week.
- **Week-over-week trend** — each played week's offense/defense shown as a %
  deviation from the season baseline (green/red for offense; blue/orange for
  defense, sign-flipped so "fewer points allowed" reads as positive). Shows
  momentum at a glance — is a team trending up or down relative to their own
  average, not just relative to the schedule.
- **Performance cluster position** — that team's specific trajectory through
  the frozen 2025 Power 4 + Notre Dame field (same underlying data as
  `pca_progression.png`, cropped to one team and given most of the page).

**Pages 2-3 — how to trust it, for readers who don't:** a plain-language
walkthrough of how the Power Score is built, how win probability is
calculated, what the PCA cluster chart actually means, the backtest results
(Brier score across 10,370 real games) as evidence it isn't just guessing,
and a glossary (SOS, EPA, success rate, Brier score, continuity index,
cluster, vig) — written explicitly for someone skeptical of "the algorithm
said so," emphasizing that every number is a checkable formula, not a
trained black-box model.

**Page 4 — model performance tracker:** a game-by-game table of every
prediction the model made for THIS team this season — predicted win
probability, actual result, hit/miss, predicted margin vs. actual margin —
using the *preseason* prediction for each game, never anything adjusted
after the fact. Below it, a season-to-date summary (accuracy %, this team's
own Brier score, average margin error) with context against the model's
larger multi-season validation, so a good or bad small sample doesn't get
mistaken for proof either way.

Needs `team_yards`/`opp_yards` logged in `weekly_results.csv` for the full
picture on page 1 (the YPG baseline and the PCA trajectory both depend on
it) — without yardage, everything else on the report still renders
correctly, just with YPG shown as "—" and the cluster position frozen at
2025. Pages 2-4 don't depend on yardage; page 4 needs at least one played
game to show anything (renders a clean placeholder message otherwise).

**Team logos:** each page's header shows a small badge to the left of the
team name. Drop a real logo file in a `logos/` folder next to the scripts,
named to match the team (lowercase, spaces to underscores):

```
logos/ole_miss.png
logos/notre_dame.png
logos/indiana.png
logos/illinois.png
```

PNG, JPG, or JPEG all work; a roughly square image looks best. No `logos/`
folder, or a missing file for a given team, isn't an error — that team just
gets a clean fallback badge instead (a colored circle with their initials,
not a copy of any real logo). I can't fetch official team logos myself —
they're trademarked assets, and my network access doesn't reach image hosts
or team athletics sites anyway — so this is a step you'd do yourself, e.g.
downloading from each team's official athletics site or media kit.

**Team colors** live in exactly one place — `TEAM_COLORS`/`TEAM_ACCENT` in
`cfb_data.py` — and everything else (`cfb_plots.py`, `cfb_scouting_report.py`)
imports from there instead of keeping its own copy, so there's one dictionary
to edit, not several that can quietly drift out of sync with each other.
Current values, sourced from official team/conference brand guides:

| Team | Primary (`TEAM_COLORS`) | Accent (`TEAM_ACCENT`) |
|---|---|---|
| Ole Miss | `#006BA6` Powder Blue *(alternate color, used here by choice — actual primary is Navy)* | `#13294B` Navy |
| Notre Dame | `#0C2340` ND Blue | `#C99700` Standard Dome Gold |
| Indiana | `#990000` Crimson | `#EEEDEB` Cream |
| Illinois | `#FF5F05` Illini Orange | `#13294B` Illini Blue |

## Weekly workflow

1. After each Saturday's games, open `weekly_results.csv` and add a row per
   game for each of the 4 teams:

   ```csv
   team,week,team_score,opp_score,team_yards,opp_yards
   Ole Miss,1,38,17,480,320
   Illinois,1,20,24,300,380
   ```

   The last two columns (total yards) are optional — leave them blank if you
   don't want to bother tracking yardage; everything works fine without them
   except `pca_progression.png`, which needs them to plot real movement.

2. Run:

   ```bash
   python update_tracker.py
   ```

   This prints each team's live rating (shifted by how much they over/under-
   performed their *preseason* expectation — this is what avoids circularity:
   the delta is always measured against the frozen preseason number, never the
   live one), a season projection blending locked-in wins with fresh
   probabilities for what's left, and the next unplayed game for each team.

3. Add flags for the deeper analysis:

   ```bash
   python update_tracker.py --montecarlo   # 10,000-trial season simulation
   python update_tracker.py --pca          # Power 4 + Notre Dame clustering
   ```

## Tuning

Every tunable parameter lives in exactly one place — `config.py` — not
scattered across individual files:

```python
SOS_SCALE = 40.0            # how much schedule strength affects Data Power Score
HOME_FIELD_ADV = 2.5        # points
MODEL_SCALE = 20.0          # win-probability sensitivity (lower = more upsets)
CONTINUITY_BASELINE = 0.45  # "neutral" returning-production level
CONTINUITY_WEIGHT = 15.0    # points per 1.0 of continuity deviation
IN_SEASON_K = 3.0           # how fast the model reacts to results
OFFENSE_WEIGHTS = {...}     # QB/RB/WR/TE/OL position value weights
DEFENSE_WEIGHTS = {...}     # DL/LB/DB position value weights

SCORING_WEIGHT = 0.60       # weight on final-score margin
EFFICIENCY_WEIGHT = 0.40    # weight on yardage-margin efficiency (must sum to 1.0)
YARDS_PER_POINT = 15.0      # yardage margin -> point-equivalent conversion
VIG = 0.045                 # bookmaker overround applied to posted odds
SPREAD_JUICE = -110         # standard point-spread juice

CFBD_CACHE_YEAR = 2025      # which year's cfbd_efficiency_<year>.json to use
PPA_PLAYS_PER_GAME = 70.0   # EPA/play -> point-equivalent/game conversion
EDGE_THRESHOLD = 0.07       # cfb_edge.py's model-vs-market flag threshold
```

Edit `config.py` and rerun — no other files need to change.
`cfb_model.py`, `cfb_edge.py`, and everything downstream of them (including
`cfbd_backtest.py`, which re-exports several of these by importing from
`cfb_model`) all read from this same file, so there's one number to update,
not several copies that can drift out of sync.

**Applying a `--sweep-scale` result:** once you've run
`python cfbd_backtest.py --start-year 2016 --end-year 2024 --sweep-scale`
and it reports the Brier-minimizing value, that's a one-line edit —
change `MODEL_SCALE` in `config.py` to that number.

## Efficiency blend & posted odds

Two upgrades on top of the base model:

- **Efficiency-blended power score.** Final-score margin alone bakes in garbage
  time, special-teams luck, and short fields off turnovers. `blended_margin()`
  combines it with 2025 yardage margin (`cfb_data.EFFICIENCY_STATS` — real
  off/def PPG and Yds/G for every opponent on all four schedules, not just
  Power 4 + Notre Dame) converted to a point-equivalent via `YARDS_PER_POINT`.
- **Posted odds.** `posted_odds()` takes the model's fair win probability,
  applies a proportional vig (`VIG`) to both sides, converts to an American
  moneyline, and rounds the predicted margin to a standard −110 point spread —
  i.e., what a sportsbook's actual line would say off this model, not just the
  "fair" number. Note: the vig here is a simple proportional split; real books
  use power/Shin-style methods that behave better at the extremes, so treat
  very lopsided moneylines (e.g. a 40-point favorite) as illustrative rather
  than literal.

## Using it programmatically

```python
from cfb_model import load_schedule, load_results, game_predictions, season_projection

games = load_schedule()
load_results(games)              # reads weekly_results.csv
preds = game_predictions(games)  # list of dicts, one per game
proj = season_projection(games)  # list of dicts, one per team
```

Both `game_predictions` and `season_projection` return plain lists of dicts —
drop them into a `pandas.DataFrame` if you want to slice/filter/export:

```python
import pandas as pd
df = pd.DataFrame(game_predictions(games))
```
