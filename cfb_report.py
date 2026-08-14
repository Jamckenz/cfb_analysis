from __future__ import annotations
import argparse
import os

from matplotlib.backends.backend_pdf import PdfPages

from cfb_model import load_schedule, load_results
import cfb_plots as P


def current_week(games) -> int:
    played_weeks = [g.week for g in games if g.played]
    return max(played_weeks) if played_weeks else 0


def add_week_header(fig, week: int):
    existing = fig._suptitle.get_text() if fig._suptitle is not None else None
    header = f"Football Update — Week {week}"
    if existing:
        fig.suptitle(f"{header}\n{existing}", fontsize=15, fontweight="bold", y=1.05)
    else:
        fig.suptitle(header, fontsize=16, fontweight="bold", y=1.02)


def generate_report(week: int | None = None, out_path: str | None = None) -> str:
    games = load_schedule()
    load_results(games)
    if week is None:
        week = current_week(games)
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), f"Football_Update_Week_{week}.pdf")

    figs = [
        P.plot_live_scores(games),
        P.plot_projection_evolution(games),
        P.plot_schedule_win_probabilities(games),
        P.plot_monte_carlo(games),
        P.plot_pca_clusters(),
    ]
    for fig in figs:
        add_week_header(fig, week)

    with PdfPages(out_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None,
                         help="Week number for the title (default: auto-detected from weekly_results.csv)")
    parser.add_argument("--out", type=str, default=None, help="Output PDF path")
    args = parser.parse_args()

    path = generate_report(week=args.week, out_path=args.out)
    print(f"Saved {path}")
