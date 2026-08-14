from __future__ import annotations
import sys

from cfb_model import print_report
from cfb_analysis import print_monte_carlo, print_pca_summary
from cfb_plots import save_all_plots
from cfb_report import generate_report
from cfb_recap import generate_recap
from cfb_ats import print_ats_report
from cfb_edge import find_edges
from cfb_burncard import make_burn_card
from cfb_scouting_report import save_scouting_reports
from cfb_cfp import print_cfp_probabilities


def section(title: str):
    print("\n" + "#" * 78)
    print(f"# {title}")
    print("#" * 78)


def main():
    args = sys.argv[1:]

    section("TRACKER: ratings, season projection, upcoming games")
    print_report()

    if "--montecarlo" in args:
        section("MONTE CARLO SIMULATION")
        print_monte_carlo()

    if "--pca" in args:
        section("POWER 4 PCA CLUSTERING")
        print_pca_summary()

    if "--backtest" in args:
        section("BACKTEST: 2024 data vs. actual 2025 results")
        from cfb_backtest import run_backtest
        run_backtest()

    if "--cfp" in args:
        section("12-TEAM PLAYOFF PROBABILITY")
        print_cfp_probabilities()

    section("WEEKLY RECAP")
    print(generate_recap())

    section("AGAINST THE SPREAD (ATS)")
    print_ats_report()

    section("MODEL vs. MARKET")
    find_edges()

    section("PLOTS")
    save_all_plots(show="--show" in args)

    section("PDF REPORT")
    pdf_path = generate_report()
    print(f"Saved {pdf_path}")

    section("BURN CARD")
    fig = make_burn_card()
    fig.savefig("burn_card.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("Saved burn_card.png")

    section("SCOUTING REPORTS")
    save_scouting_reports()

    section("DONE")
    print("Everything's regenerated off the current weekly_results.csv / vegas_lines.csv.")


if __name__ == "__main__":
    main()
