from __future__ import annotations
import argparse
import random

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from cfb_model import load_schedule, load_results
from cfb_report import current_week

CARDS = [
    {
        "headline": "SCHEDULE CHECK",
        "stat": ".559",
        "stat_label": "Notre Dame's 2025 Strength of Schedule",
        "sub": "Cluster 1 average (Oregon, Indiana, Ohio State, Miami, Tex Tech): .600s+",
        "footer": "the numbers, not us",
    },
    {
        "headline": "CLUSTER REJECT",
        "stat": "TIER 2",
        "stat_label": "Where the PCA model actually put Notre Dame",
        "sub": "An algorithm with no opinions still wouldn't call them elite",
        "footer": "unsupervised learning, supervised burn",
    },
    {
        "headline": "NEAREST COMP",
        "stat": "OLE MISS",
        "stat_label": "Notre Dame's closest 2025 statistical twin",
        "sub": "A 3-loss team from Oxford, Mississippi",
        "footer": "birds of a feather",
    },
    {
        "headline": "MODEL VS. REALITY",
        "stat": "#1 -> #3",
        "stat_label": "Our own backtest had ND ranked best of 4 entering 2025",
        "sub": "They finished 3rd. Indiana (ranked #2) won it all.",
        "footer": "even the model was too kind",
    },
]

BG = "#0B1F3A"
ACCENT = "#C00000"
TEXT = "#FFFFFF"
MUTE = "#9FB3D1"


def make_burn_card(week: int | None = None, seed: int | None = None):
    games = load_schedule()
    load_results(games)
    if week is None:
        week = current_week(games)

    rng = random.Random(seed if seed is not None else week)
    card = rng.choice(CARDS)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.5, 0.92, f"FOOTBALL UPDATE — WEEK {week}", ha="center", va="center",
            fontsize=13, color=MUTE, fontweight="bold", family="monospace")
    ax.plot([0.1, 0.9], [0.87, 0.87], color=ACCENT, linewidth=2)

    ax.text(0.5, 0.76, card["headline"], ha="center", va="center",
            fontsize=20, color=ACCENT, fontweight="bold", family="sans-serif")

    ax.text(0.5, 0.55, card["stat"], ha="center", va="center",
            fontsize=54, color=TEXT, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground=ACCENT)])

    ax.text(0.5, 0.38, card["stat_label"], ha="center", va="center",
            fontsize=13, color=TEXT, wrap=True)

    ax.text(0.5, 0.28, card["sub"], ha="center", va="center",
            fontsize=10.5, color=MUTE, style="italic", wrap=True)

    ax.plot([0.1, 0.9], [0.14, 0.14], color=MUTE, linewidth=0.6, alpha=0.5)
    ax.text(0.5, 0.08, card["footer"].upper(), ha="center", va="center",
            fontsize=9, color=MUTE, family="monospace")

    fig.tight_layout(pad=1.2)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--out", type=str, default="burn_card.png")
    parser.add_argument("--seed", type=int, default=None, help="Force a specific card instead of rotating by week")
    args = parser.parse_args()

    fig = make_burn_card(week=args.week, seed=args.seed)
    fig.savefig(args.out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Saved {args.out}")
