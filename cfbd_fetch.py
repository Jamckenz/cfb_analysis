from __future__ import annotations
import argparse
import json
import os
import sys

import requests

API_BASE = "https://api.collegefootballdata.com"
CACHE_PATH_TEMPLATE = os.path.join(os.path.dirname(__file__), "cfbd_efficiency_{year}.json")


def get_api_key() -> str:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        print("ERROR: CFBD_API_KEY environment variable not set.")
        print("Get a free key at https://collegefootballdata.com/key, then set it:")
        print('  Windows (PowerShell):  $env:CFBD_API_KEY = "your-key-here"')
        print("  Mac/Linux:              export CFBD_API_KEY=your-key-here")
        sys.exit(1)
    return key


def fetch_advanced_stats(year: int, api_key: str) -> list[dict]:
    url = f"{API_BASE}/stats/season/advanced"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, params={"year": year}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _flatten_side(side: dict) -> dict:
    havoc = side.get("havoc", {}) or {}
    field_position = side.get("fieldPosition", {}) or {}
    standard_downs = side.get("standardDowns", {}) or {}
    passing_downs = side.get("passingDowns", {}) or {}
    rushing_plays = side.get("rushingPlays", {}) or {}
    passing_plays = side.get("passingPlays", {}) or {}
    return {
        "plays": side.get("plays"),
        "drives": side.get("drives"),
        "ppa": side.get("ppa"),
        "total_ppa": side.get("totalPPA"),
        "success_rate": side.get("successRate"),
        "explosiveness": side.get("explosiveness"),
        "power_success": side.get("powerSuccess"),
        "stuff_rate": side.get("stuffRate"),
        "line_yards": side.get("lineYards"),
        "second_level_yards": side.get("secondLevelYards"),
        "open_field_yards": side.get("openFieldYards"),
        "total_opportunities": side.get("totalOpportunies"),
        "points_per_opportunity": side.get("pointsPerOpportunity"),
        "field_position_avg_start": field_position.get("averageStart"),
        "field_position_avg_predicted_points": field_position.get("averagePredictedPoints"),
        "havoc_total": havoc.get("total"),
        "havoc_front_seven": havoc.get("frontSeven"),
        "havoc_db": havoc.get("db"),
        "standard_downs_ppa": standard_downs.get("ppa"),
        "standard_downs_success_rate": standard_downs.get("successRate"),
        "standard_downs_explosiveness": standard_downs.get("explosiveness"),
        "passing_downs_ppa": passing_downs.get("ppa"),
        "passing_downs_success_rate": passing_downs.get("successRate"),
        "passing_downs_explosiveness": passing_downs.get("explosiveness"),
        "rushing_plays_ppa": rushing_plays.get("ppa"),
        "rushing_plays_success_rate": rushing_plays.get("successRate"),
        "rushing_plays_explosiveness": rushing_plays.get("explosiveness"),
        "passing_plays_ppa": passing_plays.get("ppa"),
        "passing_plays_success_rate": passing_plays.get("successRate"),
        "passing_plays_explosiveness": passing_plays.get("explosiveness"),
    }


def build_efficiency_dict(raw: list[dict]) -> dict:
    out = {}
    for row in raw:
        team = row.get("team")
        if not team:
            continue
        offense = _flatten_side(row.get("offense", {}) or {})
        defense = _flatten_side(row.get("defense", {}) or {})
        entry = {}
        for k, v in offense.items():
            entry[f"off_{k}"] = v
        for k, v in defense.items():
            entry[f"def_{k}"] = v
        out[team] = entry
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    api_key = get_api_key()
    print(f"Fetching {args.year} advanced season stats from CFBD...")
    raw = fetch_advanced_stats(args.year, api_key)
    print(f"Got {len(raw)} teams.")

    if raw:
        sample_path = os.path.join(os.path.dirname(__file__), f"cfbd_sample_record_{args.year}.json")
        with open(sample_path, "w") as f:
            json.dump(raw[0], f, indent=2)
        print(f"\nFull sample record (uncut) saved to {sample_path}")
        offense_keys = sorted((raw[0].get("offense") or {}).keys())
        print(f"Top-level offense/defense fields available: {offense_keys}")

    efficiency = build_efficiency_dict(raw)
    cache_path = CACHE_PATH_TEMPLATE.format(year=args.year)
    with open(cache_path, "w") as f:
        json.dump(efficiency, f, indent=2)
    print(f"\nSaved {cache_path} ({len(efficiency)} teams)")
    print("cfb_model.py will use this automatically from now on. "
          "Delete the file to fall back to box-score-based estimates.")


if __name__ == "__main__":
    main()
