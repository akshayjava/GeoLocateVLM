"""
Compare multiple benchmark result JSON files in a unified table.

Each JSON file should be the output of src/benchmark.py (run_benchmark).

Usage
-----
    python src/benchmark_compare.py results/run_a.json results/run_b.json

    # Or compare everything in the results/ directory:
    python src/benchmark_compare.py results/*.json

Output example
--------------
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ Run                    │ @1km  │ @25km │@200km │@750km │@2500k│  med  │ fail│
    ├────────────────────────┼───────┼───────┼───────┼───────┼──────┼───────┼─────┤
    │ random_baseline        │  0.0% │  0.1% │  0.8% │  3.8% │ 14%  │5020km │  0% │
    │ zero_shot              │  1.2% │  4.5% │ 14.3% │ 32.1% │ 61%  │1840km │ 12% │
    │ geolocate_vlm_r8       │  2.1% │  7.8% │ 21.5% │ 43.2% │ 73%  │1120km │  5% │
    │ geolocate_vlm_r16      │  2.9% │  9.4% │ 24.7% │ 47.6% │ 78%  │ 980km │  4% │
    └─────────────────────────────────────────────────────────────────────────────┘
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

_THRESHOLDS = [1, 25, 200, 750, 2500]
_COL_WIDTHS = {
    "name": 22,
    "acc": 6,
    "median": 7,
    "fail": 5,
}


def _row(name, metrics, *, highlight=False):
    def acc(t):
        key = f"acc_@{t}km"
        v = metrics.get(key)
        if v is None:
            return " " * _COL_WIDTHS["acc"]
        return f"{v * 100:5.1f}%"

    med = metrics.get("median_error_km")
    med_str = f"{med:6.0f}km" if med is not None else " " * _COL_WIDTHS["median"]

    fail = metrics.get("parse_failure_rate")
    fail_str = f"{fail * 100:4.0f}%" if fail is not None else " " * _COL_WIDTHS["fail"]

    cols = [
        f"{name:<{_COL_WIDTHS['name']}}",
        *[acc(t) for t in _THRESHOLDS],
        med_str,
        fail_str,
    ]
    row_str = " │ ".join(cols)
    prefix = "▶ " if highlight else "  "
    return f"│{prefix}{row_str} │"


def _header():
    cols = [
        f"{'Run':<{_COL_WIDTHS['name']}}",
        *[f"@{t}km".center(_COL_WIDTHS["acc"]) for t in _THRESHOLDS],
        "median".center(_COL_WIDTHS["median"]),
        "fail%".center(_COL_WIDTHS["fail"]),
    ]
    return "│  " + " │ ".join(cols) + " │"


def _separator(char="─"):
    parts = [
        char * (_COL_WIDTHS["name"] + 2),
        *[char * (_COL_WIDTHS["acc"] + 2) for _ in _THRESHOLDS],
        char * (_COL_WIDTHS["median"] + 2),
        char * (_COL_WIDTHS["fail"] + 2),
    ]
    return "├" + "┼".join(parts) + "┤"


def _top_border():
    total_w = len(_header())
    return "┌" + "─" * (total_w - 2) + "┐"


def _bottom_border():
    total_w = len(_header())
    return "└" + "─" * (total_w - 2) + "┘"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(result_paths, sort_by="median_error_km"):
    """
    Load benchmark JSON reports and print a comparison table.

    Parameters
    ----------
    result_paths : list of file paths produced by src/benchmark.py
    sort_by      : metric key to sort rows by (ascending). Use None for
                   file order.
    """
    reports = []
    for path in result_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found — skipped.", file=sys.stderr)
            continue
        with open(path) as fh:
            data = json.load(fh)
        name = os.path.splitext(os.path.basename(path))[0]
        reports.append((name, data.get("metrics", {}), data))

    if not reports:
        print("No valid result files found.")
        return

    if sort_by:
        reports.sort(key=lambda r: r[1].get(sort_by, float("inf")))

    # Find best (lowest) median for highlighting
    best_median = min(
        r[1].get("median_error_km", float("inf")) for r in reports
    )

    print()
    print(_top_border())
    print(_header())
    print(_separator())
    for name, metrics, raw in reports:
        highlight = metrics.get("median_error_km") == best_median
        print(_row(name, metrics, highlight=highlight))
    print(_bottom_border())
    print()

    # Extra: print per-run metadata
    for name, metrics, raw in reports:
        n = raw.get("n_samples", "?")
        elapsed = raw.get("elapsed_seconds", "?")
        # Support both model benchmark reports ("dataset") and baseline reports ("baseline")
        if "dataset" in raw:
            kind = f"dataset={raw['dataset']}"
        elif "baseline" in raw:
            kind = f"baseline={raw['baseline']}"
        else:
            kind = "type=unknown"
        extra_parts = [kind, f"n={n}"]
        if "n_missing_images" in raw:
            extra_parts.append(f"missing={raw['n_missing_images']}")
        extra_parts.append(f"elapsed={elapsed}s")
        print(f"  {name}: {', '.join(extra_parts)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple GeoLocateVLM benchmark JSON results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results",
        nargs="+",
        metavar="results/run.json",
        help="One or more JSON files produced by src/benchmark.py",
    )
    parser.add_argument(
        "--sort_by",
        default="median_error_km",
        help="Metric to sort rows by (ascending). Default: median_error_km",
    )
    args = parser.parse_args()
    compare(args.results, sort_by=args.sort_by)


if __name__ == "__main__":
    main()
