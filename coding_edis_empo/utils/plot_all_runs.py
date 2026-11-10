#!/usr/bin/env python3
"""
Comprehensive plot of all EDIS-EMPO experiments.
Groups runs by method and generates two figures:
  1. Main comparison: one representative run per method (run4, run9/20, run10/18, run12/16/19)
  2. Run-by-run view: all individual runs by method group (with shading for variance)

Output: run_logs/comparison_graph/all_runs_comparison.png
         run_logs/comparison_graph/method_groups.png
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_logs")
OUT_DIR  = os.path.join(LOGS_DIR, "comparison_graph")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Run metadata — label, color, style per method
# ─────────────────────────────────────────────
METHODS = {
    "empo_k8": {
        "label":  "EMPO k=8 (Baseline)",
        "color":  "#4C72B0",
        "ls":     "-",
        "runs":   ["run4_empo_8"],
    },
    "oversampled_empo": {
        "label":  "Oversampled EMPO (M=32→k=8)",
        "color":  "#55A868",
        "ls":     "--",
        "runs":   ["run9_empo_oversample", "run20_oversampled_32"],
    },
    "edis_filtering": {
        "label":  "EDIS Filtering (M=32→k=8, EDIS select)",
        "color":  "#C44E52",
        "ls":     "-",
        "runs":   ["run10_edis_filtering_8_32", "run18_filtering_32"],
    },
    "edis_weighting": {
        "label":  "EDIS Weighting (k=8, α=1.8)",
        "color":  "#DD8452",
        "ls":     "--",
        "runs":   ["run12_edis_weighting_32", "run13_edis_weighting_32",
                   "run16_edis_weighting", "run19_edis_weighting"],
    },
    "filtering_weighting": {
        "label":  "Filter + Weight (M=32→k=8 + α=1.8)",
        "color":  "#8172B2",
        "ls":     "-.",
        "runs":   ["run14_filtering_weighting"],
    },
}

# Teacher's expected ordering annotation
ORDERING = "Expected: EMPO-k8  <  EDIS-Weight  <  Oversample-EMPO  <  EDIS-Filter"


def parse_result(run_dir):
    """Parse result.txt → (steps[], accs[]) for a given run directory under LOGS_DIR."""
    path = os.path.join(LOGS_DIR, run_dir, "result.txt")
    steps, accs = [], []
    if not os.path.exists(path):
        return steps, accs
    seen_steps = set()
    with open(path) as f:
        for line in f:
            m = re.search(r"^\s*Step\s+(\d+):\s+[\d.]+\s+\(([\d.]+)%\)", line)
            if m:
                s = int(m.group(1))
                a = float(m.group(2))
                if s not in seen_steps:   # deduplicate (run18 has duplicate step 1500)
                    seen_steps.add(s)
                    steps.append(s)
                    accs.append(a)
    return steps, accs


def smooth(values, window=20):
    """Simple uniform moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def interp_to_common(steps, accs, common_steps):
    """Linearly interpolate a run's accs to a common step grid."""
    return np.interp(common_steps, steps, accs)


# ─────────────────────────────────────────────
# Figure 1: Best representative per method
#           + smoothed envelope for multi-run methods
# ─────────────────────────────────────────────
def plot_main_comparison():
    fig, ax = plt.subplots(figsize=(13, 7))

    for method_key, meta in METHODS.items():
        color = meta["color"]
        ls    = meta["ls"]
        label = meta["label"]
        runs  = meta["runs"]

        # Collect all run data
        all_runs = []
        for run in runs:
            s, a = parse_result(run)
            if s:
                all_runs.append((np.array(s), np.array(a)))

        if not all_runs:
            print(f"  [skip] No data for {method_key}")
            continue

        if len(all_runs) == 1:
            s, a = all_runs[0]
            ax.plot(s, smooth(a), color=color, ls=ls, lw=2, label=label)
        else:
            # Shared step range
            max_end   = min(sr[-1] for sr, _ in all_runs)
            min_start = max(sr[0]  for sr, _ in all_runs)
            common    = np.arange(min_start, max_end + 1, 10)

            interped = np.array([interp_to_common(sr, ar, common)
                                  for sr, ar in all_runs])
            mean_a   = interped.mean(axis=0)
            std_a    = interped.std(axis=0)

            # Plot mean
            ax.plot(common, smooth(mean_a), color=color, ls=ls, lw=2.2, label=label)
            # Shade ± 0.5 std for readability
            ax.fill_between(common,
                            smooth(mean_a - 0.5 * std_a),
                            smooth(mean_a + 0.5 * std_a),
                            color=color, alpha=0.15)

    # Horizontal guide lines
    for y, yt in [(50, "50%"), (52, "52%"), (54, "54%"), (56, "56%")]:
        ax.axhline(y, color="grey", ls=":", lw=0.8, alpha=0.5)

    ax.set_title("EDIS-EMPO Training: Validation Accuracy Comparison\n"
                 f"({ORDERING})",
                 fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%, AMC23 @16)", fontsize=12)
    ax.set_ylim(43, 59)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)

    out = os.path.join(OUT_DIR, "all_runs_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────
# Figure 2: All individual runs side-by-side
#           grouped by method (small multiples)
# ─────────────────────────────────────────────
def plot_individual_runs():
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharey=True)
    axes = axes.flatten()

    method_keys = list(METHODS.keys())
    for ax_i, method_key in enumerate(method_keys):
        ax    = axes[ax_i]
        meta  = METHODS[method_key]
        color = meta["color"]
        label = meta["label"]
        runs  = meta["runs"]

        peak_info = []
        for run in runs:
            s, a = parse_result(run)
            if not s:
                continue
            sa  = np.array(s)
            aa  = np.array(a)
            ax.plot(sa, smooth(aa, window=15), lw=1.8,
                    color=color, alpha=0.75, label=run)
            ax.plot(sa, aa, lw=0.5, color=color, alpha=0.25)
            peak_idx = np.argmax(aa)
            peak_info.append(f"{run}: peak {aa[peak_idx]:.1f}% @step {sa[peak_idx]}")
            ax.scatter([sa[peak_idx]], [aa[peak_idx]],
                       color=color, zorder=5, s=40, marker="*")

        ax.set_title(label.replace(" (", "\n("), fontsize=9, fontweight="bold")
        ax.set_xlabel("Steps", fontsize=8)
        ax.set_ylabel("Acc (%)", fontsize=8)
        ax.set_ylim(42, 59)
        ax.axhline(50, color="grey", ls=":", lw=0.8, alpha=0.4)
        ax.grid(True, linestyle="--", alpha=0.35)
        # Print peaks on the plot
        for j, info in enumerate(peak_info):
            ax.text(0.02, 0.97 - 0.09 * j, info.split(": ", 1)[1],
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment="top", color=color)

    # Hide unused subplot (5 methods, 6 subplots)
    for ax_i in range(len(method_keys), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("All Individual Runs by Method", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(OUT_DIR, "method_groups.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────
# Figure 3: Summary table (peak acc per method)
# ─────────────────────────────────────────────
def print_summary():
    print("\n" + "=" * 70)
    print(f"{'Method':<42} {'Runs':>5}  {'Avg Peak':>9}  {'Max Peak':>9}")
    print("=" * 70)
    rows = []
    for method_key, meta in METHODS.items():
        peaks = []
        for run in meta["runs"]:
            s, a = parse_result(run)
            if a:
                peaks.append(max(a))
        if peaks:
            rows.append((meta["label"], len(peaks), np.mean(peaks), max(peaks)))
            print(f"  {meta['label']:<42} {len(peaks):>4}   {np.mean(peaks):>8.2f}%  {max(peaks):>8.2f}%")
    print("=" * 70)
    return rows


if __name__ == "__main__":
    print("Parsing runs...")
    print_summary()
    print("\nGenerating Figure 1 (main comparison)...")
    plot_main_comparison()
    print("Generating Figure 2 (method groups)...")
    plot_individual_runs()
    print("\nDone!")
