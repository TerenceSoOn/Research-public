#!/usr/bin/env python3
"""
Plot EDIS experiment results — line graphs comparing accuracy across
varying m (multiplier) and temperature values for 3 datasets.

Generates:
  1. per_temperature_comparison.png  — 3×3 grid (metrics × datasets),
     each subplot shows 3 temperature lines as m varies
  2. avg_across_temperatures.png     — 1×3 grid (one per dataset),
     3 metric lines averaged across all temperatures
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ──────────────────────── Configuration ────────────────────────

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_result")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")

DATASETS = ["aime24", "amc23", "math500"]
DATASET_LABELS = {"aime24": "AIME24", "amc23": "AMC23", "math500": "MATH500"}

M_VALUES = [1, 2, 4, 8, 16]
M_POSITIONS = list(range(len(M_VALUES)))  # Equal spacing: 0, 1, 2, 3, 4
TEMPS = [0.2, 0.6, 1.0]
TEMP_LABELS = {0.2: "T=0.2", 0.6: "T=0.6", 1.0: "T=1.0"}

# Metrics to plot (display name → regex pattern)
METRICS = {
    "Maj_EDIS@8":  r"Maj_EDIS@\d+\s+([\d.]+)%",
    "Best_EDIS@8": r"Best_EDIS@\d+\s+([\d.]+)%",
    "Avg_EDIS@8":  r"Avg_EDIS@\d+\s+([\d.]+)%",
}

# ─── Colors & markers matching reference figure style ───
# Orange for Maj, Green for Best, Blue for Avg
METRIC_STYLES = {
    "Maj_EDIS@8":  {"color": "#E6842A", "marker": "o", "label": "Maj_EDIS@8"},
    "Best_EDIS@8": {"color": "#4CAF50", "marker": "s", "label": "Best_EDIS@8"},
    "Avg_EDIS@8":  {"color": "#2979FF", "marker": "D", "label": "Avg_EDIS@8"},
}

TEMP_STYLES = {
    0.2: {"color": "#E6842A", "marker": "o", "label": "T = 0.2"},
    0.6: {"color": "#4CAF50", "marker": "s", "label": "T = 0.6"},
    1.0: {"color": "#2979FF", "marker": "D", "label": "T = 1.0"},
}

# ──────────────────────── Parsing ────────────────────────

def parse_all_summaries():
    """
    Returns nested dict:  data[dataset][m][T][metric] = accuracy (float)
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for dataset in DATASETS:
        summary_dir = os.path.join(BASE_DIR, dataset, "summary")
        if not os.path.isdir(summary_dir):
            print(f"Warning: {summary_dir} not found, skipping")
            continue

        for fpath in glob.glob(os.path.join(summary_dir, "*_summary.txt")):
            with open(fpath) as f:
                content = f.read()

            # Extract m, T from the header line
            hdr = re.search(r"m=(\d+).*T=([\d.]+)", content)
            if not hdr:
                continue
            m_val = int(hdr.group(1))
            t_val = float(hdr.group(2))

            for metric_name, pattern in METRICS.items():
                match = re.search(pattern, content)
                if match:
                    data[dataset][m_val][t_val][metric_name] = float(match.group(1))

    return data


# ──────────────────────── Plotting helpers ────────────────────────

def _setup_ax(ax, title, ylabel="Accuracy (%)", xlabel="m (Multiplier)"):
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(M_POSITIONS)
    ax.set_xticklabels([str(v) for v in M_VALUES])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=9)


def plot_per_temperature(data):
    """
    Figure 1: 3 metrics × 3 datasets grid.
    Each subplot: 3 temperature lines, x = m, y = accuracy.
    """
    metric_names = list(METRICS.keys())
    fig, axes = plt.subplots(
        len(metric_names), len(DATASETS),
        figsize=(14, 11),
        constrained_layout=True,
    )

    fig.suptitle(
        "EDIS — Per-Temperature Accuracy Comparison (k = 8)",
        fontsize=15, fontweight="bold", y=1.02,
    )

    for row_idx, metric in enumerate(metric_names):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]

            for t_val in TEMPS:
                ys = []
                for m in M_VALUES:
                    val = data[dataset].get(m, {}).get(t_val, {}).get(metric, np.nan)
                    ys.append(val)

                style = TEMP_STYLES[t_val]
                ax.plot(
                    M_POSITIONS, ys,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=6,
                    linewidth=2,
                    label=style["label"],
                )

            subtitle = f"{DATASET_LABELS[dataset]}  —  {metric}"
            _setup_ax(ax, subtitle)

            if row_idx == 0 and col_idx == len(DATASETS) - 1:
                ax.legend(fontsize=8, loc="best")

    out_path = os.path.join(OUTPUT_DIR, "per_temperature_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_avg_across_temps(data):
    """
    Figure 2: 1 × 3 grid (one per dataset).
    Each subplot: 3 metric lines, each averaged over 3 temperatures.
    """
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 5), constrained_layout=True)

    fig.suptitle(
        "EDIS — Average Accuracy Across Temperatures (k = 8)",
        fontsize=15, fontweight="bold", y=1.04,
    )

    for col_idx, dataset in enumerate(DATASETS):
        ax = axes[col_idx]

        for metric in METRICS:
            ys = []
            for m in M_VALUES:
                vals = []
                for t_val in TEMPS:
                    v = data[dataset].get(m, {}).get(t_val, {}).get(metric)
                    if v is not None:
                        vals.append(v)
                ys.append(np.mean(vals) if vals else np.nan)

            style = METRIC_STYLES[metric]
            ax.plot(
                M_POSITIONS, ys,
                color=style["color"],
                marker=style["marker"],
                markersize=7,
                linewidth=2.2,
                label=style["label"],
            )

        _setup_ax(ax, DATASET_LABELS[dataset])
        ax.legend(fontsize=9, loc="best")

    out_path = os.path.join(OUTPUT_DIR, "avg_across_temperatures.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ──────────────────────── Main ────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Parsing summary files …")
    data = parse_all_summaries()

    # Quick sanity check
    for ds in DATASETS:
        count = sum(1 for m in data[ds] for t in data[ds][m])
        print(f"  {ds}: {count} (m, T) combinations loaded")

    print()
    plot_per_temperature(data)
    plot_avg_across_temps(data)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
