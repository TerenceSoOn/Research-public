#!/usr/bin/env python3
"""
Plot validation accuracy comparison graphs for arbitrary EDIS runs.

Usage:
  python3 plot_runs.py run10_edis_filtering_8_32 run12_edis_weighting_32 --labels "Run 10" "Run 12" --out my_comparison.png
"""

import os
import re
import argparse
import matplotlib.pyplot as plt

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison_graph")

def parse_result_file(filepath):
    """
    Parses a result.txt file and returns lists of steps and accuracies.
    Looks for lines like:   Step   10: 0.4906 (49.06%)
    """
    steps = []
    accuracies = []
    
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return steps, accuracies
        
    with open(filepath, 'r') as f:
        for line in f:
            # Match strictly the lines exactly formatted like "Step XX: 0.XXXX (XX.XX%)"
            match = re.search(r"^\s*Step\s+(\d+):\s+[\d.]+\s+\(([\d.]+)%\)", line)
            if match:
                step = int(match.group(1))
                acc = float(match.group(2))
                steps.append(step)
                accuracies.append(acc)
                
    return steps, accuracies

def main():
    parser = argparse.ArgumentParser(description="Plot comparison graph for EDIS runs.")
    parser.add_argument("runs", nargs='+', help="List of run directory names to compare (e.g., run10_edis_filtering_8_32 run12_edis_weighting_32)")
    parser.add_argument("--labels", nargs='+', help="Optional custom labels for the legend. Must match the number of runs provided.")
    parser.add_argument("--out", default="comparison_graph.png", help="Output filename (saved in comparison_graph/ directory). Default: comparison_graph.png")
    parser.add_argument("--title", default="Validation Accuracy Comparison", help="Title of the graph.")
    
    args = parser.parse_args()
    
    # Validate labels if provided
    labels = args.labels
    if labels and len(labels) != len(args.runs):
        print("Warning: Number of labels provided does not match the number of runs. Falling back to using run names as labels.")
        labels = None
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Built-in colors for up to 10 lines (matplotlib default cycle), 
    # we don't need a strict mapping since it's dynamic now.
    
    success_count = 0
    
    for i, run in enumerate(args.runs):
        filepath = os.path.join(BASE_DIR, run, "result.txt")
        steps, accuracies = parse_result_file(filepath)
        
        if steps and accuracies:
            print(f"Parsed {len(steps)} data points for {run}")
            
            # Determine label
            label = labels[i] if labels else run
            
            plt.plot(
                steps, 
                accuracies, 
                label=label,
                linewidth=2,
                marker='.'
            )
            success_count += 1
        else:
            print(f"Failed to find data for {run}")

    if success_count == 0:
        print("No valid data found to plot. Exiting.")
        return

    plt.title(args.title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    output_path = os.path.join(OUTPUT_DIR, args.out)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph successfully saved to {output_path}")

if __name__ == "__main__":
    main()
