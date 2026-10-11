#!/usr/bin/env python3
"""
plot_scaling.py
Plots strong scaling speedup, harmonic mean TEPS, and parallel efficiency
from the BFS benchmark CSV results.
"""

import os
import sys
import argparse
import numpy as np

os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot BFS strong scaling performance.")
    parser.add_argument("--csv", required=True, help="Input CSV path from benchmark_bfs")
    parser.add_argument("--output", default="data/scaling_curve.png", help="Output PNG path")
    parser.add_argument("--title", default="BFS Strong Scaling (Naive OpenMP vs Serial)", help="Plot title")
    args = parser.parse_args()

    data_by_algo = {}
    serial_mteps = None
    with open(args.csv, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 7:
                algo = parts[0]
                t = int(parts[1])
                mtep = float(parts[3])
                spd = float(parts[4])
                eff = float(parts[5])
                if "Serial" in algo:
                    serial_mteps = mtep
                else:
                    if algo not in data_by_algo:
                        data_by_algo[algo] = {"threads": [], "mteps": [], "speedup": [], "efficiency": []}
                    data_by_algo[algo]["threads"].append(t)
                    data_by_algo[algo]["mteps"].append(mtep)
                    data_by_algo[algo]["speedup"].append(spd)
                    data_by_algo[algo]["efficiency"].append(eff)

    if not data_by_algo:
        print("Error: No parallel scaling data found in CSV.")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # Collect all thread counts for x-ticks
    all_threads = sorted(list({t for a in data_by_algo.values() for t in a["threads"]}))

    # Subplot 1: Speedup vs Ideal
    ax1.plot(all_threads, all_threads, "--", color="gray", label="Ideal Linear Speedup", linewidth=2)

    color_map = {
        "Naive OpenMP BFS": ("#d62728", "o"),
        "Contention-Reduced BFS": ("#1f77b4", "s"),
        "Visited Bitmap BFS": ("#9467bd", "^"),
    }
    default_colors = ["#ff7f0e", "#8c564b", "#e377c2", "#7f7f7f"]

    c_idx = 0
    for algo, data in data_by_algo.items():
        th = np.array(data["threads"])
        spd = np.array(data["speedup"])
        color, marker = color_map.get(algo, (default_colors[c_idx % len(default_colors)], "D"))
        c_idx += 1

        ax1.plot(th, spd, f"{marker}-", color=color, label=algo, linewidth=2.5, markersize=8)
        for x, y in zip(th, spd):
            ax1.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8.5, fontweight="bold", color=color)

    ax1.set_xlabel("Number of Threads", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Speedup over Serial Baseline", fontsize=12, fontweight="bold")
    ax1.set_title("Strong Scaling Speedup ($S_T$)", fontsize=13, fontweight="bold")
    ax1.set_xticks(all_threads)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper left")

    # Subplot 2: Traversed Edges Per Second (MTEPS)
    if serial_mteps is not None:
        ax2.axhline(serial_mteps, color="#2ca02c", linestyle="--", linewidth=2, label=f"Serial Oracle ({serial_mteps:.1f} MTEPS)")

    c_idx = 0
    for algo, data in data_by_algo.items():
        th = np.array(data["threads"])
        mt = np.array(data["mteps"])
        color, marker = color_map.get(algo, (default_colors[c_idx % len(default_colors)], "D"))
        c_idx += 1

        ax2.plot(th, mt, f"{marker}-", color=color, label=f"{algo}", linewidth=2.5, markersize=8)
        for x, y in zip(th, mt):
            ax2.annotate(f"{y:.1f}M", (x, y), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8.5, fontweight="bold", color=color)

    ax2.set_xlabel("Number of Threads", fontsize=12, fontweight="bold")
    ax2.set_ylabel("MTEPS ($10^6$ Edges / s)", fontsize=12, fontweight="bold")
    ax2.set_title("Traversal Throughput (MTEPS)", fontsize=13, fontweight="bold")
    ax2.set_xticks(all_threads)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="lower right")

    plt.suptitle(args.title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()

    print(f"Scaling curve plot saved to: {args.output}")

if __name__ == "__main__":
    main()
