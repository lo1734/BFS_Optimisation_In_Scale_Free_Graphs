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

    threads = []
    mteps = []
    speedups = []
    efficiencies = []

    serial_mteps = None
    with open(args.csv, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 7:
                algo = parts[0]
                t = int(parts[1])
                if "Serial" in algo:
                    serial_mteps = float(parts[3])
                elif "Naive" in algo:
                    threads.append(t)
                    mteps.append(float(parts[3]))
                    speedups.append(float(parts[4]))
                    efficiencies.append(float(parts[5]))

    if not threads:
        print("Error: No Naive OpenMP scaling data found in CSV.")
        sys.exit(1)

    threads = np.array(threads)
    speedups = np.array(speedups)
    mteps = np.array(mteps)
    efficiencies = np.array(efficiencies)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # Subplot 1: Speedup vs Ideal
    ax1.plot(threads, threads, "--", color="gray", label="Ideal Linear Speedup", linewidth=2)
    ax1.plot(threads, speedups, "o-", color="#d62728", label="Naive OpenMP BFS", linewidth=2.5, markersize=8)
    ax1.set_xlabel("Number of Threads", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Speedup over Serial Baseline", fontsize=12, fontweight="bold")
    ax1.set_title("Strong Scaling Speedup ($S_T$)", fontsize=13, fontweight="bold")
    ax1.set_xticks(threads)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=11)

    for x, y in zip(threads, speedups):
        ax1.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontweight="bold", color="#d62728")

    # Subplot 2: Traversed Edges Per Second (MTEPS)
    if serial_mteps is not None:
        ax2.axhline(serial_mteps, color="#2ca02c", linestyle="--", linewidth=2, label=f"Serial Oracle Baseline ({serial_mteps:.1f} M)")
    ax2.plot(threads, mteps, "s-", color="#1f77b4", label="Naive OpenMP MTEPS", linewidth=2.5, markersize=8)
    ax2.set_xlabel("Number of Threads", fontsize=12, fontweight="bold")
    ax2.set_ylabel("MTEPS ($10^6$ Edges / s)", fontsize=12, fontweight="bold")
    ax2.set_title("Traversal Throughput (MTEPS)", fontsize=13, fontweight="bold")
    ax2.set_xticks(threads)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=11)

    for x, y in zip(threads, mteps):
        ax2.annotate(f"{y:.1f} M", (x, y), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontweight="bold", color="#1f77b4")

    plt.suptitle(args.title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()

    print(f"Scaling curve plot saved to: {args.output}")

if __name__ == "__main__":
    main()
