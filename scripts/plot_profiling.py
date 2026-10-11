#!/usr/bin/env python3
"""
plot_profiling.py
Generates multi-panel diagnostic visualization illustrating the bottlenecks
identified during Phase 3 BFS profiling:
1. Level-by-level atomic CAS contention & failure ratio
2. Per-thread edge scanning load imbalance
3. Active compute time vs barrier stall latency
"""

import os
import sys
import argparse
import numpy as np

os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Phase 3 BFS bottleneck profiling data.")
    parser.add_argument("--levels", default="data/bfs_profile_levels.csv", help="Input level profile CSV")
    parser.add_argument("--threads", default="data/bfs_profile_threads.csv", help="Input thread profile CSV")
    parser.add_argument("--output", default="data/profiling_bottlenecks.png", help="Output PNG path")
    args = parser.parse_args()

    if not os.path.exists(args.levels) or not os.path.exists(args.threads):
        print(f"Error: Missing profile CSVs ({args.levels} or {args.threads})")
        sys.exit(1)

    # 1. Parse level stats
    lvl_indices = []
    lvl_frontiers = []
    lvl_edges = []
    lvl_cas_attempts = []
    lvl_cas_success = []
    lvl_cas_fails = []
    lvl_elapsed_ms = []
    lvl_barrier_ms = []

    with open(args.levels, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 9:
                lvl_indices.append(int(p[0]))
                lvl_frontiers.append(int(p[1]))
                lvl_edges.append(int(p[2]))
                lvl_cas_attempts.append(int(p[3]))
                lvl_cas_success.append(int(p[4]))
                lvl_cas_fails.append(int(p[5]))
                lvl_elapsed_ms.append(float(p[6]))
                lvl_barrier_ms.append(float(p[7]))

    # 2. Parse thread stats
    t_ids = []
    t_edges = []
    t_active_ms = []
    t_barrier_ms = []

    with open(args.threads, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 7:
                t_ids.append(f"T{p[0]}")
                t_edges.append(int(p[2]))
                t_active_ms.append(float(p[5]))
                t_barrier_ms.append(float(p[6]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

    # Subplot 1: Atomic CAS Failures vs Successes per Level
    x = np.arange(len(lvl_indices))
    width = 0.55

    success_k = np.array(lvl_cas_success) / 1000.0
    fails_k = np.array(lvl_cas_fails) / 1000.0

    ax1.bar(x, success_k, width, label="CAS Success (New Claims)", color="#2ca02c", alpha=0.9)
    ax1.bar(x, fails_k, width, bottom=success_k, label="CAS Failures (Redundant RFOs)", color="#d62728", alpha=0.9)

    ax1.set_xlabel("BFS Level Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Atomic CAS Operations ($10^3$ ops)", fontsize=11, fontweight="bold")
    ax1.set_title("Atomic CAS Contention per Level", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"L{i}" for i in lvl_indices])
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=10)

    # Annotate total fail percentage for key levels
    for i, (tot, f_count) in enumerate(zip(lvl_cas_attempts, lvl_cas_fails)):
        if tot > 1000:
            pct = 100.0 * f_count / tot
            top_y = (tot / 1000.0)
            ax1.annotate(f"{pct:.0f}% fails", (x[i], top_y), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=9, fontweight="bold", color="#d62728")

    # Subplot 2: Per-Thread Load Imbalance (Edges Scanned)
    tx = np.arange(len(t_ids))
    t_edges_k = np.array(t_edges) / 1000.0
    avg_edges_k = np.mean(t_edges_k)

    bars2 = ax2.bar(tx, t_edges_k, color="#1f77b4", width=0.6, alpha=0.85, label="Edges Scanned")
    ax2.axhline(avg_edges_k, color="#ff7f0e", linestyle="--", linewidth=2, label=f"Average ({avg_edges_k:.1f}k edges)")

    ax2.set_xlabel("OpenMP Thread ID", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Edges Scanned ($10^3$ edges)", fontsize=11, fontweight="bold")
    ax2.set_title("Per-Thread Edge Load Imbalance", fontsize=12, fontweight="bold")
    ax2.set_xticks(tx)
    ax2.set_xticklabels(t_ids)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=10)

    # Subplot 3: Per-Thread Time Breakdown (Active vs Barrier)
    ax3.bar(tx, t_active_ms, width=0.6, label="Active Edge Processing", color="#3b528b", alpha=0.9)
    ax3.bar(tx, t_barrier_ms, width=0.6, bottom=t_active_ms, label="Level Barrier Stall", color="#fde725", edgecolor="black", linewidth=0.5, alpha=0.9)

    ax3.set_xlabel("OpenMP Thread ID", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Execution Time (ms)", fontsize=11, fontweight="bold")
    ax3.set_title("Thread Execution Time Breakdown", fontsize=12, fontweight="bold")
    ax3.set_xticks(tx)
    ax3.set_xticklabels(t_ids)
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=10)

    plt.suptitle("Phase 3 Profiling: Empirical Bottleneck Diagnosis (Naive OpenMP BFS)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()

    print(f"Profiling bottleneck visualization saved to: {args.output}")

if __name__ == "__main__":
    main()
