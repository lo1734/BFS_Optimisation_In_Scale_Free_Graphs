#!/usr/bin/env python3
"""
plot_degree_dist.py
Plots degree distribution on a log-log scale to verify power-law scaling
and estimates the power-law exponent gamma: P(k) ~ k^(-gamma).
"""

import os
import sys
import argparse
import numpy as np

# Ensure matplotlib uses a writable temp cache directory to avoid macOS sandbox warnings
os.environ["MPLCONFIGDIR"] = os.path.expanduser("/tmp/matplotlib")

import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot degree distribution log-log curve.")
    parser.add_argument("--csv", required=True, help="Input CSV path (degree,frequency,probability)")
    parser.add_argument("--output", default="data/degree_distribution.png", help="Output PNG path")
    parser.add_argument("--title", default="Degree Distribution (Log-Log Scale)", help="Plot title")
    args = parser.parse_args()

    degrees = []
    frequencies = []
    probabilities = []

    with open(args.csv, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                deg = int(parts[0])
                freq = int(parts[1])
                prob = float(parts[2])
                if deg > 0 and freq > 0:
                    degrees.append(deg)
                    frequencies.append(freq)
                    probabilities.append(prob)

    if not degrees:
        print("Error: No valid non-zero degree data found in CSV.")
        sys.exit(1)

    degrees = np.array(degrees)
    probabilities = np.array(probabilities)

    # Sort by degree
    sort_idx = np.argsort(degrees)
    degrees = degrees[sort_idx]
    probabilities = probabilities[sort_idx]

    # Power-law linear regression in log-log space
    # Restrict fit to intermediate degrees (avoid noise at the extreme tail with freq=1)
    mask = (degrees >= 2) & (probabilities > 0)
    log_deg = np.log10(degrees[mask])
    log_prob = np.log10(probabilities[mask])

    slope, intercept = np.polyfit(log_deg, log_prob, 1)
    gamma = -slope
    correlation = np.corrcoef(log_deg, log_prob)[0, 1]
    r_squared = correlation ** 2

    # Plot
    plt.figure(figsize=(9, 6), dpi=150)
    plt.loglog(degrees, probabilities, "o", color="#1f77b4", alpha=0.7, markersize=5, label="Empirical $P(k)$")

    fit_k = np.linspace(min(degrees[mask]), max(degrees[mask]), 100)
    fit_p = (10 ** intercept) * (fit_k ** slope)
    plt.loglog(fit_k, fit_p, "--", color="#d62728", linewidth=2, 
               label=f"Power-Law Fit $P(k) \\propto k^{{-{gamma:.2f}}}$ ($R^2 = {r_squared:.2f}$)")

    plt.title(args.title, fontsize=14, fontweight="bold")
    plt.xlabel("Degree $k$", fontsize=12)
    plt.ylabel("Probability $P(k)$", fontsize=12)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output)
    plt.close()

    print(f"Degree distribution plot saved to: {args.output}")
    print(f"Estimated Power-Law Exponent gamma: {gamma:.2f}")
    print(f"Goodness of Fit R^2:                {r_squared:.4f}")
    print(f"Max Degree:                         {np.max(degrees)}")

if __name__ == "__main__":
    main()
