#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  Executing Phase 0 & Phase 1 Verification Pipeline       "
echo "=========================================================="

echo "[1/4] Compiling all targets with make..."
make all

echo "[2/4] Running unit and structural integrity tests..."
./bin/test_graph

echo "[3/4] Running Phase 0 atomic contention microbenchmark..."
./bin/atomic_microbench

echo "[4/4] Generating SCALE 16 R-MAT graph and plotting degree distribution..."
./bin/generate_graph --scale 16 --edge-factor 16 --output data/rmat_s16.bin --csv data/rmat_s16_deg.csv --validate
python3 scripts/plot_degree_dist.py --csv data/rmat_s16_deg.csv --output data/rmat_s16_power_law.png

echo "Pipeline executed successfully!"
