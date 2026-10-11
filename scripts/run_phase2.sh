#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  Executing Phase 2 Verification & Benchmark Pipeline     "
echo "=========================================================="

echo "[1/4] Compiling all targets with make..."
make all

echo "[2/4] Running BFS correctness & oracle tests..."
./bin/test_bfs

echo "[3/4] Running Baseline BFS Scaling Benchmark (SCALE 16, 16 roots)..."
mkdir -p data
./bin/benchmark_bfs --input data/rmat_s16.bin --roots 16 --threads 1,2,4,8 --csv data/bfs_scaling.csv

echo "[4/4] Generating BFS strong scaling plots..."
python3 scripts/plot_scaling.py --csv data/bfs_scaling.csv --output data/scaling_curve.png

echo "Phase 2 Pipeline executed successfully!"
