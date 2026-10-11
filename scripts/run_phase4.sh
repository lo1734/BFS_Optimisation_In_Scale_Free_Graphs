#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  Executing Phase 4 Verification & Benchmark Pipeline     "
echo "=========================================================="

echo "[1/5] Compiling all targets with make..."
make all

echo "[2/5] Running BFS correctness & oracle tests (validating Phase 4)..."
./bin/test_bfs

echo "[3/5] Profiling Phase 4 Contention-Reduced BFS (SCALE 16, 8 threads)..."
./bin/profile_bfs --input data/rmat_s16.bin --threads 8 --mode contention-reduced

echo "[4/5] Running Multi-Algorithm Comparative Benchmark (SCALE 16, 16 roots)..."
mkdir -p data
./bin/benchmark_bfs --input data/rmat_s16.bin --roots 16 --threads 1,2,4,8 --csv data/bfs_scaling_phase4.csv

echo "[5/5] Generating Phase 4 comparative scaling curve plot..."
python3 scripts/plot_scaling.py \
    --csv data/bfs_scaling_phase4.csv \
    --output data/scaling_curve_phase4.png \
    --title "Phase 4 Strong Scaling: Contention-Reduced vs Naive vs Serial"

echo "Phase 4 Pipeline executed successfully!"
