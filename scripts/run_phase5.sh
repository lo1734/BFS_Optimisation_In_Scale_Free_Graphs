#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  Executing Phase 5 Verification & Benchmark Pipeline     "
echo "=========================================================="

echo "[1/4] Compiling all targets with make..."
make all

echo "[2/4] Running BFS correctness & oracle tests (validating Phase 5)..."
./bin/test_bfs

echo "[3/4] Running Phase 5 Comparative Benchmark (Padded vs Unpadded Ablation)..."
mkdir -p data
./bin/benchmark_bfs --input data/rmat_s16.bin --roots 16 --threads 1,2,4,8 --csv data/bfs_scaling_phase5.csv

echo "[4/4] Generating Phase 5 comparative scaling curve plot..."
python3 scripts/plot_scaling.py \
    --csv data/bfs_scaling_phase5.csv \
    --output data/scaling_curve_phase5.png \
    --title "Phase 5 Strong Scaling: False Sharing Elimination & Ablation"

echo "Phase 5 Pipeline executed successfully!"
