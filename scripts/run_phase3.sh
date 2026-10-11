#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "  Executing Phase 3 Verification & Profiling Pipeline     "
echo "=========================================================="

echo "[1/3] Compiling all targets with make..."
make all

echo "[2/3] Running Naive BFS Bottleneck Profiler (SCALE 16, 8 threads)..."
mkdir -p data
./bin/profile_bfs --input data/rmat_s16.bin --threads 8 \
    --csv-levels data/bfs_profile_levels.csv \
    --csv-threads data/bfs_profile_threads.csv

echo "[3/3] Generating bottleneck diagnostic plots..."
python3 scripts/plot_profiling.py \
    --levels data/bfs_profile_levels.csv \
    --threads data/bfs_profile_threads.csv \
    --output data/profiling_bottlenecks.png

echo "Phase 3 Pipeline executed successfully!"
