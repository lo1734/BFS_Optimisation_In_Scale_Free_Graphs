#include "bfs.hpp"
#include <atomic>
#include <omp.h>
#include <algorithm>
#include <iostream>

namespace bfs {

BFSResult run_naive_parallel_bfs(const graph::CSRGraph& g, uint32_t root, int num_threads, bool track_parents) {
    uint32_t n = g.num_vertices;
    BFSResult result;
    result.allocate(n, track_parents);
    result.root = root;

    if (root >= n) {
        return result;
    }

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (uint32_t i = 0; i < n; ++i) {
        result.dist[i] = -1;
        if (track_parents) {
            result.parent[i] = UINT32_MAX;
        }
    }

    // Allocate double-buffered frontier arrays
    uint32_t* frontier = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);
    uint32_t* next_frontier = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);

    result.dist[root] = 0;
    if (track_parents) {
        result.parent[root] = root;
    }

    frontier[0] = root;
    size_t frontier_size = 1;
    int32_t current_level = 0;
    uint32_t total_visited = 1;

    // Contended shared global queue tail counter
    std::atomic<uint64_t> next_frontier_tail{0};

    utils::Timer timer;

    while (frontier_size > 0) {
        next_frontier_tail.store(0, std::memory_order_relaxed);

        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 64)
        for (size_t i = 0; i < frontier_size; ++i) {
            uint32_t u = frontier[i];
            uint64_t start_e = g.row_ptr[u];
            uint64_t end_e = g.row_ptr[u + 1];

            for (uint64_t e = start_e; e < end_e; ++e) {
                uint32_t v = g.col_idx[e];

                // Naive approach: Direct atomic CAS on dist[v] with sequential consistency
                // (no test-before-CAS; all threads ping-pong the cache line containing dist[v])
                int32_t expected = -1;
                if (__atomic_compare_exchange_n(&result.dist[v], &expected, current_level + 1,
                                                false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
                    if (track_parents) {
                        result.parent[v] = u;
                    }

                    // Contended atomic append to shared global frontier tail
                    uint64_t idx = next_frontier_tail.fetch_add(1, std::memory_order_seq_cst);
                    next_frontier[idx] = v;
                }
            }
        } // Implicit OpenMP barrier

        frontier_size = next_frontier_tail.load(std::memory_order_relaxed);
        total_visited += frontier_size;
        current_level++;

        // Swap frontier buffers for next level
        std::swap(frontier, next_frontier);
    }

    result.elapsed_seconds = timer.elapsed_seconds();
    result.traversed_vertices = total_visited;
    result.num_levels = current_level;

    // Calculate reachable edges
    uint64_t deg_sum = 0;
    #pragma omp parallel for num_threads(num_threads) reduction(+:deg_sum) schedule(static)
    for (uint32_t u = 0; u < n; ++u) {
        if (result.dist[u] >= 0) {
            deg_sum += g.degree(u);
        }
    }
    result.traversed_edges = deg_sum / 2;

    if (result.elapsed_seconds > 0.0) {
        result.teps = static_cast<double>(result.traversed_edges) / result.elapsed_seconds;
    } else {
        result.teps = 0.0;
    }

    utils::aligned_free(frontier);
    utils::aligned_free(next_frontier);

    return result;
}

} // namespace bfs
