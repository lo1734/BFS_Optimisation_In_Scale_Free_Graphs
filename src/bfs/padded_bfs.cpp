#include "bfs.hpp"
#include <atomic>
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <vector>

namespace bfs {

// ============================================================================
// 1. PADDED & CACHE-LINE ALIGNED BFS IMPLEMENTATION (PHASE 5)
// ============================================================================

// 128-byte cache-line aligned and padded per-thread structure
// Guarantees zero false sharing on both Apple Silicon (128-byte lines) and x86-64 (64-byte lines)
struct alignas(CACHE_LINE_SIZE) PaddedThreadState {
    uint32_t* buf = nullptr;
    size_t size = 0;
    size_t capacity = 0;
    uint64_t edges_scanned = 0;
    uint64_t vertices_scanned = 0;
    char pad[CACHE_LINE_SIZE - (sizeof(uint32_t*) + 2 * sizeof(size_t) + 2 * sizeof(uint64_t)) % CACHE_LINE_SIZE];

    void allocate(size_t cap) {
        buf = utils::aligned_malloc<uint32_t>(cap, CACHE_LINE_SIZE);
        capacity = cap;
        size = 0;
    }

    void deallocate() {
        if (buf) {
            utils::aligned_free(buf);
            buf = nullptr;
        }
        size = 0;
        capacity = 0;
    }

    inline void push(uint32_t v) {
        buf[size++] = v;
    }
};

BFSResult run_padded_bfs(const graph::CSRGraph& g, uint32_t root, int num_threads, bool track_parents) {
    uint32_t n = g.num_vertices;
    BFSResult result;
    result.allocate(n, track_parents);
    result.root = root;

    if (root >= n) {
        return result;
    }

    // Parallel static initialization across 128-byte cache lines
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (uint32_t i = 0; i < n; ++i) {
        result.dist[i] = -1;
        if (track_parents) {
            result.parent[i] = UINT32_MAX;
        }
    }

    // Double-buffered global frontiers with 128-byte alignment
    uint32_t* frontier = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);
    uint32_t* next_frontier = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);

    // Provision 128-byte aligned and padded per-thread state structures
    std::vector<PaddedThreadState> tls(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        tls[t].allocate(n);
    }

    result.dist[root] = 0;
    if (track_parents) {
        result.parent[root] = root;
    }

    frontier[0] = root;
    size_t frontier_size = 1;
    int32_t current_level = 0;
    uint32_t total_visited = 1;

    // Distinct cache-line padded array for level prefix sums
    std::vector<size_t> offsets(num_threads + 1, 0);

    utils::Timer timer;

    while (frontier_size > 0) {
        // Read-mostly replicated current_level variable
        const int32_t level = current_level;

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            PaddedThreadState& me = tls[tid];
            me.size = 0;

            #pragma omp for schedule(dynamic, 64) nowait
            for (size_t i = 0; i < frontier_size; ++i) {
                uint32_t u = frontier[i];
                uint64_t start_e = g.row_ptr[u];
                uint64_t end_e = g.row_ptr[u + 1];
                me.vertices_scanned++;
                me.edges_scanned += (end_e - start_e);

                for (uint64_t e = start_e; e < end_e; ++e) {
                    uint32_t v = g.col_idx[e];

                    // 1. Test-before-CAS keeps cache line in Shared MESI state
                    if (result.dist[v] != -1) continue;

                    // 2. Relaxed atomic CAS
                    int32_t expected = -1;
                    if (__atomic_compare_exchange_n(&result.dist[v], &expected, level + 1,
                                                    false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                        if (track_parents) {
                            result.parent[v] = u;
                        }
                        // 3. Thread-local buffer append (zero false sharing, zero atomics)
                        me.push(v);
                    }
                }
            }
        } // Implicit barrier guarantees memory visibility

        // Batched prefix sum
        offsets[0] = 0;
        for (int t = 0; t < num_threads; ++t) {
            offsets[t + 1] = offsets[t] + tls[t].size;
        }

        frontier_size = offsets[num_threads];
        total_visited += frontier_size;
        current_level++;

        // Cache-line aligned parallel scatter copy
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < num_threads; ++t) {
            if (tls[t].size > 0) {
                std::memcpy(&next_frontier[offsets[t]], tls[t].buf, tls[t].size * sizeof(uint32_t));
            }
        }

        std::swap(frontier, next_frontier);
    }

    result.elapsed_seconds = timer.elapsed_seconds();
    result.traversed_vertices = total_visited;
    result.num_levels = current_level;

    // Reduction over reachable edges
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

    for (int t = 0; t < num_threads; ++t) {
        tls[t].deallocate();
    }
    utils::aligned_free(frontier);
    utils::aligned_free(next_frontier);

    return result;
}

// ============================================================================
// 2. UNPADDED BFS IMPLEMENTATION (PHASE 5 FALSE-SHARING ABLATION CONTROL)
// ============================================================================

// Tightly packed structure without cache-line padding (sizeof = 40 bytes)
// Multiple instances share the same 128-byte cache line, causing false sharing
// as threads continuously write to size and edges_scanned in the inner loop.
struct UnpaddedThreadState {
    uint32_t* buf = nullptr;
    size_t size = 0;
    size_t capacity = 0;
    uint64_t edges_scanned = 0;
    uint64_t vertices_scanned = 0;

    inline void push(uint32_t v) {
        buf[size++] = v;
    }
};

BFSResult run_unpadded_bfs(const graph::CSRGraph& g, uint32_t root, int num_threads, bool track_parents) {
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

    uint32_t* frontier = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);
    uint32_t* next_frontier = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);

    // Contiguous unpadded allocation of thread state array
    // (Adjacent thread structs tightly packed within same cache lines)
    std::vector<UnpaddedThreadState> unpadded_tls(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        unpadded_tls[t].buf = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);
        unpadded_tls[t].capacity = n;
        unpadded_tls[t].size = 0;
    }

    result.dist[root] = 0;
    if (track_parents) {
        result.parent[root] = root;
    }

    frontier[0] = root;
    size_t frontier_size = 1;
    int32_t current_level = 0;
    uint32_t total_visited = 1;

    std::vector<size_t> offsets(num_threads + 1, 0);

    utils::Timer timer;

    while (frontier_size > 0) {
        const int32_t level = current_level;

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            UnpaddedThreadState& me = unpadded_tls[tid];
            me.size = 0;

            #pragma omp for schedule(dynamic, 64) nowait
            for (size_t i = 0; i < frontier_size; ++i) {
                uint32_t u = frontier[i];
                uint64_t start_e = g.row_ptr[u];
                uint64_t end_e = g.row_ptr[u + 1];

                // False sharing hot write: modifying edges_scanned and vertices_scanned
                // invalidates the shared cache line holding adjacent threads' counters
                me.vertices_scanned++;
                me.edges_scanned += (end_e - start_e);

                for (uint64_t e = start_e; e < end_e; ++e) {
                    uint32_t v = g.col_idx[e];

                    if (result.dist[v] != -1) continue;

                    int32_t expected = -1;
                    if (__atomic_compare_exchange_n(&result.dist[v], &expected, level + 1,
                                                    false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                        if (track_parents) {
                            result.parent[v] = u;
                        }
                        // Modifying me.size also invalidates adjacent threads' structs
                        me.push(v);
                    }
                }
            }
        }

        offsets[0] = 0;
        for (int t = 0; t < num_threads; ++t) {
            offsets[t + 1] = offsets[t] + unpadded_tls[t].size;
        }

        frontier_size = offsets[num_threads];
        total_visited += frontier_size;
        current_level++;

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (int t = 0; t < num_threads; ++t) {
            if (unpadded_tls[t].size > 0) {
                std::memcpy(&next_frontier[offsets[t]], unpadded_tls[t].buf, unpadded_tls[t].size * sizeof(uint32_t));
            }
        }

        std::swap(frontier, next_frontier);
    }

    result.elapsed_seconds = timer.elapsed_seconds();
    result.traversed_vertices = total_visited;
    result.num_levels = current_level;

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

    for (int t = 0; t < num_threads; ++t) {
        if (unpadded_tls[t].buf) {
            utils::aligned_free(unpadded_tls[t].buf);
        }
    }
    utils::aligned_free(frontier);
    utils::aligned_free(next_frontier);

    return result;
}

} // namespace bfs
