#include "rmat_generator.hpp"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <unordered_set>

namespace graph {

CSRGraph GraphGenerator::generate_rmat(const RMATParams& params) {
    const uint32_t scale = params.scale;
    const uint32_t num_vertices = 1U << scale;
    const uint64_t num_raw_edges = static_cast<uint64_t>(params.edge_factor) * num_vertices;

    const double a = params.a;
    const double b = params.b;
    const double c = params.c;
    const double ab = a + b;
    const double abc = a + b + c;

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<Edge>> thread_edges(num_threads);

    utils::Timer gen_timer;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Initialize thread-private RNG with distinct stream jumps
        utils::Xoshiro256StarStar rng(params.seed);
        for (int i = 0; i < tid; ++i) {
            rng.jump();
        }

        uint64_t edges_per_thread = (num_raw_edges + nthreads - 1) / nthreads;
        uint64_t start = tid * edges_per_thread;
        uint64_t end = std::min(start + edges_per_thread, num_raw_edges);

        auto& local_edges = thread_edges[tid];
        local_edges.reserve(end - start);

        for (uint64_t e = start; e < end; ++e) {
            uint32_t u = 0;
            uint32_t v = 0;

            for (uint32_t step = 0; step < scale; ++step) {
                double r = rng.next_double();
                uint32_t u_bit = 0;
                uint32_t v_bit = 0;

                if (r < a) {
                    u_bit = 0;
                    v_bit = 0;
                } else if (r < ab) {
                    u_bit = 0;
                    v_bit = 1;
                } else if (r < abc) {
                    u_bit = 1;
                    v_bit = 0;
                } else {
                    u_bit = 1;
                    v_bit = 1;
                }

                u = (u << 1) | u_bit;
                v = (v << 1) | v_bit;
            }

            // Discard self-loops immediately
            if (u != v) {
                local_edges.push_back({u, v});
            }
        }
    }

    // Merge thread edge vectors into single edge array
    std::vector<size_t> thread_offsets(num_threads + 1, 0);
    for (int t = 0; t < num_threads; ++t) {
        thread_offsets[t + 1] = thread_offsets[t] + thread_edges[t].size();
    }
    size_t total_valid_raw = thread_offsets[num_threads];

    std::vector<Edge> all_edges(total_valid_raw);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_threads; ++t) {
        if (!thread_edges[t].empty()) {
            std::copy(thread_edges[t].begin(), thread_edges[t].end(), 
                      all_edges.begin() + thread_offsets[t]);
            thread_edges[t].clear();
            thread_edges[t].shrink_to_fit();
        }
    }

    return build_csr_from_edges(num_vertices, all_edges);
}

CSRGraph GraphGenerator::generate_barabasi_albert(uint32_t num_vertices, uint32_t edges_per_node, uint64_t seed) {
    if (num_vertices <= edges_per_node) {
        throw std::invalid_argument("num_vertices must be greater than edges_per_node");
    }

    utils::Xoshiro256StarStar rng(seed);
    std::vector<Edge> edges;
    std::vector<uint32_t> repeated_nodes;

    uint32_t m0 = edges_per_node;
    // Initial complete clique of size m0
    for (uint32_t i = 0; i < m0; ++i) {
        for (uint32_t j = i + 1; j < m0; ++j) {
            edges.push_back({i, j});
            repeated_nodes.push_back(i);
            repeated_nodes.push_back(j);
        }
    }

    // Preferential attachment for nodes m0 ... num_vertices - 1
    std::unordered_set<uint32_t> targets;
    for (uint32_t i = m0; i < num_vertices; ++i) {
        targets.clear();
        while (targets.size() < edges_per_node) {
            size_t idx = rng.next() % repeated_nodes.size();
            uint32_t v = repeated_nodes[idx];
            if (v != i) {
                targets.insert(v);
            }
        }

        for (uint32_t v : targets) {
            edges.push_back({i, v});
            repeated_nodes.push_back(i);
            repeated_nodes.push_back(v);
        }
    }

    return build_csr_from_edges(num_vertices, edges);
}

CSRGraph GraphGenerator::build_csr_from_edges(uint32_t num_vertices, 
                                              const std::vector<Edge>& raw_edges) {
    const size_t num_raw = raw_edges.size();
    const int num_threads = omp_get_max_threads();

    // Symmetrization: We have 2 * num_raw directed edges: (u, v) and (v, u).
    // To construct CSR without atomic contention on hub vertices, we use a 2D parallel
    // counting sort over vertices:
    // 1. Each thread counts degrees for its slice of directed edges into a thread-private histogram.
    // 2. We sum histograms across threads to obtain raw degree and global offsets per vertex.
    // 3. Threads scatter directed edges into contiguous per-vertex buckets with ZERO atomics.
    // 4. Each vertex's neighbors are sorted and deduplicated in parallel.
    // 5. Final 128-byte aligned CSR arrays (row_ptr, col_idx) are populated.

    // Step 1: Thread-private degree histograms
    // Shape: [num_threads][num_vertices]
    std::vector<std::vector<uint32_t>> thread_deg(num_threads, std::vector<uint32_t>(num_vertices, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        auto& local_hist = thread_deg[tid];

        size_t chunk = (num_raw + nthreads - 1) / nthreads;
        size_t start = tid * chunk;
        size_t end = std::min(start + chunk, num_raw);

        for (size_t i = start; i < end; ++i) {
            uint32_t u = raw_edges[i].u;
            uint32_t v = raw_edges[i].v;
            if (u < num_vertices && v < num_vertices && u != v) {
                local_hist[u]++;
                local_hist[v]++;
            }
        }
    }

    // Step 2: Sum histograms across threads to determine raw degrees and scatter offsets
    std::vector<uint64_t> raw_row_ptr(num_vertices + 1, 0);

    #pragma omp parallel for schedule(static)
    for (uint32_t u = 0; u < num_vertices; ++u) {
        uint64_t deg_u = 0;
        for (int t = 0; t < num_threads; ++t) {
            deg_u += thread_deg[t][u];
        }
        raw_row_ptr[u + 1] = deg_u;
    }

    // Prefix sum over raw degrees
    for (uint32_t u = 0; u < num_vertices; ++u) {
        raw_row_ptr[u + 1] += raw_row_ptr[u];
    }
    uint64_t total_directed_raw = raw_row_ptr[num_vertices];

    // Compute scatter offsets for each thread:
    // scatter_offsets[t][u] = raw_row_ptr[u] + sum_{i=0}^{t-1} thread_deg[i][u]
    std::vector<std::vector<uint64_t>> scatter_offset(num_threads, std::vector<uint64_t>(num_vertices));
    #pragma omp parallel for schedule(static)
    for (uint32_t u = 0; u < num_vertices; ++u) {
        uint64_t curr = raw_row_ptr[u];
        for (int t = 0; t < num_threads; ++t) {
            scatter_offset[t][u] = curr;
            curr += thread_deg[t][u];
        }
    }

    // Step 3: Parallel scatter of directed edges into raw_col array
    std::vector<uint32_t> raw_col(total_directed_raw);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        auto& my_scatter = scatter_offset[tid];

        size_t chunk = (num_raw + nthreads - 1) / nthreads;
        size_t start = tid * chunk;
        size_t end = std::min(start + chunk, num_raw);

        for (size_t i = start; i < end; ++i) {
            uint32_t u = raw_edges[i].u;
            uint32_t v = raw_edges[i].v;
            if (u < num_vertices && v < num_vertices && u != v) {
                raw_col[my_scatter[u]++] = v;
                raw_col[my_scatter[v]++] = u;
            }
        }
    }

    // Free intermediate histogram buffers to conserve memory
    thread_deg.clear();
    thread_deg.shrink_to_fit();
    scatter_offset.clear();
    scatter_offset.shrink_to_fit();

    // Step 4: In parallel, sort and deduplicate neighbors of each vertex
    std::vector<uint32_t> true_degrees(num_vertices, 0);

    #pragma omp parallel for schedule(dynamic, 512)
    for (uint32_t u = 0; u < num_vertices; ++u) {
        uint64_t start_idx = raw_row_ptr[u];
        uint64_t end_idx = raw_row_ptr[u + 1];
        if (start_idx < end_idx) {
            auto begin_it = raw_col.begin() + start_idx;
            auto end_it = raw_col.begin() + end_idx;

            std::sort(begin_it, end_it);
            auto new_end = std::unique(begin_it, end_it);
            true_degrees[u] = static_cast<uint32_t>(std::distance(begin_it, new_end));
        } else {
            true_degrees[u] = 0;
        }
    }

    // Step 5: Allocate final 128-byte aligned CSRGraph
    CSRGraph csr;
    std::vector<uint64_t> final_row_ptr(num_vertices + 1, 0);
    for (uint32_t u = 0; u < num_vertices; ++u) {
        final_row_ptr[u + 1] = final_row_ptr[u] + true_degrees[u];
    }
    uint64_t final_edges = final_row_ptr[num_vertices];

    csr.allocate(num_vertices, final_edges);

    // Copy row_ptr
    #pragma omp parallel for schedule(static)
    for (uint32_t u = 0; u <= num_vertices; ++u) {
        csr.row_ptr[u] = final_row_ptr[u];
    }

    // Copy deduplicated col_idx
    #pragma omp parallel for schedule(dynamic, 512)
    for (uint32_t u = 0; u < num_vertices; ++u) {
        uint32_t deg = true_degrees[u];
        if (deg > 0) {
            uint64_t src_start = raw_row_ptr[u];
            uint64_t dst_start = csr.row_ptr[u];
            std::copy(raw_col.begin() + src_start, 
                      raw_col.begin() + src_start + deg, 
                      csr.col_idx + dst_start);
        }
    }

    return csr;
}

} // namespace graph
