#include "bfs.hpp"
#include "graph.hpp"
#include "rmat_generator.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <atomic>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <numeric>

// 128-byte cache-line padded per-thread profiling metrics to eliminate probe effect
struct alignas(CACHE_LINE_SIZE) ThreadMetrics {
    uint64_t vertices_scanned = 0;
    uint64_t edges_scanned = 0;
    uint64_t cas_attempts = 0;
    uint64_t cas_successes = 0;
    uint64_t cas_failures = 0;
    uint64_t queue_pushes = 0;
    double active_time_sec = 0.0;
    double barrier_time_sec = 0.0;
    char pad[CACHE_LINE_SIZE - (6 * sizeof(uint64_t) + 2 * sizeof(double)) % CACHE_LINE_SIZE];
};

struct LevelMetrics {
    int32_t level = 0;
    size_t frontier_size = 0;
    uint64_t edges_scanned = 0;
    uint64_t cas_attempts = 0;
    uint64_t cas_successes = 0;
    uint64_t cas_failures = 0;
    uint64_t queue_pushes = 0;
    double elapsed_sec = 0.0;
    double max_active_sec = 0.0;
    double avg_barrier_sec = 0.0;
    double load_imbalance = 1.0;
};

struct HubContention {
    uint32_t vertex = 0;
    uint64_t degree = 0;
    uint64_t cas_attempts = 0;
    uint64_t cas_failures = 0;
};

void run_profiled_bfs(
    const graph::CSRGraph& g,
    uint32_t root,
    int num_threads,
    bool contention_reduced,
    std::vector<LevelMetrics>& level_stats,
    std::vector<ThreadMetrics>& total_thread_stats,
    std::vector<HubContention>& hub_stats,
    double& total_elapsed_sec)
{
    uint32_t n = g.num_vertices;
    int32_t* dist = utils::aligned_malloc<int32_t>(n, CACHE_LINE_SIZE);
    uint32_t* frontier = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);
    uint32_t* next_frontier = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (uint32_t i = 0; i < n; ++i) {
        dist[i] = -1;
    }

    dist[root] = 0;
    frontier[0] = root;
    size_t frontier_size = 1;
    int32_t current_level = 0;

    std::atomic<uint64_t> next_frontier_tail{0};

    total_thread_stats.assign(num_threads, ThreadMetrics{});

    // Thread-local buffers for contention-reduced mode
    struct alignas(CACHE_LINE_SIZE) LocalBuf {
        std::vector<uint32_t> items;
    };
    std::vector<LocalBuf> tls_frontier_bufs(num_threads);
    if (contention_reduced) {
        for (int t = 0; t < num_threads; ++t) {
            tls_frontier_bufs[t].items.reserve(n / num_threads + 128);
        }
    }

    // Identify top-10 hub vertices for contention tracking
    std::vector<std::pair<uint64_t, uint32_t>> degree_pairs;
    degree_pairs.reserve(n);
    for (uint32_t u = 0; u < n; ++u) {
        degree_pairs.emplace_back(g.degree(u), u);
    }
    std::sort(degree_pairs.rbegin(), degree_pairs.rend());

    size_t num_hubs = std::min(size_t(10), degree_pairs.size());
    hub_stats.resize(num_hubs);
    std::vector<uint32_t> hub_lookup(n, UINT32_MAX);
    for (size_t i = 0; i < num_hubs; ++i) {
        hub_stats[i].vertex = degree_pairs[i].second;
        hub_stats[i].degree = degree_pairs[i].first;
        hub_lookup[degree_pairs[i].second] = static_cast<uint32_t>(i);
    }

    // Thread-private hub counter buffers (128-byte padded per thread)
    struct alignas(CACHE_LINE_SIZE) ThreadHubBuffer {
        std::vector<uint64_t> attempts;
        std::vector<uint64_t> failures;
    };
    std::vector<ThreadHubBuffer> thread_hubs(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        thread_hubs[t].attempts.assign(num_hubs, 0);
        thread_hubs[t].failures.assign(num_hubs, 0);
    }

    auto start_all = std::chrono::high_resolution_clock::now();

    while (frontier_size > 0) {
        next_frontier_tail.store(0, std::memory_order_relaxed);
        if (contention_reduced) {
            for (int t = 0; t < num_threads; ++t) {
                tls_frontier_bufs[t].items.clear();
            }
        }

        std::vector<ThreadMetrics> lvl_thread_metrics(num_threads);

        auto lvl_start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            ThreadMetrics& me = lvl_thread_metrics[tid];

            auto t_work_start = std::chrono::high_resolution_clock::now();

            #pragma omp for schedule(dynamic, 64) nowait
            for (size_t i = 0; i < frontier_size; ++i) {
                uint32_t u = frontier[i];
                me.vertices_scanned++;

                uint64_t start_e = g.row_ptr[u];
                uint64_t end_e = g.row_ptr[u + 1];
                me.edges_scanned += (end_e - start_e);

                for (uint64_t e = start_e; e < end_e; ++e) {
                    uint32_t v = g.col_idx[e];
                    uint32_t hub_idx = hub_lookup[v];

                    if (contention_reduced) {
                        // 1. Test-before-CAS: Plain relaxed read keeps line in Shared state
                        if (dist[v] != -1) continue;

                        me.cas_attempts++;
                        if (hub_idx != UINT32_MAX) {
                            thread_hubs[tid].attempts[hub_idx]++;
                        }

                        // 2. Atomic CAS with relaxed memory order
                        int32_t expected = -1;
                        if (__atomic_compare_exchange_n(&dist[v], &expected, current_level + 1,
                                                        false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                            me.cas_successes++;
                            // 3. Thread-local push: 0 queue tail atomics
                            tls_frontier_bufs[tid].items.push_back(v);
                        } else {
                            me.cas_failures++;
                            if (hub_idx != UINT32_MAX) {
                                thread_hubs[tid].failures[hub_idx]++;
                            }
                        }
                    } else {
                        // Naive atomic CAS on dist[v]
                        me.cas_attempts++;
                        if (hub_idx != UINT32_MAX) {
                            thread_hubs[tid].attempts[hub_idx]++;
                        }

                        int32_t expected = -1;
                        if (__atomic_compare_exchange_n(&dist[v], &expected, current_level + 1,
                                                        false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
                            me.cas_successes++;
                            me.queue_pushes++;
                            uint64_t idx = next_frontier_tail.fetch_add(1, std::memory_order_seq_cst);
                            next_frontier[idx] = v;
                        } else {
                            me.cas_failures++;
                            if (hub_idx != UINT32_MAX) {
                                thread_hubs[tid].failures[hub_idx]++;
                            }
                        }
                    }
                }
            }

            auto t_work_end = std::chrono::high_resolution_clock::now();
            me.active_time_sec = std::chrono::duration<double>(t_work_end - t_work_start).count();

            // Explicit level barrier to record barrier stall time
            #pragma omp barrier

            auto t_barrier_end = std::chrono::high_resolution_clock::now();
            me.barrier_time_sec = std::chrono::duration<double>(t_barrier_end - t_work_end).count();
        }

        auto lvl_end = std::chrono::high_resolution_clock::now();
        double lvl_elapsed = std::chrono::duration<double>(lvl_end - lvl_start).count();

        // Aggregate level metrics
        LevelMetrics lm;
        lm.level = current_level;
        lm.frontier_size = frontier_size;
        lm.elapsed_sec = lvl_elapsed;

        uint64_t max_edges = 0;
        uint64_t total_lvl_edges = 0;
        double max_active = 0.0;
        double sum_barrier = 0.0;

        for (int t = 0; t < num_threads; ++t) {
            lm.edges_scanned += lvl_thread_metrics[t].edges_scanned;
            lm.cas_attempts += lvl_thread_metrics[t].cas_attempts;
            lm.cas_successes += lvl_thread_metrics[t].cas_successes;
            lm.cas_failures += lvl_thread_metrics[t].cas_failures;
            lm.queue_pushes += lvl_thread_metrics[t].queue_pushes;

            max_edges = std::max(max_edges, lvl_thread_metrics[t].edges_scanned);
            total_lvl_edges += lvl_thread_metrics[t].edges_scanned;
            max_active = std::max(max_active, lvl_thread_metrics[t].active_time_sec);
            sum_barrier += lvl_thread_metrics[t].barrier_time_sec;

            // Accumulate to total thread stats
            total_thread_stats[t].vertices_scanned += lvl_thread_metrics[t].vertices_scanned;
            total_thread_stats[t].edges_scanned += lvl_thread_metrics[t].edges_scanned;
            total_thread_stats[t].cas_attempts += lvl_thread_metrics[t].cas_attempts;
            total_thread_stats[t].cas_successes += lvl_thread_metrics[t].cas_successes;
            total_thread_stats[t].cas_failures += lvl_thread_metrics[t].cas_failures;
            total_thread_stats[t].queue_pushes += lvl_thread_metrics[t].queue_pushes;
            total_thread_stats[t].active_time_sec += lvl_thread_metrics[t].active_time_sec;
            total_thread_stats[t].barrier_time_sec += lvl_thread_metrics[t].barrier_time_sec;
        }

        lm.max_active_sec = max_active;
        lm.avg_barrier_sec = sum_barrier / num_threads;

        double avg_edges = (num_threads > 0) ? (static_cast<double>(total_lvl_edges) / num_threads) : 1.0;
        lm.load_imbalance = (avg_edges > 0) ? (static_cast<double>(max_edges) / avg_edges) : 1.0;

        level_stats.push_back(lm);

        if (contention_reduced) {
            std::vector<size_t> offsets(num_threads + 1, 0);
            for (int t = 0; t < num_threads; ++t) {
                offsets[t + 1] = offsets[t] + tls_frontier_bufs[t].items.size();
            }
            frontier_size = offsets[num_threads];
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int t = 0; t < num_threads; ++t) {
                if (!tls_frontier_bufs[t].items.empty()) {
                    std::memcpy(&next_frontier[offsets[t]], tls_frontier_bufs[t].items.data(),
                                tls_frontier_bufs[t].items.size() * sizeof(uint32_t));
                }
            }
        } else {
            frontier_size = next_frontier_tail.load(std::memory_order_relaxed);
        }

        current_level++;
        std::swap(frontier, next_frontier);
    }

    auto end_all = std::chrono::high_resolution_clock::now();
    total_elapsed_sec = std::chrono::duration<double>(end_all - start_all).count();

    // Aggregate hub contention metrics
    for (size_t i = 0; i < num_hubs; ++i) {
        for (int t = 0; t < num_threads; ++t) {
            hub_stats[i].cas_attempts += thread_hubs[t].attempts[i];
            hub_stats[i].cas_failures += thread_hubs[t].failures[i];
        }
    }

    utils::aligned_free(dist);
    utils::aligned_free(frontier);
    utils::aligned_free(next_frontier);
}

void print_level_report(const std::vector<LevelMetrics>& levels) {
    std::cout << "\n" << std::string(105, '=') << "\n";
    std::cout << "                               LEVEL-BY-LEVEL BOTTLENECK PROFILE\n";
    std::cout << std::string(105, '=') << "\n";
    std::cout << std::left
              << std::setw(8)  << "Level"
              << std::setw(12) << "Frontier"
              << std::setw(14) << "Edges Scanned"
              << std::setw(14) << "CAS Attempts"
              << std::setw(14) << "CAS Failures"
              << std::setw(14) << "Fail Ratio"
              << std::setw(15) << "Active (ms)"
              << std::setw(14) << "Barrier (ms)"
              << "\n";
    std::cout << std::string(105, '-') << "\n";

    for (const auto& l : levels) {
        double fail_pct = (l.cas_attempts > 0) ? (100.0 * l.cas_failures / l.cas_attempts) : 0.0;
        std::cout << std::left
                  << std::setw(8)  << l.level
                  << std::setw(12) << l.frontier_size
                  << std::setw(14) << l.edges_scanned
                  << std::setw(14) << l.cas_attempts
                  << std::setw(14) << l.cas_failures
                  << std::fixed << std::setprecision(1)
                  << std::setw(14) << (std::to_string(static_cast<int>(fail_pct)) + "%")
                  << std::setprecision(3)
                  << std::setw(15) << (l.max_active_sec * 1000.0)
                  << std::setw(14) << (l.avg_barrier_sec * 1000.0)
                  << "\n";
    }
    std::cout << std::string(105, '=') << "\n\n";
}

void print_thread_report(const std::vector<ThreadMetrics>& threads) {
    std::cout << std::string(105, '=') << "\n";
    std::cout << "                               PER-THREAD LOAD & CONTENTION PROFILE\n";
    std::cout << std::string(105, '=') << "\n";
    std::cout << std::left
              << std::setw(10) << "Thread"
              << std::setw(14) << "Vertices"
              << std::setw(16) << "Edges Scanned"
              << std::setw(14) << "CAS Attempts"
              << std::setw(14) << "Queue Pushes"
              << std::setw(16) << "Active Time (ms)"
              << std::setw(16) << "Barrier Time (ms)"
              << "\n";
    std::cout << std::string(105, '-') << "\n";

    for (size_t t = 0; t < threads.size(); ++t) {
        const auto& m = threads[t];
        std::cout << std::left
                  << std::setw(10) << ("T" + std::to_string(t))
                  << std::setw(14) << m.vertices_scanned
                  << std::setw(16) << m.edges_scanned
                  << std::setw(14) << m.cas_attempts
                  << std::setw(14) << m.queue_pushes
                  << std::fixed << std::setprecision(3)
                  << std::setw(16) << (m.active_time_sec * 1000.0)
                  << std::setw(16) << (m.barrier_time_sec * 1000.0)
                  << "\n";
    }
    std::cout << std::string(105, '=') << "\n\n";
}

void print_hub_report(const std::vector<HubContention>& hubs) {
    std::cout << std::string(80, '=') << "\n";
    std::cout << "                 TOP HUB VERTEX CONTENTION & COLLISION ANALYSIS\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::left
              << std::setw(12) << "Rank"
              << std::setw(14) << "Vertex ID"
              << std::setw(14) << "Degree"
              << std::setw(18) << "CAS Attempts"
              << std::setw(18) << "CAS Collisions"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t i = 0; i < hubs.size(); ++i) {
        std::cout << std::left
                  << std::setw(12) << ("#" + std::to_string(i + 1))
                  << std::setw(14) << hubs[i].vertex
                  << std::setw(14) << hubs[i].degree
                  << std::setw(18) << hubs[i].cas_attempts
                  << std::setw(18) << hubs[i].cas_failures
                  << "\n";
    }
    std::cout << std::string(80, '=') << "\n\n";
}

int main(int argc, char* argv[]) {
    std::string input_path = "";
    uint32_t scale = 16;
    int num_threads = 8;
    std::string csv_levels = "";
    std::string csv_threads = "";
    uint32_t custom_root = UINT32_MAX;
    bool contention_reduced = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            scale = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--root") == 0 && i + 1 < argc) {
            custom_root = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--csv-levels") == 0 && i + 1 < argc) {
            csv_levels = argv[++i];
        } else if (std::strcmp(argv[i], "--csv-threads") == 0 && i + 1 < argc) {
            csv_threads = argv[++i];
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "contention-reduced" || m == "cr") {
                contention_reduced = true;
            }
        }
    }

    graph::CSRGraph g;
    if (!input_path.empty()) {
        std::cout << "Loading graph from: " << input_path << " ...\n";
        g = graph::CSRGraph::load_binary(input_path);
    } else {
        std::cout << "Generating synthetic R-MAT graph (SCALE " << scale << ") ...\n";
        graph::RMATParams params;
        params.scale = scale;
        params.edge_factor = 16;
        g = graph::GraphGenerator::generate_rmat(params);
    }

    g.print_summary();

    uint32_t root = custom_root;
    if (root == UINT32_MAX || root >= g.num_vertices) {
        auto roots = bfs::select_random_roots(g, 1, 9999);
        root = roots[0];
    }

    std::string mode_str = contention_reduced ? "Contention-Reduced OpenMP BFS (Phase 4)" : "Naive OpenMP BFS";
    std::cout << "Profiling " << mode_str << " with " << num_threads << " threads from root: " << root << " ...\n";

    std::vector<LevelMetrics> level_stats;
    std::vector<ThreadMetrics> thread_stats;
    std::vector<HubContention> hub_stats;
    double total_time = 0.0;

    run_profiled_bfs(g, root, num_threads, contention_reduced, level_stats, thread_stats, hub_stats, total_time);

    // Compute totals
    uint64_t total_edges = 0;
    uint64_t total_cas = 0;
    uint64_t total_cas_fails = 0;
    uint64_t total_queue_pushes = 0;
    double total_active_time = 0.0;
    double total_barrier_time = 0.0;

    for (const auto& l : level_stats) {
        total_edges += l.edges_scanned;
        total_cas += l.cas_attempts;
        total_cas_fails += l.cas_failures;
        total_queue_pushes += l.queue_pushes;
    }
    for (const auto& t : thread_stats) {
        total_active_time += t.active_time_sec;
        total_barrier_time += t.barrier_time_sec;
    }

    print_level_report(level_stats);
    print_thread_report(thread_stats);
    print_hub_report(hub_stats);

    double cas_fail_rate = (total_cas > 0) ? (100.0 * total_cas_fails / total_cas) : 0.0;
    double atomics_per_edge = (total_edges > 0) ? (static_cast<double>(total_cas + total_queue_pushes) / total_edges) : 0.0;
    double barrier_overhead_pct = (total_active_time + total_barrier_time > 0.0) 
        ? (100.0 * total_barrier_time / (total_active_time + total_barrier_time)) : 0.0;

    std::cout << "---------------------- QUANTITATIVE SUMMARY ----------------------\n";
    std::cout << "Total Traversal Time:            " << std::fixed << std::setprecision(3) << (total_time * 1000.0) << " ms\n";
    std::cout << "Total Directed Edges Scanned:    " << total_edges << "\n";
    std::cout << "Total Atomic CAS Operations:     " << total_cas << " (" << std::setprecision(2) << (static_cast<double>(total_cas) / total_edges) << " per edge)\n";
    std::cout << "Failed / Redundant CAS Ops:      " << total_cas_fails << " (" << std::setprecision(1) << cas_fail_rate << "% redundant RFOs!)\n";
    std::cout << "Shared Queue Tail Atomic Pushes: " << total_queue_pushes << " (hammering single cache line)\n";
    std::cout << "Total Atomic Operations / Edge:  " << std::setprecision(2) << atomics_per_edge << " atomics/edge\n";
    std::cout << "Barrier Sync Overhead:           " << std::setprecision(1) << barrier_overhead_pct << "% of total thread-seconds\n";
    std::cout << "------------------------------------------------------------------\n\n";

    if (!csv_levels.empty()) {
        std::ofstream out(csv_levels);
        if (out) {
            out << "level,frontier_size,edges_scanned,cas_attempts,cas_successes,cas_failures,elapsed_ms,barrier_ms,load_imbalance\n";
            for (const auto& l : level_stats) {
                out << l.level << "," << l.frontier_size << "," << l.edges_scanned << ","
                    << l.cas_attempts << "," << l.cas_successes << "," << l.cas_failures << ","
                    << (l.elapsed_sec * 1000.0) << "," << (l.avg_barrier_sec * 1000.0) << ","
                    << l.load_imbalance << "\n";
            }
            std::cout << "Exported level profile to: " << csv_levels << "\n";
        }
    }

    if (!csv_threads.empty()) {
        std::ofstream out(csv_threads);
        if (out) {
            out << "thread,vertices_scanned,edges_scanned,cas_attempts,queue_pushes,active_ms,barrier_ms\n";
            for (size_t t = 0; t < thread_stats.size(); ++t) {
                out << t << "," << thread_stats[t].vertices_scanned << "," << thread_stats[t].edges_scanned << ","
                    << thread_stats[t].cas_attempts << "," << thread_stats[t].queue_pushes << ","
                    << (thread_stats[t].active_time_sec * 1000.0) << "," << (thread_stats[t].barrier_time_sec * 1000.0) << "\n";
            }
            std::cout << "Exported thread profile to: " << csv_threads << "\n";
        }
    }

    return 0;
}
