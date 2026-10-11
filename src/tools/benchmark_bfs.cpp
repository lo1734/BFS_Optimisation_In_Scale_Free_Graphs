#include "bfs.hpp"
#include "rmat_generator.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>
#include <algorithm>

struct ConfigResult {
    std::string name;
    int threads;
    double harmonic_teps;
    double mteps;
    double median_time_ms;
    double speedup;
    double efficiency;
};

void print_benchmark_table(const std::vector<ConfigResult>& results) {
    std::cout << "\n" << std::string(92, '=') << "\n";
    std::cout << std::left 
              << std::setw(22) << "Algorithm"
              << std::setw(10) << "Threads"
              << std::setw(18) << "Harmonic TEPS"
              << std::setw(14) << "MTEPS"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Efficiency"
              << std::setw(16) << "Median Time"
              << "\n";
    std::cout << std::string(92, '=') << "\n";

    for (const auto& r : results) {
        std::cout << std::left 
                  << std::setw(22) << r.name
                  << std::setw(10) << r.threads
                  << std::scientific << std::setprecision(3)
                  << std::setw(18) << r.harmonic_teps
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << r.mteps
                  << std::setw(12) << r.speedup
                  << std::setw(14) << (std::to_string(static_cast<int>(r.efficiency)) + "%")
                  << std::setw(16) << (std::to_string(r.median_time_ms) + " ms")
                  << "\n";
    }
    std::cout << std::string(92, '=') << "\n\n";
}

int main(int argc, char* argv[]) {
    std::string input_path = "";
    uint32_t scale = 16;
    size_t num_roots = 16;
    std::vector<int> thread_counts = {1, 2, 4, 8};
    std::string csv_path = "";
    bool track_parents = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            scale = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--roots") == 0 && i + 1 < argc) {
            num_roots = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "--track-parents") == 0) {
            track_parents = true;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            thread_counts.clear();
            std::stringstream ss(argv[++i]);
            std::string t_str;
            while (std::getline(ss, t_str, ',')) {
                thread_counts.push_back(std::stoi(t_str));
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

    // Select random source roots in giant component
    std::cout << "Selecting " << num_roots << " random source root vertices...\n";
    std::vector<uint32_t> roots = bfs::select_random_roots(g, num_roots, 54321);
    if (roots.empty()) {
        std::cerr << "Error: No valid root vertices found!\n";
        return 1;
    }

    std::cout << "Roots selected: ";
    for (size_t i = 0; i < std::min(roots.size(), size_t(8)); ++i) {
        std::cout << roots[i] << " ";
    }
    if (roots.size() > 8) std::cout << "...";
    std::cout << "\n\nStarting BFS Scaling Benchmark (Harmonic Mean TEPS across " 
              << roots.size() << " roots)\n";

    // Warm-up run
    bfs::run_serial_bfs(g, roots[0], track_parents);
    bfs::run_naive_parallel_bfs(g, roots[0], 2, track_parents);

    std::vector<ConfigResult> benchmark_results;

    // 1. Run Serial BFS Baseline
    {
        std::cout << "Running Serial BFS Oracle... " << std::flush;
        std::vector<double> teps_list;
        std::vector<double> times_ms;

        for (uint32_t r : roots) {
            auto res = bfs::run_serial_bfs(g, r, track_parents);
            teps_list.push_back(res.teps);
            times_ms.push_back(res.elapsed_seconds * 1000.0);
        }

        std::sort(times_ms.begin(), times_ms.end());
        double h_teps = bfs::compute_harmonic_mean_teps(teps_list);
        double median_ms = times_ms[times_ms.size() / 2];

        benchmark_results.push_back({
            "Serial BFS (Oracle)",
            1,
            h_teps,
            h_teps / 1e6,
            median_ms,
            1.00,
            100.0
        });
        std::cout << "Done (Harmonic TEPS: " << std::scientific << std::setprecision(2) << h_teps << ")\n";
    }

    double serial_teps = benchmark_results[0].harmonic_teps;

    // 2. Run Naive Parallel BFS across thread counts
    for (int t : thread_counts) {
        std::cout << "Running Naive Parallel BFS (" << t << " threads)... " << std::flush;
        std::vector<double> teps_list;
        std::vector<double> times_ms;

        for (uint32_t r : roots) {
            auto res = bfs::run_naive_parallel_bfs(g, r, t, track_parents);
            teps_list.push_back(res.teps);
            times_ms.push_back(res.elapsed_seconds * 1000.0);
        }

        std::sort(times_ms.begin(), times_ms.end());
        double h_teps = bfs::compute_harmonic_mean_teps(teps_list);
        double median_ms = times_ms[times_ms.size() / 2];
        double speedup = (serial_teps > 0.0) ? (h_teps / serial_teps) : 0.0;
        double efficiency = (t > 0) ? (speedup / t) * 100.0 : 0.0;

        benchmark_results.push_back({
            "Naive OpenMP BFS",
            t,
            h_teps,
            h_teps / 1e6,
            median_ms,
            speedup,
            efficiency
        });
        std::cout << "Done (Harmonic TEPS: " << std::scientific << std::setprecision(2) 
                  << h_teps << ", Speedup: " << std::fixed << std::setprecision(2) << speedup << "x)\n";
    }

    // 3. Run Contention-Reduced BFS (Phase 4) across thread counts
    for (int t : thread_counts) {
        std::cout << "Running Contention-Reduced BFS (" << t << " threads)... " << std::flush;
        std::vector<double> teps_list;
        std::vector<double> times_ms;

        for (uint32_t r : roots) {
            auto res = bfs::run_contention_reduced_bfs(g, r, t, track_parents);
            teps_list.push_back(res.teps);
            times_ms.push_back(res.elapsed_seconds * 1000.0);
        }

        std::sort(times_ms.begin(), times_ms.end());
        double h_teps = bfs::compute_harmonic_mean_teps(teps_list);
        double median_ms = times_ms[times_ms.size() / 2];
        double speedup = (serial_teps > 0.0) ? (h_teps / serial_teps) : 0.0;
        double efficiency = (t > 0) ? (speedup / t) * 100.0 : 0.0;

        benchmark_results.push_back({
            "Contention-Reduced BFS",
            t,
            h_teps,
            h_teps / 1e6,
            median_ms,
            speedup,
            efficiency
        });
        std::cout << "Done (Harmonic TEPS: " << std::scientific << std::setprecision(2) 
                  << h_teps << ", Speedup: " << std::fixed << std::setprecision(2) << speedup << "x)\n";
    }

    // 4. Run Visited Bitmap BFS (Phase 4) across thread counts
    for (int t : thread_counts) {
        std::cout << "Running Visited Bitmap BFS (" << t << " threads)... " << std::flush;
        std::vector<double> teps_list;
        std::vector<double> times_ms;

        for (uint32_t r : roots) {
            auto res = bfs::run_bitmap_bfs(g, r, t, track_parents);
            teps_list.push_back(res.teps);
            times_ms.push_back(res.elapsed_seconds * 1000.0);
        }

        std::sort(times_ms.begin(), times_ms.end());
        double h_teps = bfs::compute_harmonic_mean_teps(teps_list);
        double median_ms = times_ms[times_ms.size() / 2];
        double speedup = (serial_teps > 0.0) ? (h_teps / serial_teps) : 0.0;
        double efficiency = (t > 0) ? (speedup / t) * 100.0 : 0.0;

        benchmark_results.push_back({
            "Visited Bitmap BFS",
            t,
            h_teps,
            h_teps / 1e6,
            median_ms,
            speedup,
            efficiency
        });
        std::cout << "Done (Harmonic TEPS: " << std::scientific << std::setprecision(2) 
                  << h_teps << ", Speedup: " << std::fixed << std::setprecision(2) << speedup << "x)\n";
    }

    // Print summary table
    print_benchmark_table(benchmark_results);

    // Export CSV if requested
    if (!csv_path.empty()) {
        std::ofstream out(csv_path);
        if (out) {
            out << "algorithm,threads,harmonic_teps,mteps,speedup,efficiency,median_time_ms\n";
            for (const auto& r : benchmark_results) {
                out << r.name << "," << r.threads << "," << r.harmonic_teps << "," 
                    << r.mteps << "," << r.speedup << "," << r.efficiency << "," 
                    << r.median_time_ms << "\n";
            }
            std::cout << "Scaling benchmark results exported to: " << csv_path << "\n";
        }
    }

    return 0;
}
