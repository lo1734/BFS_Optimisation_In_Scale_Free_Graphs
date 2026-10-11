#include "atomic_bench.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

void print_header() {
    std::cout << std::string(85, '=') << "\n";
    std::cout << std::left 
              << std::setw(30) << "Benchmark Mode"
              << std::setw(10) << "Threads"
              << std::setw(16) << "Total Ops"
              << std::setw(14) << "Time (s)"
              << std::setw(15) << "Latency (ns)"
              << "\n";
    std::cout << std::string(85, '=') << "\n";
}

void print_result(const benchmark::BenchmarkResult& res) {
    std::cout << std::left 
              << std::setw(30) << res.test_name
              << std::setw(10) << res.num_threads
              << std::setw(16) << res.total_ops
              << std::fixed << std::setprecision(4)
              << std::setw(14) << res.elapsed_seconds
              << std::fixed << std::setprecision(2)
              << std::setw(15) << res.avg_latency_ns
              << "\n";
}

int main() {
    std::cout << "\n=====================================================================================\n";
    std::cout << "  PHASE 0: ATOMIC CONTENTION & FALSE SHARING MICROBENCHMARK\n";
    std::cout << "  Cache Line Alignment: " << CACHE_LINE_SIZE << " bytes\n";
    std::cout << "  Max OpenMP Threads:   " << omp_get_max_threads() << "\n";
    std::cout << "=====================================================================================\n\n";

    std::vector<int> thread_counts = {1, 2, 4, 8};
    const uint64_t total_contended_ops = 20000000ULL; // 20 million ops
    const uint64_t per_thread_uncontended = 20000000ULL; // 20 million per thread

    print_header();

    // Warm-up
    benchmark::run_uncontended(1, 1000000ULL, std::memory_order_relaxed);

    for (int t : thread_counts) {
        // 1. Uncontended padded fetch_add
        auto r_uncontended = benchmark::run_uncontended(t, per_thread_uncontended, std::memory_order_relaxed);
        print_result(r_uncontended);

        // 2. False sharing unpadded fetch_add
        auto r_false = benchmark::run_false_sharing(t, per_thread_uncontended, std::memory_order_relaxed);
        print_result(r_false);

        // 3. Contended shared fetch_add (relaxed)
        auto r_contended_rel = benchmark::run_contended(t, total_contended_ops, std::memory_order_relaxed);
        print_result(r_contended_rel);

        // 4. Contended shared fetch_add (seq_cst)
        auto r_contended_seq = benchmark::run_contended(t, total_contended_ops, std::memory_order_seq_cst);
        print_result(r_contended_seq);

        // 5. Contended CAS loop
        auto r_cas = benchmark::run_cas_contended(t, total_contended_ops);
        print_result(r_cas);

        std::cout << std::string(85, '-') << "\n";
    }

    std::cout << "\n[Key Takeaways from Phase 0 Microbenchmark]\n";
    std::cout << "1. Contention on a shared atomic causes cache line ping-ponging (RFO invalidations).\n";
    std::cout << "2. False sharing on unpadded adjacent variables degrades throughput without sharing data.\n";
    std::cout << "3. In BFS, global frontier counters and dist[] updates create this exact contention.\n\n";

    return 0;
}
