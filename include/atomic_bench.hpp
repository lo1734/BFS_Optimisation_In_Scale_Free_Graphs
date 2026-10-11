#ifndef ATOMIC_BENCH_HPP
#define ATOMIC_BENCH_HPP

#include <atomic>
#include <vector>
#include <cstdint>
#include <omp.h>
#include "utils.hpp"

namespace benchmark {

// Padded atomic structure ensuring strict cache line isolation (128 bytes)
struct alignas(CACHE_LINE_SIZE) PaddedAtomic {
    std::atomic<uint64_t> val{0};
    uint8_t pad[CACHE_LINE_SIZE - sizeof(std::atomic<uint64_t>)];
};

// Unpadded atomic structure: adjacent elements share the same cache line,
// inducing false sharing when accessed by distinct cores.
struct UnpaddedAtomic {
    std::atomic<uint64_t> val{0};
};

struct BenchmarkResult {
    std::string test_name;
    int num_threads;
    uint64_t total_ops;
    double elapsed_seconds;
    double ops_per_sec;
    double avg_latency_ns;
};

// 1. Uncontended: Each thread exclusively increments its own private, cache-line-padded atomic counter
inline BenchmarkResult run_uncontended(int num_threads, uint64_t iters_per_thread, std::memory_order order) {
    std::vector<PaddedAtomic> counters(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        counters[i].val.store(0, std::memory_order_relaxed);
    }

    utils::Timer timer;
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        for (uint64_t i = 0; i < iters_per_thread; ++i) {
            counters[tid].val.fetch_add(1, order);
        }
    }
    double elapsed = timer.elapsed_seconds();
    uint64_t total_ops = num_threads * iters_per_thread;

    return {
        "Uncontended (Padded)",
        num_threads,
        total_ops,
        elapsed,
        total_ops / elapsed,
        (elapsed / total_ops) * 1e9
    };
}

// 2. Contended (True Sharing): All threads concurrently hammer the exact same atomic variable
inline BenchmarkResult run_contended(int num_threads, uint64_t total_iters, std::memory_order order) {
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> shared_counter{0};
    uint64_t iters_per_thread = total_iters / num_threads;

    utils::Timer timer;
    #pragma omp parallel num_threads(num_threads)
    {
        for (uint64_t i = 0; i < iters_per_thread; ++i) {
            shared_counter.fetch_add(1, order);
        }
    }
    double elapsed = timer.elapsed_seconds();

    return {
        (order == std::memory_order_relaxed) ? "Contended (fetch_add relaxed)" : "Contended (fetch_add seq_cst)",
        num_threads,
        total_iters,
        elapsed,
        total_iters / elapsed,
        (elapsed / total_iters) * 1e9
    };
}

// 3. False Sharing: Distinct threads write to separate atomic counters packed on the same cache line
inline BenchmarkResult run_false_sharing(int num_threads, uint64_t iters_per_thread, std::memory_order order) {
    // Array of unpadded atomics allocated tightly next to each other
    std::vector<UnpaddedAtomic> counters(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        counters[i].val.store(0, std::memory_order_relaxed);
    }

    utils::Timer timer;
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        for (uint64_t i = 0; i < iters_per_thread; ++i) {
            counters[tid].val.fetch_add(1, order);
        }
    }
    double elapsed = timer.elapsed_seconds();
    uint64_t total_ops = num_threads * iters_per_thread;

    return {
        "False Sharing (Unpadded)",
        num_threads,
        total_ops,
        elapsed,
        total_ops / elapsed,
        (elapsed / total_ops) * 1e9
    };
}

// 4. Compare-And-Swap (CAS) Contention: Threads repeatedly contend on CAS loop
inline BenchmarkResult run_cas_contended(int num_threads, uint64_t total_iters) {
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> shared_val{0};
    uint64_t iters_per_thread = total_iters / num_threads;

    utils::Timer timer;
    #pragma omp parallel num_threads(num_threads)
    {
        for (uint64_t i = 0; i < iters_per_thread; ++i) {
            uint64_t current = shared_val.load(std::memory_order_relaxed);
            while (!shared_val.compare_exchange_weak(current, current + 1,
                                                     std::memory_order_relaxed,
                                                     std::memory_order_relaxed)) {
                // backoff / retry
            }
        }
    }
    double elapsed = timer.elapsed_seconds();

    return {
        "Contended CAS loop",
        num_threads,
        total_iters,
        elapsed,
        total_iters / elapsed,
        (elapsed / total_iters) * 1e9
    };
}

} // namespace benchmark

#endif // ATOMIC_BENCH_HPP
