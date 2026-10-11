#ifndef BFS_HPP
#define BFS_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include "graph.hpp"
#include "utils.hpp"

namespace bfs {

struct BFSResult {
    uint32_t num_vertices = 0;
    uint32_t root = 0;
    int32_t* dist = nullptr;       // 128-byte aligned, -1 for unreached
    uint32_t* parent = nullptr;   // Optional 128-byte aligned parent array
    bool has_parents = false;

    uint32_t traversed_vertices = 0;
    uint64_t traversed_edges = 0;  // Undirected edges incident to reached vertices
    int32_t num_levels = 0;
    double elapsed_seconds = 0.0;
    double teps = 0.0;

    BFSResult() = default;

    ~BFSResult() {
        deallocate();
    }

    // Disable copy
    BFSResult(const BFSResult&) = delete;
    BFSResult& operator=(const BFSResult&) = delete;

    // Move semantics
    BFSResult(BFSResult&& other) noexcept
        : num_vertices(other.num_vertices),
          root(other.root),
          dist(other.dist),
          parent(other.parent),
          has_parents(other.has_parents),
          traversed_vertices(other.traversed_vertices),
          traversed_edges(other.traversed_edges),
          num_levels(other.num_levels),
          elapsed_seconds(other.elapsed_seconds),
          teps(other.teps) {
        other.dist = nullptr;
        other.parent = nullptr;
        other.num_vertices = 0;
    }

    BFSResult& operator=(BFSResult&& other) noexcept {
        if (this != &other) {
            deallocate();
            num_vertices = other.num_vertices;
            root = other.root;
            dist = other.dist;
            parent = other.parent;
            has_parents = other.has_parents;
            traversed_vertices = other.traversed_vertices;
            traversed_edges = other.traversed_edges;
            num_levels = other.num_levels;
            elapsed_seconds = other.elapsed_seconds;
            teps = other.teps;

            other.dist = nullptr;
            other.parent = nullptr;
            other.num_vertices = 0;
        }
        return *this;
    }

    void allocate(uint32_t n, bool track_parents) {
        deallocate();
        num_vertices = n;
        has_parents = track_parents;
        if (n > 0) {
            dist = utils::aligned_malloc<int32_t>(n, CACHE_LINE_SIZE);
            if (track_parents) {
                parent = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);
            }
        }
    }

    void deallocate() {
        if (dist) {
            utils::aligned_free(dist);
            dist = nullptr;
        }
        if (parent) {
            utils::aligned_free(parent);
            parent = nullptr;
        }
        num_vertices = 0;
    }
};

// 1. Serial BFS correctness oracle
BFSResult run_serial_bfs(const graph::CSRGraph& g, uint32_t root, bool track_parents = false);

// 2. Naive OpenMP Level-Synchronous Parallel BFS (with atomic contention & false sharing)
BFSResult run_naive_parallel_bfs(const graph::CSRGraph& g, uint32_t root, int num_threads, bool track_parents = false);

// 3. Contention-Reduced OpenMP Parallel BFS (Phase 4: Test-Before-CAS, TLS buffers, batched merge, relaxed atomics)
BFSResult run_contention_reduced_bfs(const graph::CSRGraph& g, uint32_t root, int num_threads, bool track_parents = false);

// 4. Visited Bitmap OpenMP Parallel BFS (Phase 4: 1-bit visited bitmap, TLS buffers, batched merge)
BFSResult run_bitmap_bfs(const graph::CSRGraph& g, uint32_t root, int num_threads, bool track_parents = false);

// 5. Cache-Line Padded & Aligned OpenMP BFS (Phase 5: 128-byte padding, aligned buffers & chunking)
BFSResult run_padded_bfs(const graph::CSRGraph& g, uint32_t root, int num_threads, bool track_parents = false);

// 6. Unpadded OpenMP BFS (Phase 5 Ablation: packed thread structs demonstrating false-sharing penalty)
BFSResult run_unpadded_bfs(const graph::CSRGraph& g, uint32_t root, int num_threads, bool track_parents = false);

// 7. Validation: compare test result against serial oracle and check Graph500 invariants
bool validate_bfs(const graph::CSRGraph& g, const BFSResult& serial_res, const BFSResult& test_res);

// 6. Helper: select random non-isolated root vertices in the giant component
std::vector<uint32_t> select_random_roots(const graph::CSRGraph& g, size_t count, uint64_t seed = 42);

// 7. Helper: compute harmonic mean TEPS
double compute_harmonic_mean_teps(const std::vector<double>& teps_list);

} // namespace bfs

#endif // BFS_HPP
