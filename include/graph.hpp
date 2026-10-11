#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <iostream>
#include "utils.hpp"

namespace graph {

// Degree statistics summary
struct DegreeStats {
    uint32_t min_degree = 0;
    uint32_t max_degree = 0;
    double avg_degree = 0.0;
    double stddev_degree = 0.0;
    double gini_coefficient = 0.0;
    uint32_t isolated_vertices = 0;
};

// Compact, 128-byte cache-line aligned Compressed Sparse Row (CSR) Graph
class CSRGraph {
public:
    uint32_t num_vertices = 0;
    uint64_t num_edges = 0; // Number of directed edges stored in CSR (2 * M_undirected)

    // row_ptr has size num_vertices + 1, aligned to 128 bytes
    uint64_t* row_ptr = nullptr;

    // col_idx has size num_edges, aligned to 128 bytes
    uint32_t* col_idx = nullptr;

    CSRGraph() = default;

    CSRGraph(uint32_t n, uint64_t m);

    ~CSRGraph();

    // Disable copy to prevent accidental deep copy of massive graph buffers
    CSRGraph(const CSRGraph&) = delete;
    CSRGraph& operator=(const CSRGraph&) = delete;

    // Enable move semantics
    CSRGraph(CSRGraph&& other) noexcept;
    CSRGraph& operator=(CSRGraph&& other) noexcept;

    // Allocate memory with 128-byte alignment
    void allocate(uint32_t n, uint64_t m);

    // Free memory
    void deallocate();

    // Degree of vertex u
    inline uint32_t degree(uint32_t u) const {
        return static_cast<uint32_t>(row_ptr[u + 1] - row_ptr[u]);
    }

    // Neighbors range of vertex u: [begin, end)
    inline const uint32_t* neighbors_begin(uint32_t u) const {
        return &col_idx[row_ptr[u]];
    }

    inline const uint32_t* neighbors_end(uint32_t u) const {
        return &col_idx[row_ptr[u + 1]];
    }

    // Compute comprehensive degree distribution statistics
    DegreeStats compute_degree_stats() const;

    // Export degree frequency histogram to CSV (for power-law plotting)
    void export_degree_distribution_csv(const std::string& filepath) const;

    // Binary serialization
    bool save_binary(const std::string& filepath) const;

    // Binary deserialization
    static CSRGraph load_binary(const std::string& filepath);

    // Validate structural integrity (symmetry, no duplicates, no self-loops, sorted neighbors)
    bool validate_integrity(bool check_symmetry = true) const;

    // Print summary to console
    void print_summary() const;
};

// Simple raw edge representation for generators
struct Edge {
    uint32_t u;
    uint32_t v;
};

} // namespace graph

#endif // GRAPH_HPP
