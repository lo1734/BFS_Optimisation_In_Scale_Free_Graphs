#ifndef RMAT_GENERATOR_HPP
#define RMAT_GENERATOR_HPP

#include <cstdint>
#include <vector>
#include <string>
#include "graph.hpp"

namespace graph {

// Parameters for Kronecker / R-MAT graph generation
struct RMATParams {
    uint32_t scale = 16;         // Number of vertices N = 2^scale
    uint32_t edge_factor = 16;   // Number of undirected edges M = edge_factor * N
    double a = 0.57;             // Graph500 standard parameter a
    double b = 0.19;             // Graph500 standard parameter b
    double c = 0.19;             // Graph500 standard parameter c
    double d = 0.05;             // Graph500 standard parameter d
    uint64_t seed = 42;          // Base random seed
};

class GraphGenerator {
public:
    // Generate an undirected R-MAT scale-free graph according to Graph500 specs
    static CSRGraph generate_rmat(const RMATParams& params);

    // Generate a Barabási-Albert preferential attachment scale-free graph
    static CSRGraph generate_barabasi_albert(uint32_t num_vertices, uint32_t edges_per_node, uint64_t seed = 42);

    // Helper: convert raw edges to symmetrical, deduplicated, 128-byte aligned CSRGraph
    static CSRGraph build_csr_from_edges(uint32_t num_vertices, 
                                         const std::vector<Edge>& raw_edges);
};

} // namespace graph

#endif // RMAT_GENERATOR_HPP
