#include "graph.hpp"
#include "rmat_generator.hpp"
#include <iostream>
#include <cassert>
#include <cstdio>

void test_aligned_allocation() {
    std::cout << "[Test 1] Aligned Memory Allocation... ";
    size_t count = 1024;
    uint64_t* ptr = utils::aligned_malloc<uint64_t>(count, CACHE_LINE_SIZE);
    assert(ptr != nullptr);
    assert(reinterpret_cast<uintptr_t>(ptr) % CACHE_LINE_SIZE == 0);
    for (size_t i = 0; i < count; ++i) ptr[i] = i;
    assert(ptr[500] == 500);
    utils::aligned_free(ptr);
    std::cout << "PASSED (aligned to " << CACHE_LINE_SIZE << " bytes)\n";
}

void test_small_toy_graph() {
    std::cout << "[Test 2] Toy Graph Symmetrization & Deduplication... ";
    // Raw edges with duplicates, self loops, reverse edges
    // 0-1, 1-0 (duplicate when undirected)
    // 0-0 (self loop)
    // 1-2
    // 2-3, 2-3 (duplicate)
    std::vector<graph::Edge> raw_edges = {
        {0, 1},
        {1, 0},
        {0, 0},
        {1, 2},
        {2, 3},
        {2, 3}
    };
    uint32_t num_vertices = 4;

    graph::CSRGraph g = graph::GraphGenerator::build_csr_from_edges(num_vertices, raw_edges);

    assert(g.num_vertices == 4);
    // Undirected unique edges: (0,1), (1,2), (2,3) -> 3 edges -> 6 directed in CSR
    assert(g.num_edges == 6);
    assert(g.validate_integrity(true));

    // Check degree of each vertex
    // 0 -> {1} (deg 1)
    // 1 -> {0, 2} (deg 2)
    // 2 -> {1, 3} (deg 2)
    // 3 -> {2} (deg 1)
    assert(g.degree(0) == 1);
    assert(g.degree(1) == 2);
    assert(g.degree(2) == 2);
    assert(g.degree(3) == 1);

    assert(g.col_idx[g.row_ptr[0]] == 1);
    assert(g.col_idx[g.row_ptr[1]] == 0 && g.col_idx[g.row_ptr[1] + 1] == 2);

    std::cout << "PASSED\n";
}

void test_rmat_generation() {
    std::cout << "[Test 3] R-MAT Graph Generation & Symmetry (SCALE 10)... ";
    graph::RMATParams params;
    params.scale = 10; // 1024 vertices
    params.edge_factor = 16;
    params.seed = 12345;

    graph::CSRGraph g = graph::GraphGenerator::generate_rmat(params);
    assert(g.num_vertices == 1024);
    assert(g.num_edges > 0);
    assert(reinterpret_cast<uintptr_t>(g.row_ptr) % CACHE_LINE_SIZE == 0);
    assert(reinterpret_cast<uintptr_t>(g.col_idx) % CACHE_LINE_SIZE == 0);

    bool valid = g.validate_integrity(true);
    assert(valid);

    auto stats = g.compute_degree_stats();
    assert(stats.max_degree > stats.avg_degree); // Power-law skew check
    assert(stats.gini_coefficient > 0.3); // High inequality check

    std::cout << "PASSED (Vertices: " << g.num_vertices << ", Directed Edges: " << g.num_edges 
              << ", Max Deg: " << stats.max_degree << ", Gini: " << stats.gini_coefficient << ")\n";
}

void test_binary_roundtrip() {
    std::cout << "[Test 4] Binary CSR Serialization Round-Trip... ";
    graph::RMATParams params;
    params.scale = 9; // 512 vertices
    params.edge_factor = 8;
    graph::CSRGraph original = graph::GraphGenerator::generate_rmat(params);

    const std::string tmp_file = "data/test_roundtrip.bin";
    bool saved = original.save_binary(tmp_file);
    assert(saved);

    graph::CSRGraph loaded = graph::CSRGraph::load_binary(tmp_file);
    assert(loaded.num_vertices == original.num_vertices);
    assert(loaded.num_edges == original.num_edges);
    assert(reinterpret_cast<uintptr_t>(loaded.row_ptr) % CACHE_LINE_SIZE == 0);
    assert(reinterpret_cast<uintptr_t>(loaded.col_idx) % CACHE_LINE_SIZE == 0);

    for (uint32_t i = 0; i <= original.num_vertices; ++i) {
        assert(loaded.row_ptr[i] == original.row_ptr[i]);
    }
    for (uint64_t i = 0; i < original.num_edges; ++i) {
        assert(loaded.col_idx[i] == original.col_idx[i]);
    }
    assert(loaded.validate_integrity(true));

    std::remove(tmp_file.c_str());
    std::cout << "PASSED\n";
}

void test_barabasi_albert() {
    std::cout << "[Test 5] Barabási-Albert Graph Generation... ";
    graph::CSRGraph g = graph::GraphGenerator::generate_barabasi_albert(500, 8, 999);
    assert(g.num_vertices == 500);
    assert(g.validate_integrity(true));
    auto stats = g.compute_degree_stats();
    assert(stats.max_degree > stats.avg_degree);
    std::cout << "PASSED (Max Deg: " << stats.max_degree << ", Avg: " << stats.avg_degree << ")\n";
}

int main() {
    std::cout << "\n=========================================\n";
    std::cout << "  RUNNING GRAPH UNIT & INTEGRITY TESTS   \n";
    std::cout << "=========================================\n\n";

    test_aligned_allocation();
    test_small_toy_graph();
    test_rmat_generation();
    test_binary_roundtrip();
    test_barabasi_albert();

    std::cout << "\nALL 5 TEST SUITES PASSED SUCCESSFULLY!\n\n";
    return 0;
}
