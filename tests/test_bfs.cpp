#include "graph.hpp"
#include "rmat_generator.hpp"
#include "bfs.hpp"
#include <iostream>
#include <cassert>

void test_small_graph() {
    std::cout << "[Test 1] Small Graph BFS Correctness... ";
    // Graph:
    // 0 - 1 - 2 - 3
    // |   |
    // 4 - 5
    std::vector<graph::Edge> edges = {
        {0, 1}, {1, 2}, {2, 3},
        {0, 4}, {1, 5}, {4, 5}
    };
    graph::CSRGraph g = graph::GraphGenerator::build_csr_from_edges(6, edges);

    auto serial_res = bfs::run_serial_bfs(g, 0, true);
    assert(serial_res.dist[0] == 0);
    assert(serial_res.dist[1] == 1);
    assert(serial_res.dist[4] == 1);
    assert(serial_res.dist[2] == 2);
    assert(serial_res.dist[5] == 2);
    assert(serial_res.dist[3] == 3);
    assert(serial_res.traversed_vertices == 6);
    assert(serial_res.traversed_edges == 6);

    for (int t : {1, 2, 4}) {
        auto naive_res = bfs::run_naive_parallel_bfs(g, 0, t, true);
        assert(bfs::validate_bfs(g, serial_res, naive_res));

        auto cr_res = bfs::run_contention_reduced_bfs(g, 0, t, true);
        assert(bfs::validate_bfs(g, serial_res, cr_res));

        auto bmp_res = bfs::run_bitmap_bfs(g, 0, t, true);
        assert(bfs::validate_bfs(g, serial_res, bmp_res));
    }

    std::cout << "PASSED\n";
}

void test_disconnected_components() {
    std::cout << "[Test 2] Disconnected Graph Components... ";
    // Component 1: {0, 1, 2}
    // Component 2: {3, 4, 5}
    std::vector<graph::Edge> edges = {
        {0, 1}, {1, 2}, {2, 0},
        {3, 4}, {4, 5}, {5, 3}
    };
    graph::CSRGraph g = graph::GraphGenerator::build_csr_from_edges(6, edges);

    auto serial_res = bfs::run_serial_bfs(g, 0, false);
    assert(serial_res.dist[0] == 0);
    assert(serial_res.dist[1] == 1);
    assert(serial_res.dist[2] == 1);
    assert(serial_res.dist[3] == -1);
    assert(serial_res.dist[4] == -1);
    assert(serial_res.dist[5] == -1);
    assert(serial_res.traversed_vertices == 3);

    for (int t : {1, 2, 4}) {
        auto naive_res = bfs::run_naive_parallel_bfs(g, 0, t, false);
        assert(bfs::validate_bfs(g, serial_res, naive_res));

        auto cr_res = bfs::run_contention_reduced_bfs(g, 0, t, false);
        assert(bfs::validate_bfs(g, serial_res, cr_res));

        auto bmp_res = bfs::run_bitmap_bfs(g, 0, t, false);
        assert(bfs::validate_bfs(g, serial_res, bmp_res));
    }

    std::cout << "PASSED\n";
}

void test_rmat_multi_roots() {
    std::cout << "[Test 3] R-MAT Graph Multi-Root Validation (SCALE 12, 10 roots)... ";
    graph::RMATParams params;
    params.scale = 12; // 4096 vertices
    params.edge_factor = 16;
    params.seed = 42;
    graph::CSRGraph g = graph::GraphGenerator::generate_rmat(params);

    auto roots = bfs::select_random_roots(g, 10, 12345);
    assert(roots.size() == 10);

    for (uint32_t root : roots) {
        auto serial_res = bfs::run_serial_bfs(g, root, true);
        assert(serial_res.traversed_vertices > 0);

        for (int t : {1, 2, 4, 8}) {
            auto naive_res = bfs::run_naive_parallel_bfs(g, root, t, true);
            assert(bfs::validate_bfs(g, serial_res, naive_res));

            auto cr_res = bfs::run_contention_reduced_bfs(g, root, t, true);
            assert(bfs::validate_bfs(g, serial_res, cr_res));

            auto bmp_res = bfs::run_bitmap_bfs(g, root, t, true);
            assert(bfs::validate_bfs(g, serial_res, bmp_res));
        }
    }

    std::cout << "PASSED (All roots matched serial oracle with 1, 2, 4, 8 threads)\n";
}

void test_barabasi_albert_validation() {
    std::cout << "[Test 4] Barabási-Albert Graph Validation (1000 vertices)... ";
    graph::CSRGraph g = graph::GraphGenerator::generate_barabasi_albert(1000, 10, 777);

    auto roots = bfs::select_random_roots(g, 5, 888);
    for (uint32_t root : roots) {
        auto serial_res = bfs::run_serial_bfs(g, root, false);
        for (int t : {1, 2, 4}) {
            auto naive_res = bfs::run_naive_parallel_bfs(g, root, t, false);
            assert(bfs::validate_bfs(g, serial_res, naive_res));

            auto cr_res = bfs::run_contention_reduced_bfs(g, root, t, false);
            assert(bfs::validate_bfs(g, serial_res, cr_res));

            auto bmp_res = bfs::run_bitmap_bfs(g, root, t, false);
            assert(bfs::validate_bfs(g, serial_res, bmp_res));
        }
    }

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "\n============================================\n";
    std::cout << "  RUNNING BFS CORRECTNESS & ORACLE TESTS    \n";
    std::cout << "============================================\n\n";

    test_small_graph();
    test_disconnected_components();
    test_rmat_multi_roots();
    test_barabasi_albert_validation();

    std::cout << "\nALL BFS CORRECTNESS TESTS PASSED SUCCESSFULLY!\n\n";
    return 0;
}
