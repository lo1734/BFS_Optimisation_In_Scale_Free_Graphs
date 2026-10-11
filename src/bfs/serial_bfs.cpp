#include "bfs.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <unordered_set>

namespace bfs {

BFSResult run_serial_bfs(const graph::CSRGraph& g, uint32_t root, bool track_parents) {
    uint32_t n = g.num_vertices;
    BFSResult result;
    result.allocate(n, track_parents);
    result.root = root;

    // Initialize distances and parents
    for (uint32_t i = 0; i < n; ++i) {
        result.dist[i] = -1;
        if (track_parents) {
            result.parent[i] = UINT32_MAX;
        }
    }

    if (root >= n) {
        return result;
    }

    // Flat array queue for cache efficiency
    uint32_t* queue = utils::aligned_malloc<uint32_t>(n, CACHE_LINE_SIZE);
    uint32_t head = 0;
    uint32_t tail = 0;

    result.dist[root] = 0;
    if (track_parents) {
        result.parent[root] = root;
    }
    queue[tail++] = root;

    utils::Timer timer;

    int32_t max_level = 0;
    while (head < tail) {
        uint32_t u = queue[head++];
        int32_t u_dist = result.dist[u];
        if (u_dist > max_level) {
            max_level = u_dist;
        }

        uint64_t start_edge = g.row_ptr[u];
        uint64_t end_edge = g.row_ptr[u + 1];

        for (uint64_t e = start_edge; e < end_edge; ++e) {
            uint32_t v = g.col_idx[e];
            if (result.dist[v] == -1) {
                result.dist[v] = u_dist + 1;
                if (track_parents) {
                    result.parent[v] = u;
                }
                queue[tail++] = v;
            }
        }
    }

    result.elapsed_seconds = timer.elapsed_seconds();
    result.traversed_vertices = tail;
    result.num_levels = (tail > 0) ? (max_level + 1) : 0;

    // Calculate traversed edges (Graph500 undirected reachable edges count)
    uint64_t deg_sum = 0;
    for (uint32_t i = 0; i < tail; ++i) {
        deg_sum += g.degree(queue[i]);
    }
    result.traversed_edges = deg_sum / 2;

    if (result.elapsed_seconds > 0.0) {
        result.teps = static_cast<double>(result.traversed_edges) / result.elapsed_seconds;
    } else {
        result.teps = 0.0;
    }

    utils::aligned_free(queue);
    return result;
}

std::vector<uint32_t> select_random_roots(const graph::CSRGraph& g, size_t count, uint64_t seed) {
    std::vector<uint32_t> candidates;
    candidates.reserve(g.num_vertices);

    // Pick non-isolated vertices
    for (uint32_t u = 0; u < g.num_vertices; ++u) {
        if (g.degree(u) > 0) {
            candidates.push_back(u);
        }
    }

    if (candidates.empty()) {
        return {};
    }

    utils::Xoshiro256StarStar rng(seed);
    std::vector<uint32_t> roots;
    std::unordered_set<uint32_t> selected;

    size_t target_count = std::min(count, candidates.size());
    while (roots.size() < target_count) {
        size_t idx = rng.next() % candidates.size();
        uint32_t r = candidates[idx];
        if (selected.insert(r).second) {
            roots.push_back(r);
        }
    }

    return roots;
}

double compute_harmonic_mean_teps(const std::vector<double>& teps_list) {
    if (teps_list.empty()) return 0.0;
    double inv_sum = 0.0;
    size_t valid_count = 0;

    for (double t : teps_list) {
        if (t > 0.0) {
            inv_sum += 1.0 / t;
            valid_count++;
        }
    }

    if (valid_count == 0 || inv_sum == 0.0) return 0.0;
    return static_cast<double>(valid_count) / inv_sum;
}

bool validate_bfs(const graph::CSRGraph& g, const BFSResult& serial_res, const BFSResult& test_res) {
    if (serial_res.num_vertices != test_res.num_vertices) {
        std::cerr << "Validation Error: Vertex count mismatch ("
                  << serial_res.num_vertices << " vs " << test_res.num_vertices << ")\n";
        return false;
    }
    if (serial_res.root != test_res.root) {
        std::cerr << "Validation Error: Root mismatch ("
                  << serial_res.root << " vs " << test_res.root << ")\n";
        return false;
    }

    uint32_t n = g.num_vertices;
    uint32_t root = serial_res.root;

    // Check root distance
    if (test_res.dist[root] != 0) {
        std::cerr << "Validation Error: Root " << root << " has dist " << test_res.dist[root] << " != 0\n";
        return false;
    }

    // Check exact match with serial oracle
    uint32_t diff_count = 0;
    for (uint32_t v = 0; v < n; ++v) {
        if (serial_res.dist[v] != test_res.dist[v]) {
            if (diff_count < 10) {
                std::cerr << "Validation Error: Vertex " << v << " distance mismatch: serial=" 
                          << serial_res.dist[v] << ", test=" << test_res.dist[v] << "\n";
            }
            diff_count++;
        }
    }

    if (diff_count > 0) {
        std::cerr << "Validation Error: Total distance discrepancies: " << diff_count << "\n";
        return false;
    }

    // Graph500 Tree Edge Validation:
    // For every edge (u, v): if u is reached, then v must be reached, and |dist[u] - dist[v]| <= 1
    for (uint32_t u = 0; u < n; ++u) {
        int32_t du = test_res.dist[u];
        if (du < 0) continue; // Unreached

        uint64_t start_e = g.row_ptr[u];
        uint64_t end_e = g.row_ptr[u + 1];
        for (uint64_t e = start_e; e < end_e; ++e) {
            uint32_t v = g.col_idx[e];
            int32_t dv = test_res.dist[v];
            if (dv < 0) {
                std::cerr << "Validation Error: Edge (" << u << ", " << v << ") reached u (dist=" 
                          << du << ") but v is unreached!\n";
                return false;
            }
            if (std::abs(du - dv) > 1) {
                std::cerr << "Validation Error: Edge (" << u << ", " << v << ") distance diff > 1: dist[" 
                          << u << "]=" << du << ", dist[" << v << "]=" << dv << "\n";
                return false;
            }
        }
    }

    // If parent pointers tracked:
    if (test_res.has_parents) {
        if (test_res.parent[root] != root) {
            std::cerr << "Validation Error: Parent of root is not root!\n";
            return false;
        }

        for (uint32_t v = 0; v < n; ++v) {
            if (v == root || test_res.dist[v] < 0) continue;
            uint32_t p = test_res.parent[v];
            if (p >= n) {
                std::cerr << "Validation Error: Vertex " << v << " has invalid parent " << p << "\n";
                return false;
            }
            if (test_res.dist[v] != test_res.dist[p] + 1) {
                std::cerr << "Validation Error: Vertex " << v << " dist=" << test_res.dist[v] 
                          << " != parent " << p << " dist + 1 (" << test_res.dist[p] + 1 << ")\n";
                return false;
            }
        }
    }

    return true;
}

} // namespace bfs
