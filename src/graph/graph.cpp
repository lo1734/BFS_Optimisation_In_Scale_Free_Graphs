#include "graph.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <omp.h>

namespace graph {

static const uint64_t GRAPH_MAGIC = 0x43414F5F42465331ULL; // "CAO_BFS1"
static const uint32_t GRAPH_VERSION = 1;

CSRGraph::CSRGraph(uint32_t n, uint64_t m) {
    allocate(n, m);
}

CSRGraph::~CSRGraph() {
    deallocate();
}

CSRGraph::CSRGraph(CSRGraph&& other) noexcept
    : num_vertices(other.num_vertices),
      num_edges(other.num_edges),
      row_ptr(other.row_ptr),
      col_idx(other.col_idx) {
    other.num_vertices = 0;
    other.num_edges = 0;
    other.row_ptr = nullptr;
    other.col_idx = nullptr;
}

CSRGraph& CSRGraph::operator=(CSRGraph&& other) noexcept {
    if (this != &other) {
        deallocate();
        num_vertices = other.num_vertices;
        num_edges = other.num_edges;
        row_ptr = other.row_ptr;
        col_idx = other.col_idx;

        other.num_vertices = 0;
        other.num_edges = 0;
        other.row_ptr = nullptr;
        other.col_idx = nullptr;
    }
    return *this;
}

void CSRGraph::allocate(uint32_t n, uint64_t m) {
    deallocate();
    num_vertices = n;
    num_edges = m;
    if (n > 0) {
        row_ptr = utils::aligned_malloc<uint64_t>(n + 1, CACHE_LINE_SIZE);
    }
    if (m > 0) {
        col_idx = utils::aligned_malloc<uint32_t>(m, CACHE_LINE_SIZE);
    }
}

void CSRGraph::deallocate() {
    if (row_ptr) {
        utils::aligned_free(row_ptr);
        row_ptr = nullptr;
    }
    if (col_idx) {
        utils::aligned_free(col_idx);
        col_idx = nullptr;
    }
    num_vertices = 0;
    num_edges = 0;
}

DegreeStats CSRGraph::compute_degree_stats() const {
    DegreeStats stats;
    if (num_vertices == 0) return stats;

    std::vector<uint32_t> degs(num_vertices);
    uint32_t min_deg = UINT32_MAX;
    uint32_t max_deg = 0;
    uint64_t sum_deg = 0;
    uint32_t isolated = 0;

    #pragma omp parallel for reduction(min:min_deg) reduction(max:max_deg) \
                             reduction(+:sum_deg)  \
                             reduction(+:isolated) schedule(static)

    for (uint32_t i = 0; i < num_vertices; ++i) {
        uint32_t d = degree(i);
        degs[i] = d;
        if (d < min_deg) min_deg = d;
        if (d > max_deg) max_deg = d;
        sum_deg += d;
        if (d == 0) isolated++;
    }

    stats.min_degree = min_deg;
    stats.max_degree = max_deg;
    stats.isolated_vertices = isolated;
    stats.avg_degree = static_cast<double>(sum_deg) / num_vertices;

    // Variance & Standard Deviation
    double var_sum = 0.0;
    double avg = stats.avg_degree;
    #pragma omp parallel for reduction(+:var_sum) schedule(static)
    for (uint32_t i = 0; i < num_vertices; ++i) {
        double diff = degs[i] - avg;
        var_sum += diff * diff;
    }
    stats.stddev_degree = std::sqrt(var_sum / num_vertices);

    // Gini coefficient calculation:
    // G = (2 * sum_{i=1}^n i * y_i) / (n * sum_{i=1}^n y_i) - (n + 1) / n
    // for sorted degrees y_1 <= y_2 <= ... <= y_n
    if (sum_deg > 0) {
        // Parallel or fast sort for degrees
        std::vector<uint32_t> sorted_degs = degs;
        std::sort(sorted_degs.begin(), sorted_degs.end());

        double weighted_sum = 0.0;
        #pragma omp parallel for reduction(+:weighted_sum) schedule(static)
        for (uint64_t i = 0; i < num_vertices; ++i) {
            weighted_sum += (static_cast<double>(i + 1)) * sorted_degs[i];
        }

        double n = static_cast<double>(num_vertices);
        double total = static_cast<double>(sum_deg);
        stats.gini_coefficient = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n;
    } else {
        stats.gini_coefficient = 0.0;
    }

    return stats;
}

void CSRGraph::export_degree_distribution_csv(const std::string& filepath) const {
    // Map degree -> frequency
    std::map<uint32_t, uint64_t> freq_map;

    #pragma omp parallel
    {
        std::map<uint32_t, uint64_t> local_map;
        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < num_vertices; ++i) {
            local_map[degree(i)]++;
        }

        #pragma omp critical
        {
            for (const auto& [deg, count] : local_map) {
                freq_map[deg] += count;
            }
        }
    }

    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "Error: Could not open " << filepath << " for writing CSV.\n";
        return;
    }

    out << "degree,frequency,probability\n";
    for (const auto& [deg, count] : freq_map) {
        double prob = static_cast<double>(count) / num_vertices;
        out << deg << "," << count << "," << prob << "\n";
    }
    out.close();
}

bool CSRGraph::save_binary(const std::string& filepath) const {
    FILE* fp = std::fopen(filepath.c_str(), "wb");
    if (!fp) {
        std::cerr << "Error: Failed to open file for writing: " << filepath << "\n";
        return false;
    }

    // Write header
    if (std::fwrite(&GRAPH_MAGIC, sizeof(uint64_t), 1, fp) != 1 ||
        std::fwrite(&GRAPH_VERSION, sizeof(uint32_t), 1, fp) != 1 ||
        std::fwrite(&num_vertices, sizeof(uint32_t), 1, fp) != 1 ||
        std::fwrite(&num_edges, sizeof(uint64_t), 1, fp) != 1) {
        std::fclose(fp);
        return false;
    }

    // Write row_ptr
    size_t row_written = std::fwrite(row_ptr, sizeof(uint64_t), num_vertices + 1, fp);
    if (row_written != num_vertices + 1) {
        std::fclose(fp);
        return false;
    }

    // Write col_idx
    if (num_edges > 0) {
        size_t col_written = std::fwrite(col_idx, sizeof(uint32_t), num_edges, fp);
        if (col_written != num_edges) {
            std::fclose(fp);
            return false;
        }
    }

    std::fclose(fp);
    return true;
}

CSRGraph CSRGraph::load_binary(const std::string& filepath) {
    FILE* fp = std::fopen(filepath.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("Failed to open graph file: " + filepath);
    }

    uint64_t magic = 0;
    uint32_t version = 0;
    uint32_t n = 0;
    uint64_t m = 0;

    if (std::fread(&magic, sizeof(uint64_t), 1, fp) != 1 ||
        std::fread(&version, sizeof(uint32_t), 1, fp) != 1 ||
        std::fread(&n, sizeof(uint32_t), 1, fp) != 1 ||
        std::fread(&m, sizeof(uint64_t), 1, fp) != 1) {
        std::fclose(fp);
        throw std::runtime_error("Corrupted graph header in: " + filepath);
    }

    if (magic != GRAPH_MAGIC) {
        std::fclose(fp);
        throw std::runtime_error("Invalid graph magic identifier in: " + filepath);
    }

    CSRGraph g(n, m);

    size_t row_read = std::fread(g.row_ptr, sizeof(uint64_t), n + 1, fp);
    if (row_read != n + 1) {
        std::fclose(fp);
        throw std::runtime_error("Truncated row_ptr array in: " + filepath);
    }

    if (m > 0) {
        size_t col_read = std::fread(g.col_idx, sizeof(uint32_t), m, fp);
        if (col_read != m) {
            std::fclose(fp);
            throw std::runtime_error("Truncated col_idx array in: " + filepath);
        }
    }

    std::fclose(fp);
    return g;
}

bool CSRGraph::validate_integrity(bool check_symmetry) const {
    if (num_vertices == 0) return true;
    if (row_ptr == nullptr || (num_edges > 0 && col_idx == nullptr)) return false;
    if (row_ptr[0] != 0 || row_ptr[num_vertices] != num_edges) return false;

    bool valid = true;

    #pragma omp parallel for reduction(&&:valid) schedule(dynamic, 1024)
    for (uint32_t u = 0; u < num_vertices; ++u) {
        uint64_t start = row_ptr[u];
        uint64_t end = row_ptr[u + 1];
        if (start > end || end > num_edges) {
            valid = false;
            continue;
        }

        uint32_t prev_v = 0;
        for (uint64_t e = start; e < end; ++e) {
            uint32_t v = col_idx[e];
            // Vertex bounds check
            if (v >= num_vertices) {
                valid = false;
                break;
            }
            // No self loops check
            if (v == u) {
                valid = false;
                break;
            }
            // Sorted neighbors and no duplicates check
            if (e > start && v <= prev_v) {
                valid = false;
                break;
            }
            prev_v = v;
        }
    }

    if (!valid) return false;

    // Check symmetry if requested: for every edge (u, v), v must have u in its neighbor list
    if (check_symmetry) {
        #pragma omp parallel for reduction(&&:valid) schedule(dynamic, 1024)
        for (uint32_t u = 0; u < num_vertices; ++u) {
            uint64_t start = row_ptr[u];
            uint64_t end = row_ptr[u + 1];
            for (uint64_t e = start; e < end; ++e) {
                uint32_t v = col_idx[e];
                // Search for u in neighbors of v using binary search
                const uint32_t* v_begin = neighbors_begin(v);
                const uint32_t* v_end = neighbors_end(v);
                if (!std::binary_search(v_begin, v_end, u)) {
                    valid = false;
                    break;
                }
            }
        }
    }

    return valid;
}

void CSRGraph::print_summary() const {
    auto stats = compute_degree_stats();
    std::cout << "---------------- Graph Summary ----------------\n";
    std::cout << "Vertices:           " << num_vertices << "\n";
    std::cout << "Directed Edges:     " << num_edges << " (" << (num_edges / 2) << " undirected)\n";
    std::cout << "Min Degree:         " << stats.min_degree << "\n";
    std::cout << "Max Degree:         " << stats.max_degree << " (Hub vertex)\n";
    std::cout << "Avg Degree:         " << std::fixed << std::setprecision(2) << stats.avg_degree << "\n";
    std::cout << "Std Dev Degree:     " << std::fixed << std::setprecision(2) << stats.stddev_degree << "\n";
    std::cout << "Gini Coefficient:   " << std::fixed << std::setprecision(4) << stats.gini_coefficient << "\n";
    std::cout << "Isolated Vertices:  " << stats.isolated_vertices << "\n";
    std::cout << "Memory footprint:   " << std::fixed << std::setprecision(2) 
              << ((num_vertices + 1) * sizeof(uint64_t) + num_edges * sizeof(uint32_t)) / (1024.0 * 1024.0) 
              << " MB\n";
    std::cout << "-----------------------------------------------\n";
}

} // namespace graph
