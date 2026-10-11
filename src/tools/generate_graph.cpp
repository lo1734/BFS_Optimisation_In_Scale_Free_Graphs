#include "rmat_generator.hpp"
#include <iostream>
#include <string>
#include <cstring>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --scale <N>        Scale factor (vertices = 2^scale, default: 16)\n"
              << "  --edge-factor <F>  Edge factor (edges = F * 2^scale, default: 16)\n"
              << "  --type <rmat|ba>   Graph generator type (default: rmat)\n"
              << "  --a <val>          R-MAT parameter a (default: 0.57)\n"
              << "  --b <val>          R-MAT parameter b (default: 0.19)\n"
              << "  --c <val>          R-MAT parameter c (default: 0.19)\n"
              << "  --d <val>          R-MAT parameter d (default: 0.05)\n"
              << "  --seed <S>         RNG seed (default: 42)\n"
              << "  --output <path>    Path to save binary CSR (.bin)\n"
              << "  --csv <path>       Path to export degree distribution CSV\n"
              << "  --validate         Validate graph integrity & symmetry\n"
              << "  --help             Show this help message\n";
}

int main(int argc, char* argv[]) {
    graph::RMATParams params;
    std::string type = "rmat";
    std::string output_path = "";
    std::string csv_path = "";
    bool validate = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            params.scale = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--edge-factor") == 0 && i + 1 < argc) {
            params.edge_factor = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            type = argv[++i];
        } else if (std::strcmp(argv[i], "--a") == 0 && i + 1 < argc) {
            params.a = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--b") == 0 && i + 1 < argc) {
            params.b = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--c") == 0 && i + 1 < argc) {
            params.c = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
            params.d = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            params.seed = std::stoull(argv[++i]);
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "--validate") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "\n=================================================================\n";
    std::cout << "  PHASE 1: SYNTHETIC GRAPH GENERATION (SCALE " << params.scale << ")\n";
    std::cout << "  Vertices:    " << (1U << params.scale) << " (2^" << params.scale << ")\n";
    std::cout << "  Edge Factor: " << params.edge_factor << "\n";
    std::cout << "  Type:        " << type << "\n";
    std::cout << "=================================================================\n\n";

    utils::Timer timer;
    graph::CSRGraph g;

    if (type == "rmat") {
        g = graph::GraphGenerator::generate_rmat(params);
    } else if (type == "ba") {
        uint32_t n = 1U << params.scale;
        g = graph::GraphGenerator::generate_barabasi_albert(n, params.edge_factor, params.seed);
    } else {
        std::cerr << "Unknown graph type: " << type << "\n";
        return 1;
    }

    double gen_time = timer.elapsed_seconds();
    std::cout << "Graph generation & CSR layout completed in: " << std::fixed << std::setprecision(3) 
              << gen_time << " seconds\n\n";

    g.print_summary();

    if (validate) {
        std::cout << "\nValidating structural integrity and symmetry...\n";
        utils::Timer val_timer;
        bool ok = g.validate_integrity(true);
        std::cout << "Integrity check " << (ok ? "PASSED" : "FAILED") 
                  << " in " << std::fixed << std::setprecision(3) << val_timer.elapsed_seconds() << "s\n";
        if (!ok) {
            std::cerr << "Error: Graph validation failed!\n";
            return 1;
        }
    }

    if (!csv_path.empty()) {
        std::cout << "Exporting degree distribution CSV to: " << csv_path << "\n";
        g.export_degree_distribution_csv(csv_path);
    }

    if (!output_path.empty()) {
        std::cout << "Saving binary CSR to: " << output_path << "\n";
        utils::Timer save_timer;
        bool ok = g.save_binary(output_path);
        if (ok) {
            std::cout << "Binary CSR saved successfully in " << std::fixed << std::setprecision(3)
                      << save_timer.elapsed_seconds() << "s\n";
        } else {
            std::cerr << "Error saving binary CSR to " << output_path << "\n";
            return 1;
        }
    }

    return 0;
}
