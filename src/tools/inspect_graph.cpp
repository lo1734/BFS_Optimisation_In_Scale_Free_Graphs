#include "graph.hpp"
#include <iostream>
#include <string>
#include <cstring>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <binary_graph.bin> [options]\n"
              << "Options:\n"
              << "  --csv <path>     Export degree distribution CSV\n"
              << "  --validate       Run integrity and symmetry validation\n"
              << "  --help           Show this help message\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2 || std::strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    std::string input_path = argv[1];
    std::string csv_path = "";
    bool validate = false;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "--validate") == 0) {
            validate = true;
        }
    }

    std::cout << "Loading binary CSR graph from: " << input_path << " ...\n";
    utils::Timer timer;
    try {
        graph::CSRGraph g = graph::CSRGraph::load_binary(input_path);
        std::cout << "Loaded successfully in " << std::fixed << std::setprecision(3) 
                  << timer.elapsed_seconds() << "s\n\n";

        g.print_summary();

        if (validate) {
            std::cout << "\nValidating graph structural integrity and symmetry...\n";
            utils::Timer val_timer;
            bool ok = g.validate_integrity(true);
            std::cout << "Validation " << (ok ? "PASSED" : "FAILED") 
                      << " in " << std::fixed << std::setprecision(3) << val_timer.elapsed_seconds() << "s\n";
            if (!ok) return 1;
        }

        if (!csv_path.empty()) {
            std::cout << "Exporting degree distribution CSV to: " << csv_path << "\n";
            g.export_degree_distribution_csv(csv_path);
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
