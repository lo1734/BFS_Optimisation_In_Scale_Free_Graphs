# OpenMP BFS Project Makefile
# Targets:
#   make all           - Build all benchmarks, tools, and tests
#   make microbench    - Build atomic contention microbenchmark
#   make tools         - Build graph generator and inspection tools
#   make test          - Build and run unit tests
#   make clean         - Clean object and binary directories

CXX := $(shell which g++-15 2>/dev/null || which g++-14 2>/dev/null || which g++ 2>/dev/null || echo clang++)

CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread -Iinclude

# OpenMP flag detection (GCC vs Apple Clang with Homebrew libomp)
ifeq ($(findstring clang,$(CXX)),clang)
    OPENMP_FLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
else
    OPENMP_FLAGS = -fopenmp
    LDFLAGS += -fopenmp
endif

CXXFLAGS += $(OPENMP_FLAGS)

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

COMMON_SRCS = $(SRC_DIR)/graph/graph.cpp $(SRC_DIR)/graph/rmat_generator.cpp
COMMON_OBJS = $(OBJ_DIR)/graph/graph.o $(OBJ_DIR)/graph/rmat_generator.o

TARGETS = $(BIN_DIR)/atomic_microbench \
          $(BIN_DIR)/generate_graph \
          $(BIN_DIR)/inspect_graph \
          $(BIN_DIR)/test_graph

.PHONY: all clean test microbench tools

all: directories $(TARGETS)

directories:
	@mkdir -p $(OBJ_DIR)/benchmarks $(OBJ_DIR)/graph $(OBJ_DIR)/tools $(OBJ_DIR)/tests $(BIN_DIR) data

$(OBJ_DIR)/graph/graph.o: $(SRC_DIR)/graph/graph.cpp $(INC_DIR)/graph.hpp $(INC_DIR)/utils.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/graph/rmat_generator.o: $(SRC_DIR)/graph/rmat_generator.cpp $(INC_DIR)/rmat_generator.hpp $(INC_DIR)/graph.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR)/atomic_microbench: $(SRC_DIR)/benchmarks/atomic_microbench.cpp $(INC_DIR)/atomic_bench.hpp $(INC_DIR)/utils.hpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/generate_graph: $(SRC_DIR)/tools/generate_graph.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/inspect_graph: $(SRC_DIR)/tools/inspect_graph.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/test_graph: tests/test_graph.cpp $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

microbench: $(BIN_DIR)/atomic_microbench
	./$(BIN_DIR)/atomic_microbench

tools: $(BIN_DIR)/generate_graph $(BIN_DIR)/inspect_graph

test: $(BIN_DIR)/test_graph
	./$(BIN_DIR)/test_graph

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
