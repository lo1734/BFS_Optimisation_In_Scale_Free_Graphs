cmake_minimum_required(VERSION 3.16)
project(CAO_BFS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

find_package(OpenMP REQUIRED)

include_directories(include)

add_library(graph_lib STATIC
    src/graph/graph.cpp
    src/graph/rmat_generator.cpp
)
target_link_libraries(graph_lib PUBLIC OpenMP::OpenMP_CXX)

add_executable(atomic_microbench src/benchmarks/atomic_microbench.cpp)
target_link_libraries(atomic_microbench PRIVATE OpenMP::OpenMP_CXX)

add_executable(generate_graph src/tools/generate_graph.cpp)
target_link_libraries(generate_graph PRIVATE graph_lib OpenMP::OpenMP_CXX)

add_executable(inspect_graph src/tools/inspect_graph.cpp)
target_link_libraries(inspect_graph PRIVATE graph_lib OpenMP::OpenMP_CXX)

add_executable(test_graph tests/test_graph.cpp)
target_link_libraries(test_graph PRIVATE graph_lib OpenMP::OpenMP_CXX)
