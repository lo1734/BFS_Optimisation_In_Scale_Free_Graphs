# OpenMP BFS on Scale-Free Graphs: Phase-Wise Implementation Plan

**Goal:** A level-synchronous parallel BFS that scales on SMP machines by cutting atomic contention and removing false sharing, evaluated on synthetic power-law graphs with millions of vertices.

---

## 1. Project Overview

### Problem Statement
BFS on scale-free graphs is memory-bound and irregular. A few hub vertices have huge degrees, so many threads hammer the same cache lines (visited flags, distance entries, queue tail counters). Under MESI/MOESI cache coherence protocols, every contended write forces a Read-For-Ownership (RFO), invalidating copies in other cores' caches, causing the cache line to ping-pong between cores.

### Core Thesis
Speed-up comes from:
1. **Doing fewer atomics:** Test-before-CAS, visited bitmaps, thread-local frontier buffers, direction switching (top-down / bottom-up).
2. **Preventing false sharing:** Ensuring unrelated data written by different threads never shares a 64-byte cache line (cache-line alignment, padding, thread-private structures).

### Deliverables
- **Graph Generator & Loader:** R-MAT / Kronecker (Graph500 specification) and Barabási-Albert generators, with a 64-byte aligned CSR loader.
- **Implementations Spectrum:**
  1. Serial BFS baseline (correctness oracle).
  2. Naive OpenMP parallel BFS (level-synchronous with atomics).
  3. Optimized OpenMP BFS (incorporating contention-reduction and false-sharing-elimination techniques).
  4. Direction-optimizing (hybrid) BFS (Beamer et al.).
- **Profiling Evidence:** Hardware counters and profiling data (`perf stat`, `perf c2c` HITM events) demonstrating reduction in cache invalidations and coherence traffic.
- **Benchmark Suite & Final Report:** Automated harness measuring strong scaling, parallel efficiency, TEPS, and ablation studies.

### Success Metrics
- **Traversed Edges Per Second (TEPS):** Graph500 standard metric ($TEPS = \frac{m_{\text{reachable}}}{t_{\text{traversal}}}$).
- **Strong-Scaling Speed-up & Parallel Efficiency:** Across $1, 2, 4, 8, \dots, N$ threads.
- **Hardware Performance Counters:** Atomic operations per edge, L1/LLC miss rate, HITM count (from `perf c2c`).
- **Correctness:** BFS distance arrays (`dist[]`) identical to serial baseline across every test graph and root node.

---

## 2. Phase Roadmap at a Glance

| Phase | Focus | Output |
| :---: | :--- | :--- |
| **0** | Environment, tooling, hardware study | Reproducible test rig, hardware profile, atomic microbenchmark |
| **1** | Graph generation and CSR layout | Parallel R-MAT generator, 64-byte aligned CSR loader |
| **2** | Serial and naive parallel BFS | Correctness oracle, naive OpenMP BFS, test harness |
| **3** | Profiling the bottlenecks | `perf stat` and `perf c2c` profile showing cache line contention & HITM |
| **4** | Reduce atomic overhead | Test-before-CAS, visited bitmap, thread-local buffers, batched merge |
| **5** | Eliminate false sharing | Cache-line-aware padding (`alignas(64)`), partitioned chunking |
| **6** | Direction-optimizing (hybrid) BFS | Beamer top-down / bottom-up hybrid traversal |
| **7** | NUMA, scheduling, load balance | First-touch placement, thread affinity, edge-balanced scheduling |
| **8** | Evaluation and ablation | Multi-threaded scaling experiments, ablation suite, plots |
| **9** | Report and documentation | Comprehensive final technical paper/report |

---

## 3. Phase 0: Environment and Foundations

### Tasks
- **Fix Toolchain:** GCC 12+ or Clang with `-O3 -march=native -fopenmp`; CMake build system; pin compiler flags.
- **Machine Topology:** Record architecture details via `lscpu`, `lstopo`, `numactl --hardware` (cores, SMT threads, L1/L2/L3 cache sizes, 64-byte line size, NUMA nodes, coherence protocol: Intel MESIF / AMD MOESI).
- **Setup Measurement Tools:** `perf stat`, `perf c2c`, `perf record`, Intel VTune / AMD uProf, and Valgrind Cachegrind.
- **Reduce Measurement Noise:** Set CPU frequency governor to `performance`, disable CPU turbo boost if possible, pin thread execution.

### Research Topics
- MESI states (Modified, Exclusive, Shared, Invalid) and transitions on read/write/RFO; MOESI's Owned state vs. MESIF's Forward state.
- Snooping vs. directory-based coherence mechanisms and impact on scaling.
- Cost of cache-line transfers across cores (same socket vs. cross-socket ping-pong microbenchmarks).
- C++11 memory models: `memory_order_relaxed` vs. acquire/release semantics; compilation of OpenMP atomic pragmas to `lock`-prefixed instructions on x86.

### Exit Criteria
- Microbenchmark demonstrating latency of uncontended vs. contended `fetch_add`.
- Documented baseline hardware topology profile.

---

## 4. Phase 1: Graph Generation and Representation

### Tasks
- **R-MAT / Kronecker Generator:** Implement Graph500 parameters ($a=0.57, b=0.19, c=0.19, d=0.05$) and optionally Barabási-Albert preferential attachment. Parameterize by `SCALE` ($2^{\text{SCALE}}$ vertices) and edge factor ($16$).
- **Parallel Edge Generation:** Parallelize edge generation using thread-private RNGs (e.g., `xoshiro256**` or `PCG`) with distinct seed states to eliminate RNG contention.
- **CSR Construction:** Eliminate self-loops and duplicate edges, symmetrize for undirected graphs, and build Compressed Sparse Row (CSR) arrays (`row_ptr`, `col_idx`) via parallel counting sort / prefix sums.
- **Memory Optimization:** Use 32-bit vertex IDs (`uint32_t`) for graphs with $< 4\times 10^9$ vertices to halve memory bandwidth; 64-bit edge offsets (`uint64_t`).
- **Memory Alignment:** Allocate large graph arrays using 64-byte alignment (`aligned_alloc`, `posix_memalign`, or `alignas(64)`).
- **Validation:** Verify degree distributions fit a power-law curve (log-log degree histogram, maximum degree, Gini coefficient).

### Exit Criteria
- Verified synthetic graphs (SCALE 18, 20, 22, 24) saved in compact binary CSR format.
- Validated degree distribution plots.

---

## 5. Phase 2: Serial and Naive Parallel Baselines

### Tasks
- **Serial BFS:** Flat array queue with `dist[]` initialized to `-1`. Serves as the correctness oracle.
- **Naive Parallel BFS (Level-Synchronous):**
  - `#pragma omp parallel for` over current frontier array.
  - Claim unvisited vertices using `#pragma omp atomic capture` or `__sync_bool_compare_and_swap` / `__atomic_compare_exchange_n` on `dist[v]`.
  - Append to next frontier using shared atomic `fetch_add` on global frontier tail index.
- **Validation Harness:** Compare BFS distances and parent tree validity against serial BFS output across multiple random source vertices in the giant component.
- **Performance Baseline:** Measure and record harmonic mean TEPS across 16 to 64 random root vertices.

### Exit Criteria
- Naive parallel BFS passes all correctness checks.
- Baseline scaling curve measured (demonstrating poor scaling due to atomic/coherence bottlenecks).

---

## 6. Phase 3: Profile and Prove the Bottlenecks

### Tasks
- **Hardware Counter Collection:** Run `perf stat -e cache-misses,cache-references,LLC-load-misses,instructions,cycles` for $1, 2, 4, \dots, N$ threads.
- **Coherence Tracking:** Run `perf c2c record` and `perf c2c report` to detect cache lines experiencing high HITM (remote cache hit modified) events. Map addresses back to source symbols.
- **Quantitative Profiling:**
  - Measure atomic operations executed per edge.
  - Measure percentage of execution cycles stalled on memory pipelines.
  - Measure synchronization time spent at the level barrier.
  - Measure per-thread load imbalance (edges scanned per thread).
- **Ranked Bottlenecks List:** Produce a prioritized table of contested cache lines (global frontier tail, `dist[]` neighbor entries, shared counters).

### Exit Criteria
- Detailed profiling report identifying the exact variables and cache lines causing high coherence traffic and scaling bottlenecks.

---

## 7. Phase 4: Minimize Atomic Operation Overhead

### Techniques to Implement
1. **Test-Before-Atomic (Test-and-Test-and-Set):** Read `dist[v]` (or visited bit) with a plain relaxed load first. Attempt CAS only if currently unvisited. On power-law graphs, most neighbor queries hit already-visited vertices; this keeps lines in the `Shared` MESI state and eliminates unnecessary RFOs.
2. **Visited Bitmap:** Allocate a 1-bit-per-vertex visited bitmap (512 vertices per 64-byte cache line). Fit the working set into higher-level cache. Use `fetch_or` on 64-bit words only upon unvisited bit checks; write `dist[v]` only by the thread winning the claim.
3. **Thread-Local Frontier Buffers:** Allocate private frontier buffers per thread. Each thread appends newly discovered vertices to its private buffer with zero atomics on the queue tail.
4. **Batched Merging:** At the end of each BFS level, compute an exclusive prefix sum over thread-local buffer sizes and copy into the global frontier in parallel. Alternatively, reserve chunks of the global queue via a single atomic `fetch_add(chunk_size)` when local buffers fill.
5. **Benign-Race Variant:** Allow threads to write the current level number directly to `dist[v]` without CAS (all writers write the exact same level value). Handle duplicate vertices in the frontier.
6. **Relaxed Memory Ordering:** Use `__ATOMIC_RELAXED` memory order for claims, relying on the level-synchronous barrier for happens-before visibility.
7. **Hub Adjacency Splitting:** For extreme high-degree hub vertices, partition their neighbor ranges into chunks distributed across multiple threads.

### Exit Criteria
- Atomic operations per edge drop by at least an order of magnitude.
- HITM count in `perf c2c` drops substantially; correctness matches serial oracle.

---

## 8. Phase 5: Eliminate False Sharing

### Techniques to Implement
1. **Cache-Line Padding:** Pad per-thread structures (buffers, counters, sizes) to 64 bytes (or `std::hardware_destructive_interference_size`):
   ```cpp
   struct alignas(64) PerThread {
       uint32_t* buf;
       size_t size;
       size_t cap;
       uint64_t edges_scanned;
       char pad[64 - (sizeof(uint32_t*) + 2 * sizeof(size_t) + sizeof(uint64_t)) % 64];
   };
   ```
2. **Distinct Buffer Allocations:** Allocate thread-local buffers from distinct 64-byte aligned addresses to prevent buffer boundaries from sharing lines.
3. **Cache-Line Aligned Partitioning:** When threads write to static ranges of `dist[]` or bitmaps, ensure chunk boundaries are multiples of 64 bytes (multiples of 16 for `uint32_t`, 8 for `uint64_t`).
4. **Bottom-Up Word Ownership:** Assign entire 64-bit bitmap words (or entire 64-byte cache lines) to single threads so writes require no atomics and no multi-core line sharing.
5. **Vertex Relabeling:** Reorder vertices (e.g., degree sorting or hub clustering) so hubs reside on compact, identifiable cache lines that can be isolated or treated with replicated buffers.
6. **Privatization & Reductions:** Replace all shared scalar counters with per-thread padded variables reduced once per level.
7. **Read-Mostly Replication:** Keep infrequently updated shared variables (e.g., `current_level`) in their own cache line or copy them to thread-local storage.

### Exit Criteria
- `perf c2c` confirms false-sharing hot lines are eliminated.
- Microbenchmarks demonstrate measurable performance gain between padded and unpadded layouts.

---

## 9. Phase 6: Direction-Optimizing (Hybrid) BFS

### Motivation
Scale-free graphs feature a few massive middle levels where the frontier contains a huge fraction of the graph's vertices. Top-down traversal wastes vast amounts of work checking already-visited edges.

### Tasks
- **Bottom-Up Traversal Step:**
  - For each unvisited vertex, scan its incoming neighbors.
  - Terminate immediately at the first neighbor present in the frontier (early exit).
  - The thread writing `dist[v]` owns $v$ exclusively—**zero atomic operations required**.
- **Dual Frontier Representations:** Maintain sparse list for top-down and dense bitmap for bottom-up; implement parallel format conversions.
- **Beamer Heuristic:**
  - Switch from **Top-Down to Bottom-Up** when $m_f > m_u / \alpha$ (frontier edge count exceeds threshold of unvisited edges; typically $\alpha \approx 14$).
  - Switch from **Bottom-Up to Top-Down** when $n_f < n / \beta$ (frontier vertex count drops; typically $\beta \approx 24$).
  - Tune $\alpha$ and $\beta$ empirically for target hardware and graph scales.
- **Integration:** Combine direction optimization with the Phase 4 & 5 contention-free layouts.

### Exit Criteria
- Major TEPS speed-up on SCALE 20–24.
- 100% distance validation matching serial baseline.

---

## 10. Phase 7: NUMA, Scheduling, and Load Balancing

### Tasks
- **First-Touch Page Placement:** Allocate and initialize `dist[]`, bitmaps, and CSR arrays in parallel using the same thread partitioning used during BFS execution, ensuring memory pages reside on the local NUMA node of the accessing core.
- **Thread Affinity:** Test and pin thread placement using `OMP_PROC_BIND=close|spread` and `OMP_PLACES=cores|threads`; evaluate performance with SMT enabled vs. disabled.
- **Degree-Aware Scheduling:** Compare `static`, `dynamic`, `guided`, and custom edge-balanced partitioning (partitioning frontier chunks by cumulative degree rather than raw vertex count).
- **Huge Pages:** Allocate large graph arrays using Transparent Huge Pages (THP) or `madvise(MADV_HUGEPAGE)` to minimize TLB miss rates.
- **Software Prefetching:** Insert `__builtin_prefetch` instructions to prefetch neighbor `dist[]`/bitmap entries ahead of vertex iteration.

### Exit Criteria
- Improved multi-socket scaling; memory bandwidth saturates near theoretical local bandwidth.

---

## 11. Phase 8: Evaluation and Ablation

### Experiment Design
- **Datasets:**
  - Synthetic: R-MAT (SCALE 18, 20, 22, 24; edge factor 16).
  - Control Graph: Uniform random Erdős–Rényi or 2D/3D Grid (to confirm optimizations specifically target power-law skew).
  - Real-World Graphs (optional): SNAP datasets (LiveJournal, Twitter, Orkut).
- **Implementations Evaluated (Cumulative Ablation):**
  1. Serial Baseline
  2. Naive OpenMP Parallel BFS
  3. + Test-Before-CAS
  4. + Visited Bitmap
  5. + Thread-Local Buffers & Batched Merge
  6. + Cache-Line Padding & Layout Alignment
  7. + Direction-Optimizing Hybrid (Beamer)
  8. + NUMA First-Touch & Affinity Tuning
- **Metrics Collected:** TEPS, strong scaling speedup, parallel efficiency, atomics/edge, L1/LLC misses, HITM counts, time per level.
- **Benchmark Rigor:**
  - Discard warm-up runs.
  - Execute at least 10 runs per configuration; report median and variance.
  - Compare against established graph frameworks (GAP Benchmark Suite, Ligra).

### Exit Criteria
- Complete tables, strong-scaling plots, and ablation breakdowns confirming performance contributions of each technique.

---

## 12. Phase 9: Report and Documentation

### Final Technical Report Structure
1. **Introduction & Motivation:** Scalability challenges of irregular, scale-free graph traversal.
2. **Background:** Cache coherence protocols (MESI/MOESI), RFO invalidations, atomics cost, false sharing.
3. **Related Work:** Graph500, Beamer direction-optimization, Ligra, GAP Benchmark Suite, Galois.
4. **Architecture & Design:** Memory layout, synchronization strategies, frontier management.
5. **Implementation Details:** Key OpenMP pragmas, data structure alignments, C++ atomic operations.
6. **Experimental Setup:** Machine specifications, test graphs, benchmarking methodology.
7. **Results & Analysis:** Scaling curves, TEPS measurements, `perf c2c` hardware counter corroboration.
8. **Limitations & Future Work:** Multi-GPU exploration, distributed graph systems.
9. **Reproducibility Appendix:** Scripts, compiler flags, seeds, and execution instructions.

---

## 13. Reference Code Architecture (Phases 4 & 5 Combined)

```cpp
#include <omp.h>
#include <vector>
#include <cstdint>
#include <cstring>

struct alignas(64) PerThread {
    uint32_t* buf;
    size_t size;
    size_t cap;
    char pad[64 - (sizeof(uint32_t*) + 2 * sizeof(size_t)) % 64];
};

void top_down_step(
    const uint64_t* row,
    const uint32_t* col,
    int32_t* dist,
    const uint32_t* frontier,
    size_t frontier_size,
    uint32_t* next_frontier,
    size_t& next_frontier_size,
    int32_t level,
    std::vector<PerThread>& tls) 
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        PerThread& me = tls[tid];
        me.size = 0;

        #pragma omp for schedule(dynamic, 64) nowait
        for (size_t i = 0; i < frontier_size; ++i) {
            uint32_t u = frontier[i];
            for (uint64_t e = row[u]; e < row[u + 1]; ++e) {
                uint32_t v = col[e];

                // 1. Test before atomic (plain read keeps line in Shared state)
                if (dist[v] != -1) continue;

                // 2. Atomic claim with relaxed memory order
                int32_t expected = -1;
                if (__atomic_compare_exchange_n(&dist[v], &expected, level + 1,
                                                false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                    // 3. Thread-local buffer append (zero shared queue contention)
                    me.buf[me.size++] = v;
                }
            }
        }
    } // Implicit OpenMP barrier guarantees happens-before memory visibility

    // 4. Batched merge: exclusive prefix sum across thread buffer sizes
    int nthreads = tls.size();
    std::vector<size_t> offsets(nthreads + 1, 0);
    for (int t = 0; t < nthreads; ++t) {
        offsets[t + 1] = offsets[t] + tls[t].size;
    }
    next_frontier_size = offsets[nthreads];

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < nthreads; ++t) {
        if (tls[t].size > 0) {
            std::memcpy(&next_frontier[offsets[t]], tls[t].buf, tls[t].size * sizeof(uint32_t));
        }
    }
}
```

---

## 14. Risks and Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **High Run-to-Run Variance** | Noisy profiling and benchmarks | Pin thread execution (`OMP_PROC_BIND`), fix CPU governor to `performance`, discard initial warm-up runs, report medians over $\ge 10$ runs. |
| **Excessive Memory Overhead from Padding** | Cache capacity blow-up | Pad only per-thread shared headers, local counters, and state structs; never pad individual graph adjacency entries. |
| **Parent Pointer Inconsistency in Benign Races** | Incorrect BFS trees | Maintain a CAS-validated variant as a fallback; validate parent pointers against Graph500 validation rules. |
| **Hub Vertices Causing Load Imbalance** | Straggler threads stall barriers | Implement edge-balanced chunking (partition work by cumulative degree) and adjacency splitting for massive hubs. |
| **Development Machine Lacks Multi-Socket NUMA** | Unable to test NUMA interconnect | Execute single-socket affinity testing (core vs. thread bindings) and test NUMA behaviors using emulated NUMA (`qemu` / Linux `memmap`). |
| **Memory Limits at Large Scales** | Out-of-memory crashes on large graphs | Use 32-bit vertex IDs, compact visited bitmaps, edge factor $\approx 16$, and cap scale to fit available physical RAM. |

---

## 15. Suggested Timeline (8 to 10 Weeks)

```
Week 1    : Phase 0 (Environment, Tooling, Hardware Study) & Phase 1 (Graph Generator & CSR)
Week 2    : Phase 2 (Serial & Naive Parallel Baselines, Test Harness)
Week 3    : Phase 3 (Hardware Profiling with perf stat / perf c2c)
Weeks 4-5 : Phase 4 (Contention Reduction) & Phase 5 (Eliminate False Sharing)
Week 6    : Phase 6 (Direction-Optimizing Hybrid BFS)
Week 7    : Phase 7 (NUMA First-Touch & Thread Affinity Tuning)
Week 8    : Phase 8 (Comprehensive Benchmark Suite & Ablation Studies)
Weeks 9-10: Phase 9 (Final Technical Report, Reproducibility Package, Buffer)
```
