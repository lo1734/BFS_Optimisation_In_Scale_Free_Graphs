# Baseline Hardware Topology & Coherence Profile

## 1. System Topology Overview

This baseline profile documents the host development architecture and contrasts it with traditional x86-64 SMP/NUMA environments targeted by OpenMP BFS implementations.

### Host Machine Specifications (Development Node)
- **Model:** Apple Silicon M1 (ARMv8.5-A architecture)
- **Total Cores:** 8 physical cores (heterogeneous big.LITTLE topology)
  - **Performance (P) Cluster ("Firestorm"):** 4 cores @ up to 3.2 GHz
  - **Efficiency (E) Cluster ("Icestorm"):** 4 cores @ up to 2.06 GHz
- **Unified Memory (RAM):** 8 GB LPDDR4X (unified memory architecture, ~68 GB/s bandwidth)
- **Cache Hierarchy:**
  - **Cache Line Size:** **128 bytes** (distinct from x86's standard 64 bytes)
  - **P-Cluster L1 Instruction Cache:** 192 KB per core
  - **P-Cluster L1 Data Cache:** 128 KB per core
  - **P-Cluster L2 Shared Cache:** 12 MB shared across the 4 Firestorm cores
  - **E-Cluster L1 Instruction Cache:** 128 KB per core
  - **E-Cluster L1 Data Cache:** 64 KB per core
  - **E-Cluster L2 Shared Cache:** 4 MB shared across the 4 Icestorm cores
  - **System Level Cache (SLC):** 8 MB shared system-level cache

---

## 2. Cache Coherence Protocols: MESI, MOESI, and Apple Silicon Fabric

### Traditional Cache Coherence Protocols
In symmetric multiprocessing (SMP) and multi-socket NUMA systems:
1. **MESI (Modified, Exclusive, Shared, Invalid):**
   - **Modified (M):** Cache line is present only in the current cache and is dirty (must be written back to memory upon eviction).
   - **Exclusive (E):** Cache line is present only in the current cache and is clean.
   - **Shared (S):** Cache line is present in one or more caches and is clean. Read-only.
   - **Invalid (I):** Cache line is not valid.
   - **RFO (Read-For-Ownership):** When a core attempts to write to a cache line currently in `Shared` state, it broadcasts an RFO request across the coherence bus. All other cores invalidate their local copies (`S` $\to$ `I`), and the writing core transitions the line to `Modified` (`M`).
2. **MOESI (Owned state added):**
   - **Owned (O):** Allows sharing of dirty cache lines without writing back to main memory immediately. Used extensively in AMD processors.
3. **MESIF (Forward state added):**
   - **Forward (F):** A designated core responds to read requests on behalf of other shared holders. Used in Intel processors.

### Coherence Dynamics on Apple Silicon M1
- Apple M1 features a point-to-point coherent interconnect fabric linking the P-cluster, E-cluster, GPU, and memory controllers.
- Within a cluster, cores snoop and share the unified L2 cache with very low latency.
- Across clusters (e.g., Thread 0 on P-core contending with Thread 4 on E-core), cache lines in L1/L2 must be invalidated via the fabric. Cross-cluster invalidations incur substantially higher latency (~4x to 8x higher than intra-cluster L2 hits).
- **Critical Architectural Parameter:** The cache line size is **128 bytes**. Any data structure padded only to 64 bytes will still experience **false sharing** across adjacent 64-byte halves on Apple Silicon! All structures in this project are therefore aligned to `max(64, CACHE_LINE_SIZE) = 128` bytes to ensure zero false sharing on both ARM64 and x86 architectures.

---

## 3. Memory Models: ARM Weak Memory Ordering vs. x86 TSO

### x86-64 (Total Store Order - TSO)
- All writes are globally visible in program order (store buffer drains to cache in FIFO order).
- Stores cannot pass earlier loads or stores.
- Atomic operations (`lock cmpxchg`, `lock xadd`) issue hardware bus locks or cache line lockouts, enforcing sequential consistency by default.

### ARMv8.5-A (Weakly Ordered Memory Model)
- Loads and stores can be aggressively reordered by out-of-order execution pipelines unless separated by explicit memory barriers (`DMB`, `DSB`, `ISB`) or one-way barriers (`LDAR` / `STLR`).
- In OpenMP BFS:
  - We can leverage `std::memory_order_relaxed` or `__ATOMIC_RELAXED` for atomic vertex claims (`compare_exchange_weak` or benign races).
  - The level-synchronous OpenMP barrier (`#pragma omp barrier` or end of `#pragma omp parallel for`) enforces the required happens-before relationship, guaranteeing that all vertices claimed in level $L$ are globally visible to all threads before level $L+1$ commences.
  - Relaxed memory order eliminates expensive memory fences on ARM architectures while preserving full algorithmic correctness.

---

## 4. Hardware Counter Profiling Reference (Linux vs. macOS)

### Profiling on Linux Deployments
On Linux test environments (e.g., server nodes with Intel Xeon or AMD EPYC), use the following commands to record and analyze cache coherence bottlenecks:

1. **Hardware Event Counting (`perf stat`):**
   ```bash
   perf stat -e cache-misses,cache-references,LLC-load-misses,L1-dcache-load-misses,instructions,cycles \
     ./bin/naive_bfs data/rmat_s20.bin
   ```
2. **Coherence & False Sharing Detection (`perf c2c`):**
   ```bash
   # Record cache-to-cache HITM (Hit Modified) events
   perf c2c record --call-graph dwarf ./bin/naive_bfs data/rmat_s20.bin

   # Generate HITM report identifying contended cache lines
   perf c2c report --stdio
   ```
   - High `HITM` counts on specific symbol offsets indicate cache lines ping-ponging across CPU sockets.

### Profiling on macOS (Darwin)
- macOS does not expose the Linux `perf` subsystem.
- Profiling is performed using:
  1. High-resolution wall-clock and per-thread micro-timers (`std::chrono::high_resolution_clock`).
  2. Apple `ktrace` / `sysmon` / `xctrace` (Xcode Instruments: Time Profiler, Counters).
  3. Contention microbenchmarks (`./bin/atomic_microbench`) to empirically measure clock cycles and nanoseconds per uncontended vs. contended atomic operation.
