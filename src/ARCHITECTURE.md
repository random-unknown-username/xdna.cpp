# xdna.cpp — Architecture & Memory Systems

This document explains the runtime architecture, memory subsystem, and execution flow of `xdna.cpp`.

---

## 1. High-Level Flow

`xdna.cpp` operates as a dynamic GGML backend plugin (`libggml-xdna.so`) loaded by `llama.cpp`:

```
   llama.cpp cgraph compute
              │
              ├── Non-GEMV Ops (RMSNorm, Softmax, RoPE, Sampling) ──► Host Zen 4 CPU (SIMD)
              │
              └── GGML_OP_MUL_MAT (Q4_0 Projections) ──────────────► ggml_backend_xdna_graph_compute
                                                                             │
                                                                             ▼
                                                                 Bounded 128 MB Staging Ring
                                                                 (2 × 64 MB Host-Only BOs)
                                                                             │
                                                                             ▼ (DRM DMA)
                                                                 AMD XDNA1 NPU (AIE2 Array)
                                                                 (4 Columns × 4 Rows = 16 Tiles)
```

---

## 2. The Memory Subsystem: Solving the 12.6 GB BO Bottleneck

### The Pathology
When running large models (like `Qwen3.8-27B`, ~15.2 GB), allocating static persistent `xrt::bo` objects for all 369 weight tensors locked **12.6 GB into `Unevictable` kernel DMA memory**. On a 32 GB machine, this starved Linux of evictable page cache, triggering direct memory reclaim and swapping (520k major page faults), collapsing inference throughput to **0.15 tok/s**.

### The Solution: Bounded Double-Buffered Staging Ring
1. Model weights remain memory-mapped directly from the `.gguf` file via standard Linux page cache (`MAP_SHARED`).
2. The runtime allocates exactly **two 64 MB host-only staging BOs (128 MB total pinned memory)**.
3. During layer-by-layer inference, tensor payloads stream through the alternating staging buffers into the NPU DMA channels.
4. Memory pressure drops to zero (`pgmajfault ≈ 0`, `pswpout = 0`), restoring full DDR streaming performance to **1.72 tok/s** (an **11.5× speedup**).

---

## 3. AIE2 Tile Compute & Dequantization

- **Array Layout**: 4 Columns $\times$ 4 Rows = 16 Compute Tiles.
- **Inner Loop**:
  - Unpacks 4-bit nibbles into 8-bit integers (`aie::unpack`).
  - Subtracts block offset 8 and converts to BF16.
  - Multiplies by block scale (FP16 scale rounded to BF16).
  - Performs fused vector multiply-accumulate (MAC) across 32-element blocks.

---

## 4. Heterogeneous Crossover Scheduling

Because tiny matrix operations (e.g. 64 KB attention projections in 0.5B models) pay host driver dispatch overhead, `xdna.cpp` utilizes a hybrid execution strategy:
- **Small Projections ($N \le 256$)**: Processed via unrolled host SIMD to avoid synchronization latency.
- **Medium & Large Projections ($N \ge 896$)**: Offloaded to the 16 AIE2 compute tiles to saturate hardware DMA pipelines and free host CPU resources.
