# xdna.cpp

Run GGUF quantized models directly on AMD XDNA1 NPUs (Ryzen AI 7040 / 8040 series) with llama.cpp.

No theoretical TFLOPS hype. Just real AIE2 vector execution, bounded staging that doesn't kill your RAM, and honest benchmarks.

---

## What is this?

`xdna.cpp` is an open-source GGML backend for AMD XDNA1 Neural Processing Units. It offloads supported `Q4_0` matrix multiplications to 16 AIE2 vector compute tiles via bounded DRM DMA streaming, while letting the host Zen 4 CPU handle non-GEMV parts (norms, attention softmax, DeltaNet recurrence).

```
   GGUF Q4_0 Weights (DDR)
             │
             ▼ (Bounded 128 MB Ping-Pong BOs)
   ┌────────────────────────────────────────────────┐
   │ AMD XDNA1 NPU (4 Columns × 4 Rows = 16 Tiles)  │
   │   - Fused uint4 unpack → BF16 scale → MAC      │
   │   - Sustained ~32–38 GB/s streaming throughput │
   └────────────────────────────────────────────────┘
             │
             ▼
   Host CPU (Norms, Attention, Sampling) → Output Tokens
```

---

## Benchmark Scoreboard

Measured on physical hardware (AMD Ryzen 7 7840U / 8840HS, 32 GB LPDDR5, Linux 6.x):

| Model | Parameters | Quant Format | CPU Baseline (llama.cpp, 8T) | XDNA Hybrid Backend (4T) | Speedup / Delta | Operational Behavior |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B** | 630 M | `Q4_0` | **106.5 ± 4.9 tok/s** | **72.6 ± 1.9 tok/s** | CPU wins | CPU favored for tiny 60 KB payloads |
| **Qwen2.5-3B** | 3.4 B | `Q4_0` | **9.8 ± 0.4 tok/s** | **16.5 ± 1.3 tok/s** | **+68.8% (1.69×)** | XDNA decisively accelerates decode |
| **Qwen3.8-27B** | 27.3 B | `Q4_0` | — *(not measured)* | **1.72 ± 0.05 tok/s** | Bounded DDR stream | 12.6 GiB Q4 path streamed on 32GB RAM |

*Note: The 72.6 tok/s and 16.5 tok/s results reflect the **XDNA hybrid backend**, where large linear projections stream to the 16 AIE2 tiles while tiny sub-matrices ($N \le 256$) execute via host SIMD to minimize driver launch overhead.*

---

## The Engineering Narrative: 0.15 → 1.72 tok/s on 27B

When first running `Qwen3.8-27B` (15.2 GB), inference collapsed to **0.15 tok/s (6.67 s/token)**.

A forensic `/proc/meminfo` audit revealed why: allocating persistent XRT Buffer Objects (BOs) for all 369 tensors locked **12.6 GB into `Unevictable` kernel DMA memory**. On a 32 GB laptop, this starved Linux of page cache, causing 520,000 major page faults and heavy swap thrashing.

`xdna.cpp` solved this with a **bounded double-buffered staging ring**:
1. Keeps weights mapped in standard page cache (`MAP_SHARED`).
2. Streams weights through alternating $2 \times 64\text{ MB}$ staging BOs (**128 MB total pinned memory**).
3. Memory pressure and swap dropped to zero, recovering throughput from **0.15 → 1.72 tok/s (an 11.5× speedup)**.

For full architectural details, see [`src/ARCHITECTURE.md`](src/ARCHITECTURE.md).

---

## Numerical Quality Verification

Verified across isolated separate processes comparing raw float32 logits against the pure CPU reference baseline:

| Model | Evaluation Sequence | Top-1 Token Match | Logit Cosine Similarity | Relative L2 Delta | Quality Status |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B** | 32 Tokens (151k Vocab) | **32 / 32 (100%)** | **0.999511** | **3.12 %** | High numerical similarity |
| **Qwen2.5-3B** | 16 Tokens (151k Vocab) | **16 / 16 (100%)** | **0.997227** | **7.08 %** | High numerical similarity |
| **Qwen3.8-27B** | 16 Tokens (248k Vocab) | **16 / 16 (100%)** | **0.999906** | **1.41 %** | High numerical similarity |

*Logit deltas arise from rounding FP16 quantization scales to BF16 for AIE2 hardware vector units. We describe this as high numerical similarity in tested deterministic sequences.*

---

## Quick Start

### 1. Build llama.cpp with the XDNA plugin

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build -S . \
    -DGGML_XDNA=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

### 2. Run Interactive Chat

```bash
./build/bin/llama-cli \
    -m /path/to/qwen2.5-3b-instruct-q4_0.gguf \
    --device XDNA0 \
    -cnv \
    -t 4
```

---

## Hardware & Driver Requirements

- **Processor**: AMD Ryzen 7040 Series ("Phoenix") or Ryzen 8040 Series ("Hawk Point")
- **Driver**: Linux `amdxdna` DRM driver (`/dev/accel/accel0`)
- **Runtime**: AMD XRT 2.18+ (`/opt/xilinx/xrt`)
- **Permissions**: User in `render` group (`sudo usermod -a -G render $USER`)

---

## Roadmap

### v1.1
- Persistent cross-tensor AIE execution (amortizing operator fill/drain overhead)
- True on-device AIE Gate+Up & QKV fusion
- Adaptive CPU vs. XDNA shape crossover routing
- Dedicated prefill GEMM ($M > 1$) acceleration

### v1.2 / Research
- `Q8_0` quantization support
- AMD XDNA2 (Ryzen AI 300 series / Strix Point) 8-column support
- Direct kernel driver userptr zero-copy optimizations

---

## License

Licensed under the [MIT License](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream acknowledgments.
