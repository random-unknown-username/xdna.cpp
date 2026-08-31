# xdna.cpp

Using AMD's hawkpoint NPUs to run LLMs! It's a llama.cpp plugin (not a standalone runtime) to keep full model compatibility, with custom Q4_0 dequant/GEMV kernels written for the XDNA1 / AIE2 architecture.

---

## Quick Start

### Pre-built release (Linux x86_64)

Download the release bundle which includes `llama-cli`, `llama-bench`, `llama-server`, and `libggml-xdna.so`:

```bash
wget https://github.com/random-unknown-username/xdna.cpp/releases/download/v1.0.0/xdna-llama-linux-x86_64.tar.gz
tar -xzf xdna-llama-linux-x86_64.tar.gz
cd xdna-llama-linux-x86_64

# download a test model (Qwen 0.5B)
wget -c https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf

# run chat on NPU
./run-chat.sh qwen2.5-0.5b-instruct-q4_0.gguf
```

---

## How It Works

Phoenix and Hawk Point APUs (Ryzen 7040 / 8040 series) include an NPU array with 16 AIE2 tiles (4 columns x 4 rows).

`xdna.cpp` registers as a dynamic GGML backend (`libggml-xdna.so`) that intercepts `GGML_OP_MUL_MAT` operations on `Q4_0` tensors. When a layer executes:
1. Weights are streamed via DMA into the 16 AIE2 tiles.
2. The kernel unpacks 4-bit nibbles to BF16, applies block delta scaling, and executes vector MAC operations.
3. Output activations return to host memory for attention, normalization, and sampling.

```
   GGUF Q4_0 Weights (mmap)
             │
             ▼ (Bounded 128 MB Staging BOs)
   ┌────────────────────────────────────────────────┐
   │ AMD XDNA1 NPU (4 Columns × 4 Rows = 16 Tiles)  │
   │   - uint4 unpack → BF16 scale → vector MAC     │
   │   - ~32–38 GB/s sustained streaming throughput │
   └────────────────────────────────────────────────┘
             │
             ▼
   Host CPU (Norms, Attention, Sampling) → Tokens
```

---

## Benchmarks

Measured on AMD Ryzen 7 7840U / 8840HS (32 GB LPDDR5, Linux 6.x):

| Model | Parameters | Quant | CPU Baseline (llama.cpp, 8T) | XDNA Hybrid Backend (8T) | Notes |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 630 M | Q4_0 | 106.5 ± 4.9 tok/s | 72.6 ± 1.9 tok/s | CPU favored for small 60 KB projections |
| Qwen2.5-3B | 3.4 B | Q4_0 | 9.8 ± 0.4 tok/s | 16.5 ± 1.3 tok/s | +68.8% speedup over CPU baseline |
| Qwen3.8-27B | 27.3 B | Q4_0 | — *(not measured)* | 1.72 ± 0.05 tok/s | 12.6 GiB streamed via bounded staging |

---

## Bounded Staging Architecture

Large models (like 27B) have hundreds of weight matrices totaling ~12.6 GiB. Statically creating persistent XRT Buffer Objects (`xrt::bo`) for every tensor pins all 12.6 GB into `Unevictable` kernel memory, starving Linux of evictable page cache and triggering heavy disk swap thrashing (dropping decode speed to 0.15 tok/s).

`xdna.cpp` solves this with a **bounded double-buffered staging ring**:
- Weights remain in normal file-backed page cache (`MAP_SHARED`).
- The runtime allocates exactly two 64 MB host-only staging BOs (**128 MB total pinned RAM**).
- Weights stream through the alternating buffers into the NPU DMA channels during layer evaluation.
- Page faults and swap drop to zero, sustaining **1.72 tok/s** on consumer laptops.

---

## Numerical Verification

Verified against pure CPU IEEE 754 reference outputs using binary logit comparisons across separate processes:

| Model | Evaluated Tokens | Top-1 Match | Logit Cosine Similarity | Relative L2 Delta |
|---|---|---|---|---|
| Qwen2.5-0.5B | 32 Tokens | 32 / 32 (100%) | 0.999511 | 3.12 % |
| Qwen2.5-3B | 16 Tokens | 16 / 16 (100%) | 0.997227 | 7.08 % |
| Qwen3.8-27B | 16 Tokens | 16 / 16 (100%) | 0.999906 | 1.41 % |

---

## Building from Source

### 1. Build xdna.cpp

```bash
git clone https://github.com/random-unknown-username/xdna.cpp.git
cd xdna.cpp

cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/xdna-cli probe
ctest --test-dir build --output-on-failure
```

### 2. Build llama.cpp with XDNA backend

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build -S . -DGGML_XDNA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

./build/bin/llama-cli -m /path/to/model.gguf --device XDNA0 -cnv -t 8
```

---

## Requirements

- AMD Ryzen 7040 ("Phoenix") or 8040 ("Hawk Point") APU.
- Linux kernel with `amdxdna` driver (`/dev/accel/accel0`).
- AMD XRT 2.18+ (`/opt/xilinx/xrt`).
- User in `render` group (`sudo usermod -a -G render $USER`).

---

## License

MIT License. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream llama.cpp and AMD acknowledgments.
