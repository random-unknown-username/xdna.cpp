# xdna.cpp

`xdna.cpp` is an experimental `llama.cpp` backend for AMD XDNA1 NPUs. It offloads Q4_0 decode GEMV to the 16 AIE2 tiles available on Phoenix and Hawk Point APUs, while keeping model loading, attention, normalization, sampling, and unsupported operations on the normal llama.cpp path.

The backend is currently focused on Ryzen 7040 and 8040 series APUs running Linux.

## Quick Start

### Pre-built release

The Linux x86_64 release includes `llama-cli`, `llama-bench`, `llama-server`, and `libggml-xdna.so`.

```bash
wget https://github.com/random-unknown-username/xdna.cpp/releases/download/v1.0.0/xdna-llama-linux-x86_64.tar.gz
tar -xzf xdna-llama-linux-x86_64.tar.gz
cd xdna-llama-linux-x86_64
```

Download a Q4_0 model:

```bash
wget -c https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
```

Run it on the XDNA backend:

```bash
./run-chat.sh qwen2.5-0.5b-instruct-q4_0.gguf
```

## How It Works

Phoenix and Hawk Point expose 16 AIE2 compute tiles arranged as 4 columns by 4 rows.

`xdna.cpp` registers as a GGML backend and handles supported `GGML_OP_MUL_MAT` operations using Q4_0 weights.

For each supported operation:

1. Q4_0 weights are read from the normal mmap backed GGUF.
2. Weights are copied through a bounded host staging buffer.
3. XRT DMA transfers them to the NPU.
4. The AIE2 kernel unpacks the 4 bit values, applies the Q4_0 scale in BF16, and executes the vector MAC.
5. Output activations return to host memory and llama.cpp continues execution.

```text
GGUF Q4_0 weights
        |
        v
file backed mmap
        |
        v
2 x 64 MiB staging BOs
        |
        v
XRT DMA
        |
        v
16 AIE2 tiles
        |
        v
Q4_0 unpack
BF16 scaling
vector MAC
        |
        v
llama.cpp CPU path
```

The larger streamed matrices currently sustain roughly 32 to 38 GB/s through the NPU path.

## CPU and NPU Crossover

Not every supported matrix is worth sending to the NPU.

Small projections can complete on the CPU faster than the XRT dispatch and synchronization overhead required to execute them on XDNA. The backend therefore uses a crossover policy and leaves smaller operations on the CPU.

This is visible on Qwen2.5 0.5B:

```text
llama.cpp CPU, 8 threads
106.5 ± 4.9 tok/s

XDNA hybrid backend
72.6 ± 1.9 tok/s
```

At 3B the larger projections give the NPU enough work for the offload to pay off:

```text
Qwen2.5 3B Q4_0

llama.cpp CPU, 8 threads
9.8 ± 0.4 tok/s

XDNA hybrid backend
16.5 ± 1.3 tok/s
```

That is about a 68.8% improvement over the measured CPU baseline.

## Large Model Streaming

The largest model tested so far is Qwen3.8 27B Q4_0:

```text
Parameters        27.3B
Streamed weights  ~12.6 GiB
Decode speed      1.72 ± 0.05 tok/s
```

The model weights are not permanently allocated as XRT buffer objects.

An earlier implementation created persistent `xrt::bo` allocations for model tensors. On the 27B model this pinned roughly 12.6 GB as unevictable kernel memory, heavily reducing available page cache and causing swap traffic.

Decode performance dropped to roughly:

```text
0.15 tok/s
```

The current runtime instead keeps weights in normal file backed memory and allocates two reusable 64 MiB host staging BOs:

```text
staging BO 0    64 MiB
staging BO 1    64 MiB

total pinned   128 MiB
```

Weights are streamed through the two buffers during layer execution, keeping the pinned allocation fixed regardless of total model size.

This is what allows the 27B model to stream roughly 12.6 GiB of weights while keeping only 128 MiB permanently pinned for staging.

## Numerical Verification

XDNA output has been compared against separate pure CPU reference runs using logit similarity and token agreement.

### Qwen2.5 0.5B Q4_0

```text
Evaluated tokens        32
Top 1 match             32 / 32
Logit cosine            0.999511
Relative L2 delta       3.12%
```

### Qwen2.5 3B Q4_0

```text
Evaluated tokens        16
Top 1 match             16 / 16
Logit cosine            0.997227
Relative L2 delta       7.08%
```

### Qwen3.8 27B Q4_0

```text
Evaluated tokens        16
Top 1 match             16 / 16
Logit cosine            0.999906
Relative L2 delta       1.41%
```

The XDNA path is not expected to be bit identical to the CPU implementation because the AIE2 kernel uses BF16 internally. All evaluated samples still matched the CPU reference on the top 1 token.

## Building from Source

### 1. Build xdna.cpp

```bash
git clone https://github.com/random-unknown-username/xdna.cpp.git
cd xdna.cpp

cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Probe the XDNA device:

```bash
./build/xdna-cli probe
```

Run the test suite:

```bash
ctest --test-dir build --output-on-failure
```

### 2. Build llama.cpp with XDNA support

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build -S . \
    -DGGML_XDNA=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

Run a model using the XDNA device:

```bash
./build/bin/llama-cli \
    -m /path/to/model.gguf \
    --device XDNA0 \
    -cnv \
    -t 8
```

## Requirements

The current backend requires:

- AMD Ryzen 7040 Phoenix or Ryzen 8040 Hawk Point APU
- Linux with the `amdxdna` driver
- `/dev/accel/accel0`
- AMD XRT 2.18 or newer
- access to the `render` group
- CMake and a C++ compiler

Add the current user to the render group if required:

```bash
sudo usermod -a -G render $USER
```

Log out and back in after changing group membership.

## Current Support

The optimized path currently targets:

```text
Operation        GGML_OP_MUL_MAT
Weight format    Q4_0
Primary use      decode GEMV
Hardware         XDNA1 / AIE2
APUs             Phoenix / Hawk Point
```

Unsupported GGML operations remain on the normal host backend.

The current work is mostly around decode performance. More quant formats, prefill acceleration, lower dispatch overhead, and better DMA overlap still need work.

## License

MIT.

See `THIRD_PARTY_NOTICES.md` for llama.cpp and AMD acknowledgements.
