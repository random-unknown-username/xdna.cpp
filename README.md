# xdna.cpp

An experimental GGML backend that runs Q4_0 decode GEMV on the NPU inside AMD
Ryzen 7040 (Phoenix) and 8040 (Hawk Point) APUs. Everything else in llama.cpp
stays where it is: model loading, attention, norms, sampling and any op the
backend does not claim are executed on the CPU as usual.

It is a plugin, not a fork. You build it against a llama.cpp checkout, point
llama.cpp at the resulting `libggml-xdna.so`, and the NPU shows up as a normal
ggml device.

```
$ GGML_BACKEND_PATH=$PWD/build/libggml-xdna.so llama-cli ... --device XDNA -ngl 99
```

## Status

Working and tested on one machine: a Ryzen 7 250 with 16 AIE2 compute tiles,
Arch Linux, `amdxdna` and XRT 2.27.

What works today:

- Q4_0 `GGML_OP_MUL_MAT` offload, which is the bulk of decode work
- Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, IQ3_S, IQ3_XXS and IQ4_XS weights on the same path
- Bounded DMA staging, so model size does not decide how much memory gets pinned
- Dynamic loading through ggml's backend registry

What does not:

- Prefill. The backend only accepts a single-row activation matrix, so prompt
  processing stays on the CPU and the `pp512` numbers below are CPU numbers.
- Token embeddings and the Qwen output projection (`K` not a multiple of the
  block size). They are rejected and fall back.
- Anything other than XDNA1. No XDNA2/Strix support yet.

## How it works

The NPU has 16 compute tiles behind four shim columns. For a supported matmul,
weights are read from the memory-mapped GGUF, copied through a pair of 128 MiB
staging buffers, and pushed over XRT DMA to the tiles. The AIE2 kernel unpacks
the 4-bit values, converts the Q4_0 block scale to BF16, and does the multiply
in registers. Results come back as F32 and llama.cpp carries on.

```
mmap'd GGUF  ->  2 x 128 MiB staging BOs  ->  XRT DMA  ->  16 AIE2 tiles
                                                              |
                                          unpack, BF16 scale, MAC
                                                              |
                                                      F32 rows out
```

Weights are not turned into persistent XRT buffer objects. An earlier version
did that, and on large models it pinned the whole weight set as unevictable
kernel memory; the page cache then had nowhere to go. The staging buffers are
allocated once and reused, so the pinned footprint is fixed at 256 MiB no
matter how big the model is.

Small matrices are not worth a dispatch. The backend has a size cutoff below
which it reports the op as unsupported and the CPU keeps it, which is why a
0.5B model and a 3B model behave so differently.

## Benchmarks

Measured on the machine described above, llama.cpp build `0ec4027a2` (12020),
gcc 16.2.1. Take these as one data point, not a hardware spec.

### Host <-> NPU DMA (`build/xdna-bench`)

| Buffer | H2D | D2H | Round trip |
| ---: | ---: | ---: | ---: |
| 4 KiB | 29.05 GB/s | 29.26 GB/s | 14.58 GB/s |
| 64 KiB | 87.26 GB/s | 87.26 GB/s | 43.63 GB/s |
| 1 MiB | 101.02 GB/s | 101.02 GB/s | 50.51 GB/s |
| 16 MiB | 101.98 GB/s | 102.02 GB/s | 51.00 GB/s |
| 128 MiB | 99.52 GB/s | 99.69 GB/s | 49.80 GB/s |

DMA latency per transfer is in the 0.1 us range for small buffers and around
1.35 ms for 128 MiB. Throughput flattens out above roughly 512 KiB.

### Sustained streaming (`build/xdna-bandwidth-bench -m transfer -d 10`)

```
5096 iterations in 10 s
effective throughput   68.40 GB/s
p50 / p90 / p99        1.915 / 2.322 / 3.938 ms per 128 MiB chunk
```

On-tile compute, same tool with `-m compute -d 10`: **208.8 GMACs/s** across
all 16 tiles.

### Q4_0 GEMV (`-m gemv`)

Each case compares the NPU result against the scalar CPU reference.

| N x K | Time | Throughput | Cosine | Max abs error |
| ---: | ---: | ---: | ---: | ---: |
| 4096 x 4096 | 0.209 ms | 39.41 GB/s | 1.00000000 | 3.91e-05 |
| 8192 x 8192 | 0.866 ms | 40.41 GB/s | 1.00000000 | 6.68e-05 |
| 17408 x 5120 | 1.286 ms | 36.67 GB/s | 1.00000000 | 5.15e-05 |

The floats differ because the kernel works in BF16; the cosine similarity
rounds to 1.0 at eight decimals in all three.

### End to end, Qwen2.5 Q4_0 (`llama-bench -p 512 -n 128 -r 5 -t 8`)

| Model | Backend | pp512 | tg |
| --- | --- | ---: | ---: |
| 0.5B | CPU | 87.16 ± 1.71 | 55.10 ± 0.76 tok/s |
| 0.5B | XDNA | 88.07 ± 0.47 | 56.03 ± 1.80 tok/s |
| 3B | CPU | 17.80 ± 0.47 | 13.34 ± 0.79 tok/s |
| 3B | XDNA | 18.03 ± 0.02 | 12.89 ± 0.93 tok/s |

The honest summary: on this chip, decode lands within a few percent of an
8-thread CPU, not far ahead of it. The 3B model is marginally slower on the NPU
and the 0.5B marginally faster. Prefill is CPU work and is unaffected.

Where it does help is headroom. During NPU decode the CPU is mostly idle, and
output is token-for-token identical to the CPU path: 32 tokens of greedy
completion on the 3B model match exactly, and 24 tokens on the 0.5B do too.
If you are running other work alongside inference, that matters more than the
raw tok/s number.

Earlier versions of this README claimed a 68% gain at 3B. That was not
reproducible after the backend was rebuilt and cleaned up, so it is gone.

## Building

Requirements: a Phoenix or Hawk Point APU, Linux with `amdxdna`, a working
`/dev/accel/accel0`, XRT 2.18+, a C++17 compiler, CMake 3.20+, OpenMP. Your
user needs access to the `render` group.

```bash
git clone https://github.com/random-unknown-username/xdna.cpp.git
cd xdna.cpp

cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DXDNA_LLAMA_CPP_DIR=/path/to/llama.cpp

cmake --build build -j$(nproc)
```

`XDNA_LLAMA_CPP_DIR` is only needed for the plugin and the tools that exercise
it. Without it the core library, probes and tests still build. Point it at a
llama.cpp checkout that has `ggml/include/ggml.h`.

Check the device:

```bash
./build/xdna-info
```

Run the tests:

```bash
ctest --test-dir build --output-on-failure
```

## Running a model

The plugin is a drop-in ggml backend, so any llama.cpp binary with dynamic
backend loading works. No rebuild of llama.cpp is needed.

```bash
export GGML_BACKEND_PATH=$PWD/build/libggml-xdna.so

llama-cli -m model.gguf \
    --device XDNA -ngl 99 \
    -t 8 -cnv
```

Drop `--device XDNA` to compare against the CPU. `libggml-xdna.so` needs the
XRT libraries on the loader path, normally `/opt/xilinx/xrt/lib`.

## Tools

| Binary | What it does |
| --- | --- |
| `xdna-info` | device discovery from sysfs, the amdxdna ioctl and XRT |
| `xdna-bench` | DMA latency and bandwidth sweep |
| `xdna-bandwidth-bench` | sustained streaming, on-tile compute, GEMV, ggml graph |
| `xdna-kernel-bench` | per-tensor GEMV against the CPU reference |
| `xdna-gguf-info` | GGUF metadata and tensor table dump |
| `xdna-gate1` | single-kernel hardware bring-up test |

## Layout

```
include/xdna/   headers: DMA streamer, GEMV engine, quant blocks, device probe
src/            backend, streaming, kernels, reference implementations
tools/          the binaries above
tests/          CTest suite
assets/phx_bins prebuilt AIE binaries used by the probe and bench tools
```

## License

MIT. See `THIRD_PARTY_NOTICES.md` for llama.cpp/ggml, AMD IRON, MLIR-AIE and
XRT attribution.
