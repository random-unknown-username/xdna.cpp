# xdna.cpp

AMD put a whole NPU in Phoenix and Hawk Point laptops, so I wanted to see if we could actually use the thing for local LLM inference.

`xdna.cpp` is an experimental XDNA1 backend for `llama.cpp`. It runs Q4_0 GEMV operations on the AMD NPU while leaving the rest of the model on the normal llama.cpp CPU path.

It is not a new inference runtime and it is not trying to replace llama.cpp. It just plugs into GGML as another backend.

Right now the main focus is getting useful decode performance out of the 16 AIE2 tiles found in Ryzen 7040 and 8040 APUs.

## What works

Currently working:

* XDNA1 / AIE2 execution through XRT
* all 16 AIE2 tiles
* Q4_0 dequant + GEMV on the NPU
* GGML backend integration
* normal GGUF models
* bounded weight staging for models much larger than the NPU memory
* CPU / NPU crossover so tiny matmuls don't get pointlessly offloaded
* tested from Qwen 0.5B up to Qwen 27B
* numerical comparisons against the normal CPU path

The backend mainly intercepts supported `GGML_OP_MUL_MAT` operations.

Stuff like attention, normalization, sampling and unsupported operations still runs on the host.

So this is a hybrid backend rn, not full model execution on the NPU.

## Performance

Tested on Ryzen 7 7840U / 8840HS machines with 32 GB RAM on Linux.

```text
Qwen2.5 0.5B Q4_0

llama.cpp CPU, 8 threads
106.5 ± 4.9 tok/s

XDNA hybrid
72.6 ± 1.9 tok/s
```

The NPU loses here.

That is actually expected. Tiny projections finish so quickly on the CPU that XDNA dispatch and synchronization overhead ends up costing more than the compute itself.

For a larger model:

```text
Qwen2.5 3B Q4_0

llama.cpp CPU, 8 threads
9.8 ± 0.4 tok/s

XDNA hybrid
16.5 ± 1.3 tok/s
```

Around a 68.8% improvement over the CPU baseline.

And the largest test so far:

```text
Qwen3.8 27B Q4_0

27.3B parameters
~12.6 GiB of streamed weights
1.72 ± 0.05 tok/s
```

The 27B run is probably the more interesting result.

The model obviously does not fit inside some tiny dedicated NPU memory pool. The weights stay mmap'd normally and get streamed through a fixed staging area when needed.

## The memory problem

The first version created persistent XRT buffer objects for model tensors.

That worked on small models and then completely fell apart on the 27B test.

Around 12.6 GiB of weights ended up pinned as unevictable memory. Linux lost a huge amount of usable page cache, started swapping heavily and decode dropped to roughly:

```text
0.15 tok/s
```

So instead of keeping every tensor inside an XRT buffer, the current backend keeps the model file backed as normal memory and uses two staging buffers:

```text
64 MiB buffer A
64 MiB buffer B

128 MiB pinned total
```

Weights are copied through these buffers as layers execute.

Same 12.6 GiB model, but only 128 MiB needs to stay pinned for staging.

That brought the 27B run up to around 1.72 tok/s without nuking system memory.

## How the NPU path works

For supported Q4_0 matmuls the flow is roughly:

```text
GGUF Q4_0 weights
        |
        v
normal mmap / page cache
        |
        v
128 MiB staging buffers
        |
        v
XDNA DMA
        |
        v
16 AIE2 tiles
        |
        v
uint4 unpack
BF16 scaling
vector MAC
        |
        v
host memory
        |
        v
rest of llama.cpp
```

The AIE kernel unpacks the 4 bit weights, applies the Q4_0 block scale and performs the matrix vector work using the AIE vector units.

The result gets sent back to the host and llama.cpp continues normally.

## CPU vs NPU

One thing that became obvious pretty quickly is that offloading every supported operation is not automatically faster.

Small matmuls are often better left on the CPU.

Large ones are where the NPU starts making sense.

So xdna.cpp uses a crossover instead of blindly throwing every matrix at XDNA.

The goal is tokens/sec, not getting a nice looking "100% NPU" number.

## Correctness

The NPU outputs were also compared against separate pure CPU runs.

```text
Qwen2.5 0.5B

32 / 32 top 1 tokens matched
logit cosine: 0.999511
relative L2: 3.12%
```

```text
Qwen2.5 3B

16 / 16 top 1 tokens matched
logit cosine: 0.997227
relative L2: 7.08%
```

```text
Qwen3.8 27B

16 / 16 top 1 tokens matched
logit cosine: 0.999906
relative L2: 1.41%
```

There are small differences from the CPU path because the AIE implementation goes through BF16 internally, but all evaluated tokens matched the CPU top 1 output.

## Quick start

There is a prebuilt Linux x86_64 release with `llama-cli`, `llama-bench`, `llama-server` and the XDNA backend.

```bash
wget https://github.com/random-unknown-username/xdna.cpp/releases/download/v1.0.0/xdna-llama-linux-x86_64.tar.gz

tar -xzf xdna-llama-linux-x86_64.tar.gz
cd xdna-llama-linux-x86_64
```

Grab a small Q4_0 model for testing:

```bash
wget -c https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
```

Run it:

```bash
./run-chat.sh qwen2.5-0.5b-instruct-q4_0.gguf
```

## Building

You currently need:

* Ryzen 7040 Phoenix or Ryzen 8040 Hawk Point
* Linux
* `amdxdna`
* `/dev/accel/accel0`
* AMD XRT 2.18+
* access to the `render` group
* CMake
* a C++ compiler

Clone and build:

```bash
git clone https://github.com/random-unknown-username/xdna.cpp.git
cd xdna.cpp

cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Check that the NPU is visible:

```bash
./build/xdna-cli probe
```

Run the tests:

```bash
ctest --test-dir build --output-on-failure
```

For llama.cpp, build with the XDNA backend enabled:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build -S . -DGGML_XDNA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Then run a Q4_0 model with the XDNA device:

```bash
./build/bin/llama-cli -m /path/to/model.gguf --device XDNA0 -cnv -t 8
```

## Current limitations

This is still pretty early.

The current fast path is mainly Q4_0 decode GEMV. It is not a general purpose accelerator for every GGML operation yet.

Phoenix and Hawk Point are the targets rn. Other XDNA generations have not been properly supported or tested.

Prefill is also not the main focus yet.

There is still a lot left to mess with, especially better overlap between DMA and compute, more quant formats, lower dispatch overhead and figuring out how far larger models can be pushed.

## Why I made this

Mostly curiosity.

These NPUs are already sitting inside a ton of Ryzen laptops and there really isn't much in the local LLM space using them directly.

I didn't want to make another completely separate inference runtime just to prove it could run a matrix multiply.

Using a llama.cpp backend means GGUF loading, model support, sampling and everything else can stay where it already works, while the parts that make sense for the NPU can move over gradually.

The first few versions were pretty cursed, especially the one casually pinning 12 GB of RAM, but it works now and there is a lot more stuff worth trying.

If you have a Phoenix or Hawk Point machine and manage to run this on it, results are welcome.

## License

MIT.

See `THIRD_PARTY_NOTICES.md` for the upstream llama.cpp and AMD acknowledgements.
