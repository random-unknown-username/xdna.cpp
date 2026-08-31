# Third-Party Notices and Licenses

xdna.cpp adapts concepts, APIs, and implementations from several open-source projects. We gratefully acknowledge the authors and contributors of:

---

### 1. llama.cpp / GGML
- **Project**: [llama.cpp](https://github.com/ggerganov/llama.cpp) & [ggml](https://github.com/ggerganov/ggml)
- **Copyright**: Copyright (c) 2023-2026 Georgi Gerganov and contributors
- **License**: MIT License
- **Usage**: Backend interface definition, tensor graph computation abstractions, GGUF parser structures, and reference quantization algorithms.

---

### 2. AMD IRON / AIE Kernels
- **Project**: [AMD IRON](https://github.com/nod-ai/iron)
- **Copyright**: Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
- **License**: Apache License 2.0
- **Usage**: AIE2 tile dataflow concepts, fused uint4 dequantization formulations, and column dataflow streaming pipeline architecture.

---

### 3. MLIR-AIE / Xilinx Runtime (XRT)
- **Project**: [mlir-aie](https://github.com/Xilinx/mlir-aie) & [XRT](https://github.com/Xilinx/XRT)
- **Copyright**: Copyright (c) 2020-2026 Advanced Micro Devices, Inc. / Xilinx, Inc.
- **License**: Apache License 2.0
- **Usage**: Device discovery, DRM driver communication via `/dev/accel/accel*`, hardware context management, buffer object (BO) allocation, and ELF kernel module registration.

---

### License Text: MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
