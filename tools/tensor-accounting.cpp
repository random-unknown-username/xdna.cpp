#include "xdna/gguf_parser.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void analyze_model(const std::string& name, const std::string& path) {
    auto model = xdna::parse_gguf_file(path);
    if (!model.is_valid) {
        std::cerr << "Failed to parse: " << path << ": " << model.error << "\n";
        return;
    }

    size_t xdna_weight_bytes = 0;
    size_t xdna_scale_bytes = 0;
    size_t xdna_total_packed_bytes = 0;
    size_t xdna_act_bytes = 0;
    size_t xdna_out_bytes = 0;
    size_t xdna_tensor_count = 0;

    size_t cpu_tensor_count = 0;
    size_t cpu_tensor_bytes = 0;

    for (const auto& t : model.tensors) {
        if (t.type == "Q4_0" && t.dimensions.size() == 2 && t.dimensions[0] % 32 == 0) {
            xdna_tensor_count++;
            size_t K = t.dimensions[0];
            size_t N = t.dimensions[1];
            size_t n_blocks = (K / 32) * N;
            size_t w_bytes = n_blocks * 16;
            size_t s_bytes = n_blocks * 2;
            size_t total_packed = n_blocks * 18;

            xdna_weight_bytes += w_bytes;
            xdna_scale_bytes += s_bytes;
            xdna_total_packed_bytes += total_packed;
            xdna_act_bytes += K * sizeof(uint16_t);
            xdna_out_bytes += N * sizeof(float);
        } else {
            cpu_tensor_count++;
            cpu_tensor_bytes += t.size_bytes;
        }
    }

    std::cout << "================================================================================\n";
    std::cout << " MODEL TENSOR & DATAFLOW ACCOUNTING: " << name << "\n";
    std::cout << " Path: " << path << "\n";
    std::cout << "================================================================================\n";
    std::cout << " Total Tensors:                 " << model.tensors.size() << "\n";
    std::cout << " Total GGUF File Size:          " << std::fixed << std::setprecision(2)
              << (model.total_file_size / (1024.0 * 1024.0)) << " MiB ("
              << (model.total_file_size / (1024.0 * 1024.0 * 1024.0)) << " GiB)\n";
    std::cout << " Total Model Weight Bytes:      "
              << (model.total_tensor_bytes / (1024.0 * 1024.0)) << " MiB ("
              << (model.total_tensor_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB)\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << " XDNA-Offloadable Projections:  " << xdna_tensor_count << " tensors\n";
    std::cout << "   - Q4_0 Raw Weight Nibbles:   " << (xdna_weight_bytes / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "   - FP16/BF16 Scales:          " << (xdna_scale_bytes / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "   - Total Packed Weights (W):  " << (xdna_total_packed_bytes / (1024.0 * 1024.0)) << " MiB ("
              << (xdna_total_packed_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB)\n";
    std::cout << "   - Activation Vector Read (x):" << (xdna_act_bytes / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "   - Output Vector Write (y):   " << (xdna_out_bytes / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "   - Total XDNA Decode Dataflow:" << ((xdna_total_packed_bytes + xdna_act_bytes + xdna_out_bytes) / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << " CPU Fallback Tensors:          " << cpu_tensor_count << " tensors\n";
    std::cout << "   - CPU Tensor Bytes:          " << (cpu_tensor_bytes / (1024.0 * 1024.0)) << " MiB ("
              << (cpu_tensor_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB)\n";
    std::cout << "================================================================================\n\n";
}

int main() {
    analyze_model("Qwen2.5-0.5B-Instruct-Q4_0", "/home/satvik/FORK/Qwen2.5-0.5B-Instruct-Q4_0.gguf");
    analyze_model("Qwen3.8-27B-Q4_0", "/home/satvik/FORK/Qwen3.8-27B-Q4_0.gguf");
    return 0;
}
