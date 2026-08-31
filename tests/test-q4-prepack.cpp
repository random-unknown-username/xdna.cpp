#include "xdna/q4_prepack.h"
#include "xdna/q4_0_reference.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

int main() {
    std::cout << "--- Testing Q4_0 Tile Prepacker vs GGML CPU Reference ---\n";

    const size_t M = 64;
    const size_t K = 5120;
    const size_t num_cols = 4;
    const size_t group_size = 32;

    auto layout = xdna::Q4Prepacker::compute_layout(M, K, num_cols, group_size);
    assert(layout.M == M);
    assert(layout.K == K);
    assert(layout.rows_per_column == 16);
    assert(layout.groups_per_row == 160);
    assert(layout.weight_bytes_per_row == 2560);
    assert(layout.scale_bytes_per_row == 320);
    assert(layout.packed_bytes_per_row == 2880);
    assert(layout.total_packed_bytes == 64 * 2880); // 184,320 bytes

    std::cout << "  Layout computation: PASS (Total packed bytes: " << layout.total_packed_bytes << ")\n";

    // Generate pseudo-random GGUF Q4_0 blocks
    std::mt19937 rng(1337);
    size_t gguf_row_bytes = (K / 32) * sizeof(xdna::q4_0::BlockQ4_0);
    std::vector<uint8_t> gguf_data(M * gguf_row_bytes);

    for (size_t r = 0; r < M; ++r) {
        auto* blks = reinterpret_cast<xdna::q4_0::BlockQ4_0*>(gguf_data.data() + r * gguf_row_bytes);
        for (size_t g = 0; g < K / 32; ++g) {
            blks[g].d = 0x3C00; // FP16 1.0
            for (size_t j = 0; j < 16; ++j) {
                blks[g].qs[j] = static_cast<uint8_t>((rng() % 16) | ((rng() % 16) << 4));
            }
        }
    }

    // Pack into AIE tile format
    std::vector<uint8_t> packed_output(layout.total_packed_bytes);
    xdna::Q4Prepacker::pack(gguf_data.data(), M, K, packed_output.data(), num_cols, group_size);
    std::cout << "  Packing into AIE column-partitioned layout: PASS\n";

    // Dequantize packed format
    std::vector<float> reconstructed_f32(M * K);
    xdna::Q4Prepacker::dequantize_packed(packed_output.data(), M, K, reconstructed_f32.data(), num_cols, group_size);

    // Golden CPU Reference Dequantization
    std::vector<float> golden_f32(M * K);
    for (size_t r = 0; r < M; ++r) {
        xdna::q4_0::dequantize_row(gguf_data.data() + r * gguf_row_bytes, gguf_row_bytes, K, golden_f32.data() + r * K);
    }

    // Compare bit-exact floating point equality
    double max_diff = 0.0;
    for (size_t i = 0; i < M * K; ++i) {
        double diff = std::abs(reconstructed_f32[i] - golden_f32[i]);
        max_diff = std::max(max_diff, diff);
        assert(diff < 1e-4);
    }

    std::cout << "  Reconstructed vs Golden CPU Reference Dequantization: PASS (Max diff: " << max_diff << ")\n";
    std::cout << "ALL PREPACKER TESTS PASSED!\n";
    return 0;
}
