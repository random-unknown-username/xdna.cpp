#include "xdna/q4_0_reference.h"
#include "xdna/q4_prepack.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

struct PrecisionComparison {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double relative_l2 = 0.0;
    double cosine_similarity = 0.0;
    size_t count = 0;
};

PrecisionComparison compare_vectors(const std::vector<float>& ref, const std::vector<float>& actual) {
    PrecisionComparison res;
    res.count = ref.size();
    double dot = 0.0, norm_ref = 0.0, norm_act = 0.0, l2_diff = 0.0, abs_sum = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        double r = ref[i];
        double a = actual[i];
        double diff = std::abs(r - a);
        res.max_abs_error = std::max(res.max_abs_error, diff);
        abs_sum += diff;
        l2_diff += (r - a) * (r - a);
        dot += r * a;
        norm_ref += r * r;
        norm_act += a * a;
    }

    res.mean_abs_error = abs_sum / ref.size();
    res.relative_l2 = (norm_ref > 1e-12) ? std::sqrt(l2_diff / norm_ref) : 0.0;
    double denom = std::sqrt(norm_ref) * std::sqrt(norm_act);
    res.cosine_similarity = (denom > 1e-12) ? (dot / denom) : 0.0;
    return res;
}

int main() {
    std::cout << "========================================================================================================\n";
    std::cout << " SCALE CONVERSION & NUMERICAL PRECISION AUDIT: FP16 (GGML) vs BF16 (XDNA)\n";
    std::cout << "========================================================================================================\n";

    const size_t N = 1024;
    const size_t K = 5120;
    const size_t num_blocks = N * (K / 32);

    std::vector<xdna::q4_0::BlockQ4_0> ggml_blocks(num_blocks);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.0001f, 0.5f);
    std::uniform_int_distribution<int> nibble_dist(0, 15);

    // Generate valid synthetic GGML Q4_0 blocks
    for (size_t b = 0; b < num_blocks; ++b) {
        float orig_scale = scale_dist(rng);
        ggml_blocks[b].d = xdna::q4_0::float_to_bf16(orig_scale).bits; // Test FP16 pattern
        // Convert to true FP16 bits:
        uint32_t x;
        std::memcpy(&x, &orig_scale, sizeof(x));
        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (x & 0x007FFFFF) >> 13;
        uint16_t fp16_val = (exp <= 0) ? sign : ((exp >= 31) ? (sign | 0x7C00) : (sign | (exp << 10) | mant));
        ggml_blocks[b].d = fp16_val;

        for (int i = 0; i < 16; ++i) {
            uint8_t q0 = nibble_dist(rng);
            uint8_t q1 = nibble_dist(rng);
            ggml_blocks[b].qs[i] = q0 | (q1 << 4);
        }
    }

    std::vector<float> activations(K);
    std::normal_distribution<float> act_dist(0.0f, 1.0f);
    for (size_t i = 0; i < K; ++i) activations[i] = act_dist(rng);

    // Reference A: GGML FP16 Scale Oracle
    std::vector<float> out_A(N, 0.0f);
    for (size_t row = 0; row < N; ++row) {
        float sum = 0.0f;
        for (size_t b = 0; b < K / 32; ++b) {
            size_t blk_idx = row * (K / 32) + b;
            const auto& blk = ggml_blocks[blk_idx];
            float d = xdna::q4_0::fp16_to_float(blk.d);

            for (int i = 0; i < 16; ++i) {
                uint8_t byte = blk.qs[i];
                int8_t v0 = (byte & 0x0F) - 8;
                int8_t v1 = (byte >> 4) - 8;
                sum += (static_cast<float>(v0) * d) * activations[b * 32 + i];
                sum += (static_cast<float>(v1) * d) * activations[b * 32 + 16 + i];
            }
        }
        out_A[row] = sum;
    }

    // Reference B: CPU Emulation with BF16-Rounded Scale
    std::vector<float> out_B(N, 0.0f);
    for (size_t row = 0; row < N; ++row) {
        float sum = 0.0f;
        for (size_t b = 0; b < K / 32; ++b) {
            size_t blk_idx = row * (K / 32) + b;
            const auto& blk = ggml_blocks[blk_idx];
            float orig_d = xdna::q4_0::fp16_to_float(blk.d);
            xdna::q4_0::Bf16 bf16_d = xdna::q4_0::float_to_bf16(orig_d);
            float d = xdna::q4_0::bf16_to_float(bf16_d);

            for (int i = 0; i < 16; ++i) {
                uint8_t byte = blk.qs[i];
                int8_t v0 = (byte & 0x0F) - 8;
                int8_t v1 = (byte >> 4) - 8;
                sum += (static_cast<float>(v0) * d) * activations[b * 32 + i];
                sum += (static_cast<float>(v1) * d) * activations[b * 32 + 16 + i];
            }
        }
        out_B[row] = sum;
    }

    // Reference C: Prepacked XDNA Layout Output
    std::vector<uint8_t> prepacked(N * (K / 32) * 18);
    xdna::Q4Prepacker::pack(reinterpret_cast<const uint8_t*>(ggml_blocks.data()), N, K, prepacked.data(), 4, 32);

    std::vector<float> dequant_matrix(N * K);
    xdna::Q4Prepacker::dequantize_packed(prepacked.data(), N, K, dequant_matrix.data(), 4, 32);

    std::vector<float> out_C(N, 0.0f);
    for (size_t row = 0; row < N; ++row) {
        float sum = 0.0f;
        for (size_t col = 0; col < K; ++col) {
            sum += dequant_matrix[row * K + col] * activations[col];
        }
        out_C[row] = sum;
    }

    auto comp_AB = compare_vectors(out_A, out_B);
    auto comp_BC = compare_vectors(out_B, out_C);
    auto comp_AC = compare_vectors(out_A, out_C);

    std::cout << std::left << std::setw(28) << "Comparison Pair"
              << std::setw(18) << "Max Abs Error"
              << std::setw(18) << "Mean Abs Error"
              << std::setw(18) << "Relative L2"
              << "Cosine Similarity\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(28) << "A (GGML FP16) vs B (BF16)"
              << std::fixed << std::setprecision(6)
              << std::setw(18) << comp_AB.max_abs_error
              << std::setw(18) << comp_AB.mean_abs_error
              << std::setw(18) << comp_AB.relative_l2
              << std::setprecision(8) << comp_AB.cosine_similarity << "\n";
    std::cout << std::left << std::setw(28) << "B (BF16) vs C (XDNA)"
              << std::fixed << std::setprecision(6)
              << std::setw(18) << comp_BC.max_abs_error
              << std::setw(18) << comp_BC.mean_abs_error
              << std::setw(18) << comp_BC.relative_l2
              << std::setprecision(8) << comp_BC.cosine_similarity << "\n";
    std::cout << std::left << std::setw(28) << "A (GGML FP16) vs C (XDNA)"
              << std::fixed << std::setprecision(6)
              << std::setw(18) << comp_AC.max_abs_error
              << std::setw(18) << comp_AC.mean_abs_error
              << std::setw(18) << comp_AC.relative_l2
              << std::setprecision(8) << comp_AC.cosine_similarity << "\n";
    std::cout << "========================================================================================================\n";
    return 0;
}
