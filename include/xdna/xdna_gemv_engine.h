#pragma once

#include "xdna/kquants.h"
#include "xdna/q4_prepack.h"
#include "xdna/q4_0_reference.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace xdna {

/// Partition specification for 4 columns and 16 compute tiles
struct TilePartition {
    size_t N = 0;
    size_t K = 0;
    size_t num_columns = 4;
    size_t tiles_per_col = 4;
    size_t total_tiles = 16;
    size_t rows_per_col = 0;       // N / 4
    size_t rows_per_tile = 0;      // N / 16
    size_t blocks_per_row = 0;     // K / 32
    size_t bytes_per_row = 0;      // blocks_per_row * 18
    size_t bytes_per_col = 0;      // rows_per_col * bytes_per_row
    size_t total_weight_bytes = 0; // N * bytes_per_row

    static TilePartition create(size_t N, size_t K, size_t num_columns = 4, size_t bytes_per_row_override = 0) {
        TilePartition p;
        p.N = N;
        p.K = K;
        p.num_columns = num_columns > 0 ? num_columns : 4;
        p.tiles_per_col = 4;
        p.total_tiles = p.num_columns * p.tiles_per_col;
        if (bytes_per_row_override > 0) {
            p.bytes_per_row = bytes_per_row_override;
            p.blocks_per_row = K / 256;
        } else {
            p.blocks_per_row = K / 32;
            p.bytes_per_row = p.blocks_per_row * 18;
        }

        if (p.num_columns > 0 && N > SIZE_MAX - (p.num_columns - 1)) {
            throw std::invalid_argument("TilePartition::create: integer overflow in rows_per_col calculation");
        }
        if (p.bytes_per_row > 0 && N > SIZE_MAX / p.bytes_per_row) {
            throw std::invalid_argument("TilePartition::create: integer overflow in total_weight_bytes calculation");
        }

        p.rows_per_col = (N + p.num_columns - 1) / p.num_columns;
        p.rows_per_tile = (p.rows_per_col + p.tiles_per_col - 1) / p.tiles_per_col;
        if (p.bytes_per_row > 0 && p.rows_per_col > SIZE_MAX / p.bytes_per_row) {
            throw std::invalid_argument("TilePartition::create: integer overflow in bytes_per_col calculation");
        }
        p.bytes_per_col = p.rows_per_col * p.bytes_per_row;
        p.total_weight_bytes = N * p.bytes_per_row;
        return p;
    }

    static TilePartition create_for_quant(QuantType type, size_t N, size_t K, size_t num_columns = 4) {
        size_t bpr = bytes_per_row_for_quant(type, K);
        return create(N, K, num_columns, bpr);
    }

    [[nodiscard]] size_t col_weight_offset(size_t c) const noexcept {
        return c * bytes_per_col;
    }

    [[nodiscard]] std::pair<size_t, size_t> tile_row_range(size_t col, size_t tile_in_col) const noexcept {
        size_t tile_idx = col * tiles_per_col + tile_in_col;
        if (rows_per_tile > 0 && tile_idx > SIZE_MAX / rows_per_tile) {
            return {N, N};
        }
        size_t start = tile_idx * rows_per_tile;
        if (start > N) start = N;
        size_t end = (SIZE_MAX - start < rows_per_tile) ? N : std::min(start + rows_per_tile, N);
        return {start, end};
    }
};

struct XdnaKernelMetrics {
    double latency_us = 0.0;
    double effective_bandwidth_gb_s = 0.0;
    double cosine_similarity = 0.0;
    double max_abs_error = 0.0;
    uint32_t active_columns = 4;
    uint32_t active_tiles = 16;
    bool hardware_executed = false;
    bool passed = false;
};

struct TokenProfile {
    uint64_t token_index = 0;
    double total_token_ms = 0.0;
    double xdna_gemv_ms = 0.0;
    double cpu_fallback_ms = 0.0;
    double prepack_ms = 0.0;
    double bo_alloc_ms = 0.0;
    double hw_ctx_ms = 0.0;
    double act_conversion_ms = 0.0;
    double sync_copies_ms = 0.0;
    uint64_t num_dispatches = 0;
    uint64_t num_transitions = 0;
};

class XdnaGemvEngine {
public:
    /// Quantitative metrics contract defined in PROJECT.md and Feature 7 tests
    struct Metrics {
        double useful_weight_bytes = 0.0;
        double physical_dma_bytes = 0.0;
        double total_time_ms = 0.0;
        double npu_kernel_time_ms = 0.0;
        double effective_gb_s = 0.0;
    };

    explicit XdnaGemvEngine(size_t device_index = 0,
                            uint32_t num_columns = 4,
                            size_t crossover_n = 256);
    ~XdnaGemvEngine();

    XdnaGemvEngine(const XdnaGemvEngine&) = delete;
    XdnaGemvEngine& operator=(const XdnaGemvEngine&) = delete;
    XdnaGemvEngine(XdnaGemvEngine&&) noexcept;
    XdnaGemvEngine& operator=(XdnaGemvEngine&&) noexcept;

    [[nodiscard]] bool is_hardware_ready() const noexcept;

    [[nodiscard]] size_t crossover_threshold() const noexcept;
    void set_crossover_threshold(size_t n) noexcept;

    [[nodiscard]] Metrics get_last_metrics() const;

    // Run fused Q4_0 decode GEMV across XDNA1 NPU array
    // Matrix W: [N, K] in Q4_0 packed format (N rows, K columns)
    // Activation x: [K] in float / BF16
    // Output y: [N] in float
    void run(const uint8_t* gguf_q4_0_weights,
             size_t weight_bytes,
             size_t N,
             size_t K,
             const float* activations_f32,
             float* output_f32);

    void run_bf16(const uint8_t* gguf_q4_0_weights,
                  size_t weight_bytes,
                  size_t N,
                  size_t K,
                  const q4_0::Bf16* activations_bf16,
                  float* output_f32);

    // Multi-quant GEMV execution across XDNA1 / AIE array with Zen 4 crossover
    void run_quant(QuantType type,
                   const uint8_t* weights,
                   size_t weight_bytes,
                   size_t N,
                   size_t K,
                   const float* activations_f32,
                   float* output_f32);

    void run_quant_bf16(QuantType type,
                        const uint8_t* weights,
                        size_t weight_bytes,
                        size_t N,
                        size_t K,
                        const q4_0::Bf16* activations_bf16,
                        float* output_f32);

    void run_q4_k(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out);
    void run_q4_k_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out);

    void run_q5_k(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out);
    void run_q5_k_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out);

    void run_q3_k(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out);
    void run_q3_k_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out);

    void run_q6_k(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out);
    void run_q6_k_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out);

    void run_iq4_xs(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out);
    void run_iq4_xs_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out);

    void run_iq3_s(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out);
    void run_iq3_s_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out);

    void start_token_profile();
    TokenProfile end_token_profile();
    static void print_token_profile(const TokenProfile& prof);

    static XdnaKernelMetrics evaluate(const float* actual,
                                     const float* reference,
                                     size_t count,
                                     size_t packed_weight_bytes,
                                     double duration_us,
                                     bool hardware_executed);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace xdna
