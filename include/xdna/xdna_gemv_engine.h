#pragma once

#include "xdna/q4_prepack.h"
#include "xdna/q4_0_reference.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xdna {

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
    explicit XdnaGemvEngine(size_t device_index = 0, uint32_t num_columns = 4);
    ~XdnaGemvEngine();

    bool is_hardware_ready() const noexcept;

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
