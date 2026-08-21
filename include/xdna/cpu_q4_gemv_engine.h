#pragma once

#include "xdna/q4_0_reference.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xdna {

struct KernelMetrics {
    double latency_us = 0.0;
    double throughput_tok_s = 0.0;
    double effective_bandwidth_gb_s = 0.0;
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double relative_l2_error = 0.0;
    double cosine_similarity = 0.0;
    bool passed = false;
};

class Q4GemvEngine {
public:
    explicit Q4GemvEngine(uint32_t num_columns = 4, uint32_t rows_per_column = 4);
    ~Q4GemvEngine();

    // Run fused Q4_0 dequant + GEMV
    // Matrix W: [N, K] in Q4_0 packed format (N rows, K columns)
    // Activation x: [K] in float / BF16
    // Output y: [N] in float
    void run(const uint8_t* packed_weights,
             size_t weight_bytes,
             size_t N,
             size_t K,
             const float* activations,
             float* output);

    void run_bf16(const uint8_t* packed_weights,
                  size_t weight_bytes,
                  size_t N,
                  size_t K,
                  const q4_0::Bf16* activations,
                  float* output);

    // Compute correctness and performance metrics against reference
    static KernelMetrics evaluate(const float* actual,
                                 const float* reference,
                                 size_t count,
                                 size_t weight_bytes,
                                 double duration_us);

private:
    uint32_t m_num_columns;
    uint32_t m_rows_per_col;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace xdna
