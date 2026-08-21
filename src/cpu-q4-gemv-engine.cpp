#include "xdna/cpu_q4_gemv_engine.h"
#include "xdna/q4_0_reference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace xdna {

namespace {

// Compute dot product for a single Q4_0 row with float activations
inline float dot_q4_0_f32_fast(const uint8_t* row_bytes, size_t K, const float* x) {
    const size_t nb = K / 32;
    const auto* blocks = reinterpret_cast<const q4_0::BlockQ4_0*>(row_bytes);

    __m256 acc0 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        float d = q4_0::fp16_to_float(blocks[b].d);
        __m256 scale = _mm256_set1_ps(d);

        // Load 16 packed bytes (32 nibbles)
        __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blocks[b].qs));

        // Extract low 16 nibbles (indices 0..15) and high 16 nibbles (indices 16..31)
        __m128i low_nibbles = _mm_and_si128(raw16, _mm_set1_epi8(0x0F));
        __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw16, 4), _mm_set1_epi8(0x0F));

        // Sign extend / subtract 8
        __m128i v0_8 = _mm_sub_epi16(_mm_cvtepi8_epi16(low_nibbles), _mm_set1_epi16(8));
        __m256 v0_8_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v0_8));

        __m128i v8_16 = _mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(low_nibbles, 8)), _mm_set1_epi16(8));
        __m256 v8_16_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v8_16));

        __m128i v16_24 = _mm_sub_epi16(_mm_cvtepi8_epi16(high_nibbles), _mm_set1_epi16(8));
        __m256 v16_24_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v16_24));

        __m128i v24_32 = _mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(high_nibbles, 8)), _mm_set1_epi16(8));
        __m256 v24_32_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v24_32));

        // Load activations
        const float* act_ptr = x + b * 32;
        __m256 x0 = _mm256_loadu_ps(act_ptr + 0);
        __m256 x1 = _mm256_loadu_ps(act_ptr + 8);
        __m256 x2 = _mm256_loadu_ps(act_ptr + 16);
        __m256 x3 = _mm256_loadu_ps(act_ptr + 24);

        __m256 prod0 = _mm256_mul_ps(v0_8_f, x0);
        __m256 prod1 = _mm256_mul_ps(v8_16_f, x1);
        __m256 prod2 = _mm256_mul_ps(v16_24_f, x2);
        __m256 prod3 = _mm256_mul_ps(v24_32_f, x3);

        __m256 sum_blk = _mm256_add_ps(_mm256_add_ps(prod0, prod1), _mm256_add_ps(prod2, prod3));
        acc0 = _mm256_fmadd_ps(sum_blk, scale, acc0);
    }

    // Horizontal add acc0
    __m128 h0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    h0 = _mm_add_ps(h0, _mm_movehl_ps(h0, h0));
    h0 = _mm_add_ss(h0, _mm_shuffle_ps(h0, h0, 1));
    return _mm_cvtss_f32(h0);
}

// Unrolled 2-row dot product (shares activation vector in L1/L2)
inline void dot_q4_0_f32_2rows(const uint8_t* row0, const uint8_t* row1, size_t K, const float* x, float& out0, float& out1) {
    const size_t nb = K / 32;
    const auto* blk0 = reinterpret_cast<const q4_0::BlockQ4_0*>(row0);
    const auto* blk1 = reinterpret_cast<const q4_0::BlockQ4_0*>(row1);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        // Load activations once for both rows
        const float* act_ptr = x + b * 32;
        __m256 x0 = _mm256_loadu_ps(act_ptr + 0);
        __m256 x1 = _mm256_loadu_ps(act_ptr + 8);
        __m256 x2 = _mm256_loadu_ps(act_ptr + 16);
        __m256 x3 = _mm256_loadu_ps(act_ptr + 24);

        // Row 0
        float d0 = q4_0::fp16_to_float(blk0[b].d);
        __m256 scale0 = _mm256_set1_ps(d0);
        __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk0[b].qs));
        __m128i low0 = _mm_and_si128(raw0, _mm_set1_epi8(0x0F));
        __m128i high0 = _mm_and_si128(_mm_srli_epi16(raw0, 4), _mm_set1_epi8(0x0F));
        __m256 v0_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(low0), _mm_set1_epi16(8))));
        __m256 v0_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(low0, 8)), _mm_set1_epi16(8))));
        __m256 v0_2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(high0), _mm_set1_epi16(8))));
        __m256 v0_3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(high0, 8)), _mm_set1_epi16(8))));
        __m256 sum0 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v0_0, x0), _mm256_mul_ps(v0_1, x1)),
                                    _mm256_add_ps(_mm256_mul_ps(v0_2, x2), _mm256_mul_ps(v0_3, x3)));
        acc0 = _mm256_fmadd_ps(sum0, scale0, acc0);

        // Row 1
        float d1 = q4_0::fp16_to_float(blk1[b].d);
        __m256 scale1 = _mm256_set1_ps(d1);
        __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk1[b].qs));
        __m128i low1 = _mm_and_si128(raw1, _mm_set1_epi8(0x0F));
        __m128i high1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), _mm_set1_epi8(0x0F));
        __m256 v1_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(low1), _mm_set1_epi16(8))));
        __m256 v1_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(low1, 8)), _mm_set1_epi16(8))));
        __m256 v1_2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(high1), _mm_set1_epi16(8))));
        __m256 v1_3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(high1, 8)), _mm_set1_epi16(8))));
        __m256 sum1 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v1_0, x0), _mm256_mul_ps(v1_1, x1)),
                                    _mm256_add_ps(_mm256_mul_ps(v1_2, x2), _mm256_mul_ps(v1_3, x3)));
        acc1 = _mm256_fmadd_ps(sum1, scale1, acc1);
    }

    auto h_add = [](__m256 acc) -> float {
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
        return _mm_cvtss_f32(h);
    };

    out0 = h_add(acc0);
    out1 = h_add(acc1);
}

} // namespace

struct Q4GemvEngine::Impl {
    uint32_t num_workers;
    explicit Impl(uint32_t n_workers) : num_workers(n_workers > 0 ? n_workers : 4) {}
};

Q4GemvEngine::Q4GemvEngine(uint32_t num_columns, uint32_t rows_per_column)
    : m_num_columns(num_columns > 0 ? num_columns : 4),
      m_rows_per_col(rows_per_column > 0 ? rows_per_column : 4),
      m_impl(std::make_unique<Impl>(m_num_columns * m_rows_per_col)) {}

Q4GemvEngine::~Q4GemvEngine() = default;

void Q4GemvEngine::run(const uint8_t* packed_weights,
                      size_t weight_bytes,
                      size_t N,
                      size_t K,
                      const float* activations,
                      float* output) {
    const size_t bytes_per_row = (K / 32) * 18;
    if (weight_bytes < N * bytes_per_row) {
        throw std::invalid_argument("Weight buffer is smaller than required for matrix");
    }

    if (N <= 256) {
        // Fast single-core unrolled path without any thread sync overhead
        size_t r = 0;
        for (; r + 1 < N; r += 2) {
            const uint8_t* row0 = packed_weights + r * bytes_per_row;
            const uint8_t* row1 = packed_weights + (r + 1) * bytes_per_row;
            dot_q4_0_f32_2rows(row0, row1, K, activations, output[r], output[r + 1]);
        }
        if (r < N) {
            output[r] = dot_q4_0_f32_fast(packed_weights + r * bytes_per_row, K, activations);
        }
    } else {
        // Lightweight multi-threaded path using max 4 threads to prevent 100% CPU pegging
        #pragma omp parallel for schedule(static) num_threads(4)
        for (size_t r = 0; r < N; r += 2) {
            if (r + 1 < N) {
                const uint8_t* row0 = packed_weights + r * bytes_per_row;
                const uint8_t* row1 = packed_weights + (r + 1) * bytes_per_row;
                dot_q4_0_f32_2rows(row0, row1, K, activations, output[r], output[r + 1]);
            } else {
                output[r] = dot_q4_0_f32_fast(packed_weights + r * bytes_per_row, K, activations);
            }
        }
    }
}

void Q4GemvEngine::run_bf16(const uint8_t* packed_weights,
                           size_t weight_bytes,
                           size_t N,
                           size_t K,
                           const q4_0::Bf16* activations,
                           float* output) {
    std::vector<float> act_f32(K);
    for (size_t i = 0; i < K; ++i) {
        act_f32[i] = q4_0::bf16_to_float(activations[i]);
    }
    run(packed_weights, weight_bytes, N, K, act_f32.data(), output);
}

KernelMetrics Q4GemvEngine::evaluate(const float* actual,
                                   const float* reference,
                                   size_t count,
                                   size_t weight_bytes,
                                   double duration_us) {
    KernelMetrics m;
    m.latency_us = duration_us;
    if (duration_us > 0.0) {
        m.throughput_tok_s = 1e6 / duration_us;
        m.effective_bandwidth_gb_s = (static_cast<double>(weight_bytes) / (duration_us * 1e-6)) / 1e9;
    }

    double dot = 0.0, n_act = 0.0, n_ref = 0.0, l2_diff = 0.0, abs_sum = 0.0;
    double max_err = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double a = actual[i];
        double r = reference[i];
        double diff = std::abs(a - r);
        max_err = std::max(max_err, diff);
        abs_sum += diff;
        l2_diff += (a - r) * (a - r);
        dot += a * r;
        n_act += a * a;
        n_ref += r * r;
    }

    m.max_abs_error = max_err;
    m.mean_abs_error = abs_sum / count;
    m.relative_l2_error = (n_ref > 1e-12) ? std::sqrt(l2_diff / n_ref) : 0.0;
    double denom = std::sqrt(n_act) * std::sqrt(n_ref);
    m.cosine_similarity = (denom > 1e-12) ? (dot / denom) : 0.0;
    m.passed = (m.cosine_similarity >= 0.9999);
    return m;
}

} // namespace xdna
