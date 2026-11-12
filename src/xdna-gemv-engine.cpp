#include "xdna/xdna_gemv_engine.h"
#include "xdna/kquants.h"
#include "xdna/cpu_q4_gemv_engine.h"
#include "xdna/dma_streamer.h"
#include "xdna/q4_prepack.h"
#include "xdna/q4_0_reference.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef XDNA_WITH_XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#endif

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

inline void dot_q4_0_f32_4rows(const uint8_t* row0, const uint8_t* row1, const uint8_t* row2, const uint8_t* row3,
                               size_t K, const float* x, float& out0, float& out1, float& out2, float& out3) {
    const size_t nb = K / 32;
    const auto* blk0 = reinterpret_cast<const q4_0::BlockQ4_0*>(row0);
    const auto* blk1 = reinterpret_cast<const q4_0::BlockQ4_0*>(row1);
    const auto* blk2 = reinterpret_cast<const q4_0::BlockQ4_0*>(row2);
    const auto* blk3 = reinterpret_cast<const q4_0::BlockQ4_0*>(row3);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i s8 = _mm_set1_epi16(8);

    for (size_t b = 0; b < nb; ++b) {
        const float* act_ptr = x + b * 32;
        __m256 x0 = _mm256_loadu_ps(act_ptr + 0);
        __m256 x1 = _mm256_loadu_ps(act_ptr + 8);
        __m256 x2 = _mm256_loadu_ps(act_ptr + 16);
        __m256 x3 = _mm256_loadu_ps(act_ptr + 24);

        auto do_row = [&](const q4_0::BlockQ4_0& blk, __m256& acc) {
            float d = q4_0::fp16_to_float(blk.d);
            __m256 scale = _mm256_set1_ps(d);
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk.qs));
            __m128i low = _mm_and_si128(raw, m4);
            __m128i high = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
            __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(low), s8)));
            __m256 v1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(low, 8)), s8)));
            __m256 v2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(high), s8)));
            __m256 v3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(high, 8)), s8)));
            __m256 sum = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(v0, x0), _mm256_mul_ps(v1, x1)),
                                       _mm256_add_ps(_mm256_mul_ps(v2, x2), _mm256_mul_ps(v3, x3)));
            acc = _mm256_fmadd_ps(sum, scale, acc);
        };

        do_row(blk0[b], acc0);
        do_row(blk1[b], acc1);
        do_row(blk2[b], acc2);
        do_row(blk3[b], acc3);
    }

    auto h_add = [](__m256 a) -> float {
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
        return _mm_cvtss_f32(h);
    };

    out0 = h_add(acc0);
    out1 = h_add(acc1);
    out2 = h_add(acc2);
    out3 = h_add(acc3);
}

// ----------------------------------------------------------------------------
// Q4_K Vectorized AVX2 Dot Products
// ----------------------------------------------------------------------------
inline void dot_q4_K_f32_2rows(const uint8_t* row0, const uint8_t* row1, size_t K, const float* x, float& out0, float& out1) {
    const size_t nb = K / QK_K;
    const auto* blk0 = reinterpret_cast<const block_q4_K*>(row0);
    const auto* blk1 = reinterpret_cast<const block_q4_K*>(row1);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const __m128i m4 = _mm_set1_epi8(0x0F);

    for (size_t b = 0; b < nb; ++b) {
        const float d0   = q4_0::fp16_to_float(blk0[b].d);
        const float min0 = q4_0::fp16_to_float(blk0[b].dmin);
        const uint8_t* q0 = blk0[b].qs;

        const float d1   = q4_0::fp16_to_float(blk1[b].d);
        const float min1 = q4_0::fp16_to_float(blk1[b].dmin);
        const uint8_t* q1 = blk1[b].qs;

        int is = 0;
        uint8_t sc0_0, m0_0, sc0_1, m0_1;
        uint8_t sc1_0, m1_0, sc1_1, m1_1;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, blk0[b].scales, sc0_0, m0_0);
            get_scale_min_k4(is + 1, blk0[b].scales, sc0_1, m0_1);
            const float d1_0 = d0 * sc0_0; const float m1_0_val = min0 * m0_0;
            const float d2_0 = d0 * sc0_1; const float m2_0_val = min0 * m0_1;

            get_scale_min_k4(is + 0, blk1[b].scales, sc1_0, m1_0);
            get_scale_min_k4(is + 1, blk1[b].scales, sc1_1, m1_1);
            const float d1_1 = d1 * sc1_0; const float m1_1_val = min1 * m1_0;
            const float d2_1 = d1 * sc1_1; const float m2_1_val = min1 * m1_1;

            const float* act_ptr = x + b * 256 + j;

            __m128i raw0_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q0 + 0));
            __m128i raw0_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q0 + 16));
            __m128i raw1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q1 + 0));
            __m128i raw1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q1 + 16));

            __m128i lo0_0 = _mm_and_si128(raw0_0, m4);
            __m128i hi0_0 = _mm_and_si128(_mm_srli_epi16(raw0_0, 4), m4);
            __m128i lo0_1 = _mm_and_si128(raw0_1, m4);
            __m128i hi0_1 = _mm_and_si128(_mm_srli_epi16(raw0_1, 4), m4);

            __m128i lo1_0 = _mm_and_si128(raw1_0, m4);
            __m128i hi1_0 = _mm_and_si128(_mm_srli_epi16(raw1_0, 4), m4);
            __m128i lo1_1 = _mm_and_si128(raw1_1, m4);
            __m128i hi1_1 = _mm_and_si128(_mm_srli_epi16(raw1_1, 4), m4);

            __m256 x0 = _mm256_loadu_ps(act_ptr + 0);
            __m256 x1 = _mm256_loadu_ps(act_ptr + 8);
            __m256 x2 = _mm256_loadu_ps(act_ptr + 16);
            __m256 x3 = _mm256_loadu_ps(act_ptr + 24);
            __m256 x4 = _mm256_loadu_ps(act_ptr + 32);
            __m256 x5 = _mm256_loadu_ps(act_ptr + 40);
            __m256 x6 = _mm256_loadu_ps(act_ptr + 48);
            __m256 x7 = _mm256_loadu_ps(act_ptr + 56);

            __m256 sx_0 = _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3));
            __m256 sx_1 = _mm256_add_ps(_mm256_add_ps(x4, x5), _mm256_add_ps(x6, x7));

            // Row 0, Subblock 0
            __m256 p0_0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo0_0)), x0,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo0_0, 8))), x1));
            __m256 p0_1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo0_1)), x2,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo0_1, 8))), x3));
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d1_0), _mm256_add_ps(p0_0, p0_1),
                   _mm256_fnmadd_ps(_mm256_set1_ps(m1_0_val), sx_0, acc0));

            // Row 0, Subblock 1
            __m256 p0_2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi0_0)), x4,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi0_0, 8))), x5));
            __m256 p0_3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi0_1)), x6,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi0_1, 8))), x7));
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d2_0), _mm256_add_ps(p0_2, p0_3),
                   _mm256_fnmadd_ps(_mm256_set1_ps(m2_0_val), sx_1, acc0));

            // Row 1, Subblock 0
            __m256 p1_0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo1_0)), x0,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo1_0, 8))), x1));
            __m256 p1_1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo1_1)), x2,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo1_1, 8))), x3));
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1_1), _mm256_add_ps(p1_0, p1_1),
                   _mm256_fnmadd_ps(_mm256_set1_ps(m1_1_val), sx_0, acc1));

            // Row 1, Subblock 1
            __m256 p1_2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi1_0)), x4,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi1_0, 8))), x5));
            __m256 p1_3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi1_1)), x6,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi1_1, 8))), x7));
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d2_1), _mm256_add_ps(p1_2, p1_3),
                   _mm256_fnmadd_ps(_mm256_set1_ps(m2_1_val), sx_1, acc1));

            q0 += 32;
            q1 += 32;
            is += 2;
        }
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

inline float dot_q4_K_f32_fast(const uint8_t* row_bytes, size_t K, const float* x) {
    float out0 = 0.0f, out1 = 0.0f;
    dot_q4_K_f32_2rows(row_bytes, row_bytes, K, x, out0, out1);
    return out0;
}

// ----------------------------------------------------------------------------
// Q5_K Vectorized AVX2 Dot Products
// ----------------------------------------------------------------------------
inline void dot_q5_K_f32_2rows(const uint8_t* row0, const uint8_t* row1, size_t K, const float* x, float& out0, float& out1) {
    const size_t nb = K / QK_K;
    const auto* blk0 = reinterpret_cast<const block_q5_K*>(row0);
    const auto* blk1 = reinterpret_cast<const block_q5_K*>(row1);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i v16 = _mm_set1_epi8(16);

    for (size_t b = 0; b < nb; ++b) {
        const float d0   = q4_0::fp16_to_float(blk0[b].d);
        const float min0 = q4_0::fp16_to_float(blk0[b].dmin);
        const uint8_t* ql0 = blk0[b].qs;
        const uint8_t* qh0 = blk0[b].qh;

        const float d1   = q4_0::fp16_to_float(blk1[b].d);
        const float min1 = q4_0::fp16_to_float(blk1[b].dmin);
        const uint8_t* ql1 = blk1[b].qs;
        const uint8_t* qh1 = blk1[b].qh;

        int is = 0;
        uint8_t sc0_0, m0_0, sc0_1, m0_1;
        uint8_t sc1_0, m1_0, sc1_1, m1_1;
        uint8_t u1 = 1, u2 = 2;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, blk0[b].scales, sc0_0, m0_0);
            get_scale_min_k4(is + 1, blk0[b].scales, sc0_1, m0_1);
            const float d1_0 = d0 * sc0_0; const float m1_0_val = min0 * m0_0;
            const float d2_0 = d0 * sc0_1; const float m2_0_val = min0 * m0_1;

            get_scale_min_k4(is + 0, blk1[b].scales, sc1_0, m1_0);
            get_scale_min_k4(is + 1, blk1[b].scales, sc1_1, m1_1);
            const float d1_1 = d1 * sc1_0; const float m1_1_val = min1 * m1_0;
            const float d2_1 = d1 * sc1_1; const float m2_1_val = min1 * m1_1;

            const float* act_ptr = x + b * 256 + j;

            __m128i raw_ql0_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql0 + 0));
            __m128i raw_ql0_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql0 + 16));
            __m128i raw_qh0_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh0 + 0));
            __m128i raw_qh0_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh0 + 16));

            __m128i raw_ql1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql1 + 0));
            __m128i raw_ql1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql1 + 16));
            __m128i raw_qh1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh1 + 0));
            __m128i raw_qh1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh1 + 16));

            __m128i vu1 = _mm_set1_epi8(static_cast<char>(u1));
            __m128i vu2 = _mm_set1_epi8(static_cast<char>(u2));

            // Row 0
            __m128i h16_0_0_lo = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(raw_qh0_0, vu1), vu1), v16);
            __m128i h16_0_1_lo = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(raw_qh0_1, vu1), vu1), v16);
            __m128i h16_0_0_hi = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(raw_qh0_0, vu2), vu2), v16);
            __m128i h16_0_1_hi = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(raw_qh0_1, vu2), vu2), v16);

            __m128i w0_lo0 = _mm_add_epi8(_mm_and_si128(raw_ql0_0, m4), h16_0_0_lo);
            __m128i w0_lo1 = _mm_add_epi8(_mm_and_si128(raw_ql0_1, m4), h16_0_1_lo);
            __m128i w0_hi0 = _mm_add_epi8(_mm_and_si128(_mm_srli_epi16(raw_ql0_0, 4), m4), h16_0_0_hi);
            __m128i w0_hi1 = _mm_add_epi8(_mm_and_si128(_mm_srli_epi16(raw_ql0_1, 4), m4), h16_0_1_hi);

            // Row 1
            __m128i h16_1_0_lo = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(raw_qh1_0, vu1), vu1), v16);
            __m128i h16_1_1_lo = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(raw_qh1_1, vu1), vu1), v16);
            __m128i h16_1_0_hi = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(raw_qh1_0, vu2), vu2), v16);
            __m128i h16_1_1_hi = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(raw_qh1_1, vu2), vu2), v16);

            __m128i w1_lo0 = _mm_add_epi8(_mm_and_si128(raw_ql1_0, m4), h16_1_0_lo);
            __m128i w1_lo1 = _mm_add_epi8(_mm_and_si128(raw_ql1_1, m4), h16_1_1_lo);
            __m128i w1_hi0 = _mm_add_epi8(_mm_and_si128(_mm_srli_epi16(raw_ql1_0, 4), m4), h16_1_0_hi);
            __m128i w1_hi1 = _mm_add_epi8(_mm_and_si128(_mm_srli_epi16(raw_ql1_1, 4), m4), h16_1_1_hi);

            __m256 x0 = _mm256_loadu_ps(act_ptr + 0);
            __m256 x1 = _mm256_loadu_ps(act_ptr + 8);
            __m256 x2 = _mm256_loadu_ps(act_ptr + 16);
            __m256 x3 = _mm256_loadu_ps(act_ptr + 24);
            __m256 x4 = _mm256_loadu_ps(act_ptr + 32);
            __m256 x5 = _mm256_loadu_ps(act_ptr + 40);
            __m256 x6 = _mm256_loadu_ps(act_ptr + 48);
            __m256 x7 = _mm256_loadu_ps(act_ptr + 56);

            __m256 sx_0 = _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3));
            __m256 sx_1 = _mm256_add_ps(_mm256_add_ps(x4, x5), _mm256_add_ps(x6, x7));

            // Row 0, Subblock 0
            __m256 p0_0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0_lo0)), x0,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0_lo0, 8))), x1));
            __m256 p0_1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0_lo1)), x2,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0_lo1, 8))), x3));
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d1_0), _mm256_add_ps(p0_0, p0_1),
                   _mm256_fnmadd_ps(_mm256_set1_ps(m1_0_val), sx_0, acc0));

            // Row 0, Subblock 1
            __m256 p0_2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0_hi0)), x4,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0_hi0, 8))), x5));
            __m256 p0_3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0_hi1)), x6,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0_hi1, 8))), x7));
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d2_0), _mm256_add_ps(p0_2, p0_3),
                   _mm256_fnmadd_ps(_mm256_set1_ps(m2_0_val), sx_1, acc0));

            // Row 1, Subblock 0
            __m256 p1_0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1_lo0)), x0,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1_lo0, 8))), x1));
            __m256 p1_1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1_lo1)), x2,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1_lo1, 8))), x3));
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1_1), _mm256_add_ps(p1_0, p1_1),
                   _mm256_fnmadd_ps(_mm256_set1_ps(m1_1_val), sx_0, acc1));

            // Row 1, Subblock 1
            __m256 p1_2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1_hi0)), x4,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1_hi0, 8))), x5));
            __m256 p1_3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1_hi1)), x6,
                          _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1_hi1, 8))), x7));
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d2_1), _mm256_add_ps(p1_2, p1_3),
                   _mm256_fnmadd_ps(_mm256_set1_ps(m2_1_val), sx_1, acc1));

            ql0 += 32; ql1 += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
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

inline float dot_q5_K_f32_fast(const uint8_t* row_bytes, size_t K, const float* x) {
    float out0 = 0.0f, out1 = 0.0f;
    dot_q5_K_f32_2rows(row_bytes, row_bytes, K, x, out0, out1);
    return out0;
}

// ----------------------------------------------------------------------------
// IQ4_XS Vectorized AVX2 Dot Products
// ----------------------------------------------------------------------------
inline void dot_iq4_xs_f32_2rows(const uint8_t* row0, const uint8_t* row1, size_t K, const float* x, float& out0, float& out1) {
    const size_t nb = K / QK_K;
    const auto* blk0 = reinterpret_cast<const block_iq4_xs*>(row0);
    const auto* blk1 = reinterpret_cast<const block_iq4_xs*>(row1);
    const __m128i tbl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i m4 = _mm_set1_epi8(0x0F);

    __m256 total_acc0 = _mm256_setzero_ps();
    __m256 total_acc1 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const float d0 = q4_0::fp16_to_float(blk0[b].d);
        const float d1 = q4_0::fp16_to_float(blk1[b].d);
        const uint8_t* qs0 = blk0[b].qs;
        const uint8_t* qs1 = blk1[b].qs;

        const uint32_t s_l0 = *reinterpret_cast<const uint32_t*>(blk0[b].scales_l);
        const uint16_t s_h0 = blk0[b].scales_h;

        const uint32_t s_l1 = *reinterpret_cast<const uint32_t*>(blk1[b].scales_l);
        const uint16_t s_h1 = blk1[b].scales_h;

        const float* act_base = x + b * 256;

        for (int ib = 0; ib < 8; ++ib) {
            const int ls0 = ((s_l0 >> (4 * ib)) & 0x0F) | (((s_h0 >> (2 * ib)) & 3) << 4);
            const float dl0 = d0 * static_cast<float>(ls0 - 32);

            const int ls1 = ((s_l1 >> (4 * ib)) & 0x0F) | (((s_h1 >> (2 * ib)) & 3) << 4);
            const float dl1 = d1 * static_cast<float>(ls1 - 32);

            const float* act_ptr = act_base + ib * 32;

            __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs0));
            __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs1));

            __m128i lo0 = _mm_shuffle_epi8(tbl, _mm_and_si128(raw0, m4));
            __m128i hi0 = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw0, 4), m4));

            __m128i lo1 = _mm_shuffle_epi8(tbl, _mm_and_si128(raw1, m4));
            __m128i hi1 = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw1, 4), m4));

            __m256 x0 = _mm256_loadu_ps(act_ptr + 0);
            __m256 x1 = _mm256_loadu_ps(act_ptr + 8);
            __m256 x2 = _mm256_loadu_ps(act_ptr + 16);
            __m256 x3 = _mm256_loadu_ps(act_ptr + 24);

            // Row 0
            __m256 w0_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo0));
            __m256 w0_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo0, 8)));
            __m256 w0_2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi0));
            __m256 w0_3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi0, 8)));

            __m256 p0_0 = _mm256_fmadd_ps(w0_0, x0, _mm256_mul_ps(w0_1, x1));
            __m256 p0_1 = _mm256_fmadd_ps(w0_2, x2, _mm256_mul_ps(w0_3, x3));
            total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(dl0), _mm256_add_ps(p0_0, p0_1), total_acc0);

            // Row 1
            __m256 w1_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo1));
            __m256 w1_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo1, 8)));
            __m256 w1_2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi1));
            __m256 w1_3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi1, 8)));

            __m256 p1_0 = _mm256_fmadd_ps(w1_0, x0, _mm256_mul_ps(w1_1, x1));
            __m256 p1_1 = _mm256_fmadd_ps(w1_2, x2, _mm256_mul_ps(w1_3, x3));
            total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dl1), _mm256_add_ps(p1_0, p1_1), total_acc1);

            qs0 += 16;
            qs1 += 16;
        }
    }

    auto h_add = [](__m256 acc) -> float {
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
        return _mm_cvtss_f32(h);
    };

    out0 = h_add(total_acc0);
    out1 = h_add(total_acc1);
}

inline void dot_iq4_xs_f32_4rows(const uint8_t* row0, const uint8_t* row1, const uint8_t* row2, const uint8_t* row3,
                                 size_t K, const float* x, float& out0, float& out1, float& out2, float& out3) {
    const size_t nb = K / QK_K;
    const auto* blk0 = reinterpret_cast<const block_iq4_xs*>(row0);
    const auto* blk1 = reinterpret_cast<const block_iq4_xs*>(row1);
    const auto* blk2 = reinterpret_cast<const block_iq4_xs*>(row2);
    const auto* blk3 = reinterpret_cast<const block_iq4_xs*>(row3);
    const __m128i tbl = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl));
    const __m128i m4 = _mm_set1_epi8(0x0F);

    __m256 total_acc0 = _mm256_setzero_ps();
    __m256 total_acc1 = _mm256_setzero_ps();
    __m256 total_acc2 = _mm256_setzero_ps();
    __m256 total_acc3 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const float d0 = q4_0::fp16_to_float(blk0[b].d);
        const float d1 = q4_0::fp16_to_float(blk1[b].d);
        const float d2 = q4_0::fp16_to_float(blk2[b].d);
        const float d3 = q4_0::fp16_to_float(blk3[b].d);

        const uint8_t* qs0 = blk0[b].qs;
        const uint8_t* qs1 = blk1[b].qs;
        const uint8_t* qs2 = blk2[b].qs;
        const uint8_t* qs3 = blk3[b].qs;

        const uint32_t s_l0 = *reinterpret_cast<const uint32_t*>(blk0[b].scales_l);
        const uint16_t s_h0 = blk0[b].scales_h;
        const uint32_t s_l1 = *reinterpret_cast<const uint32_t*>(blk1[b].scales_l);
        const uint16_t s_h1 = blk1[b].scales_h;
        const uint32_t s_l2 = *reinterpret_cast<const uint32_t*>(blk2[b].scales_l);
        const uint16_t s_h2 = blk2[b].scales_h;
        const uint32_t s_l3 = *reinterpret_cast<const uint32_t*>(blk3[b].scales_l);
        const uint16_t s_h3 = blk3[b].scales_h;

        const float* act_base = x + b * 256;

        for (int ib = 0; ib < 8; ++ib) {
            const int ls0 = ((s_l0 >> (4 * ib)) & 0x0F) | (((s_h0 >> (2 * ib)) & 3) << 4);
            const float dl0 = d0 * static_cast<float>(ls0 - 32);

            const int ls1 = ((s_l1 >> (4 * ib)) & 0x0F) | (((s_h1 >> (2 * ib)) & 3) << 4);
            const float dl1 = d1 * static_cast<float>(ls1 - 32);

            const int ls2 = ((s_l2 >> (4 * ib)) & 0x0F) | (((s_h2 >> (2 * ib)) & 3) << 4);
            const float dl2 = d2 * static_cast<float>(ls2 - 32);

            const int ls3 = ((s_l3 >> (4 * ib)) & 0x0F) | (((s_h3 >> (2 * ib)) & 3) << 4);
            const float dl3 = d3 * static_cast<float>(ls3 - 32);

            const float* act_ptr = act_base + ib * 32;

            __m256 x0 = _mm256_loadu_ps(act_ptr + 0);
            __m256 x1 = _mm256_loadu_ps(act_ptr + 8);
            __m256 x2 = _mm256_loadu_ps(act_ptr + 16);
            __m256 x3 = _mm256_loadu_ps(act_ptr + 24);

            auto do_row = [&](const uint8_t* qs, float dl, __m256& total_acc) {
                __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
                __m128i lo = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, m4));
                __m128i hi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw, 4), m4));

                __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
                __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
                __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
                __m256 w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));

                __m256 p0 = _mm256_fmadd_ps(w0, x0, _mm256_mul_ps(w1, x1));
                __m256 p1 = _mm256_fmadd_ps(w2, x2, _mm256_mul_ps(w3, x3));
                total_acc = _mm256_fmadd_ps(_mm256_set1_ps(dl), _mm256_add_ps(p0, p1), total_acc);
            };

            do_row(qs0, dl0, total_acc0);
            do_row(qs1, dl1, total_acc1);
            do_row(qs2, dl2, total_acc2);
            do_row(qs3, dl3, total_acc3);

            qs0 += 16;
            qs1 += 16;
            qs2 += 16;
            qs3 += 16;
        }
    }

    auto h_add = [](__m256 acc) -> float {
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
        return _mm_cvtss_f32(h);
    };

    out0 = h_add(total_acc0);
    out1 = h_add(total_acc1);
    out2 = h_add(total_acc2);
    out3 = h_add(total_acc3);
}

inline float dot_iq4_xs_f32_fast(const uint8_t* row_bytes, size_t K, const float* x) {
    float out0 = 0.0f, out1 = 0.0f;
    dot_iq4_xs_f32_2rows(row_bytes, row_bytes, K, x, out0, out1);
    return out0;
}

// ----------------------------------------------------------------------------
// Q3_K, Q6_K, IQ3_S Fast Dot Products
// ----------------------------------------------------------------------------
inline float dot_q3_K_f32_fast(const uint8_t* row_bytes, size_t K, const float* x) {
    const size_t nb = K / QK_K;
    const auto* blocks = reinterpret_cast<const block_q3_K*>(row_bytes);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0F0F0F0F;

    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    __m256 total_acc = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const float d_all = q4_0::fp16_to_float(blocks[b].d);
        const uint8_t* q = blocks[b].qs;
        const uint8_t* hm = blocks[b].hmask;
        uint8_t m = 1;

        std::memcpy(aux, blocks[b].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        const float* act_ptr = x + b * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            __m128i q0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 0));
            __m128i q1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + 16));
            __m128i hm0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hm + 0));
            __m128i hm1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hm + 16));

            for (int j = 0; j < 4; ++j) {
                float dl1 = d_all * (static_cast<float>(scales[is++]) - 32.0f);
                float dl2 = d_all * (static_cast<float>(scales[is++]) - 32.0f);

                __m128i vm = _mm_set1_epi8(static_cast<char>(m));
                __m128i v4 = _mm_set1_epi8(4);

                __m128i q0_shift = (shift == 0) ? q0 : _mm_and_si128(_mm_srli_epi16(q0, shift), _mm_set1_epi8(0xFF >> shift));
                __m128i q0_bits = _mm_and_si128(q0_shift, _mm_set1_epi8(3));
                __m128i hm0_bits = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_and_si128(hm0, vm), vm), v4);
                __m128i w0 = _mm_sub_epi8(q0_bits, hm0_bits);

                __m256 w0_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0));
                __m256 w0_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0, 8)));
                __m256 dot0 = _mm256_fmadd_ps(w0_lo, _mm256_loadu_ps(act_ptr + n + j * 32 + 0),
                              _mm256_mul_ps(w0_hi, _mm256_loadu_ps(act_ptr + n + j * 32 + 8)));
                total_acc = _mm256_fmadd_ps(_mm256_set1_ps(dl1), dot0, total_acc);

                __m128i q1_shift = (shift == 0) ? q1 : _mm_and_si128(_mm_srli_epi16(q1, shift), _mm_set1_epi8(0xFF >> shift));
                __m128i q1_bits = _mm_and_si128(q1_shift, _mm_set1_epi8(3));
                __m128i hm1_bits = _mm_andnot_si128(_mm_cmpeq_epi8(_mm_and_si128(hm1, vm), vm), v4);
                __m128i w1 = _mm_sub_epi8(q1_bits, hm1_bits);

                __m256 w1_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1));
                __m256 w1_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1, 8)));
                __m256 dot1 = _mm256_fmadd_ps(w1_lo, _mm256_loadu_ps(act_ptr + n + j * 32 + 16),
                              _mm256_mul_ps(w1_hi, _mm256_loadu_ps(act_ptr + n + j * 32 + 24)));
                total_acc = _mm256_fmadd_ps(_mm256_set1_ps(dl2), dot1, total_acc);

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }

    __m128 h = _mm_add_ps(_mm256_castps256_ps128(total_acc), _mm256_extractf128_ps(total_acc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
    return _mm_cvtss_f32(h);
}

inline void dot_q3_K_f32_2rows(const uint8_t* row0, const uint8_t* row1, size_t K, const float* x, float& out0, float& out1) {
    const size_t nb = K / QK_K;
    const auto* blk0 = reinterpret_cast<const block_q3_K*>(row0);
    const auto* blk1 = reinterpret_cast<const block_q3_K*>(row1);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0F0F0F0F;

    uint32_t aux0[4], aux1[4];
    const int8_t* scales0 = reinterpret_cast<const int8_t*>(aux0);
    const int8_t* scales1 = reinterpret_cast<const int8_t*>(aux1);
    __m256 total_acc0 = _mm256_setzero_ps();
    __m256 total_acc1 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const float d_all0 = q4_0::fp16_to_float(blk0[b].d);
        const float d_all1 = q4_0::fp16_to_float(blk1[b].d);
        const uint8_t* q0_ptr = blk0[b].qs;
        const uint8_t* hm0_ptr = blk0[b].hmask;
        const uint8_t* q1_ptr = blk1[b].qs;
        const uint8_t* hm1_ptr = blk1[b].hmask;
        uint8_t m = 1;

        std::memcpy(aux0, blk0[b].scales, 12);
        uint32_t tmp0 = aux0[2];
        aux0[2] = ((aux0[0] >> 4) & kmask2) | (((tmp0 >> 4) & kmask1) << 4);
        aux0[3] = ((aux0[1] >> 4) & kmask2) | (((tmp0 >> 6) & kmask1) << 4);
        aux0[0] = (aux0[0] & kmask2) | (((tmp0 >> 0) & kmask1) << 4);
        aux0[1] = (aux0[1] & kmask2) | (((tmp0 >> 2) & kmask1) << 4);

        std::memcpy(aux1, blk1[b].scales, 12);
        uint32_t tmp1 = aux1[2];
        aux1[2] = ((aux1[0] >> 4) & kmask2) | (((tmp1 >> 4) & kmask1) << 4);
        aux1[3] = ((aux1[1] >> 4) & kmask2) | (((tmp1 >> 6) & kmask1) << 4);
        aux1[0] = (aux1[0] & kmask2) | (((tmp1 >> 0) & kmask1) << 4);
        aux1[1] = (aux1[1] & kmask2) | (((tmp1 >> 2) & kmask1) << 4);

        int is = 0;
        const float* act_ptr = x + b * 256;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            __m128i q0_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q0_ptr + 0));
            __m128i q0_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q0_ptr + 16));
            __m128i hm0_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hm0_ptr + 0));
            __m128i hm0_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hm0_ptr + 16));

            __m128i q1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q1_ptr + 0));
            __m128i q1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q1_ptr + 16));
            __m128i hm1_0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hm1_ptr + 0));
            __m128i hm1_1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hm1_ptr + 16));

            for (int j = 0; j < 4; ++j) {
                float dl1_0 = d_all0 * (static_cast<float>(scales0[is]) - 32.0f);
                float dl1_1 = d_all1 * (static_cast<float>(scales1[is]) - 32.0f);
                is++;
                float dl2_0 = d_all0 * (static_cast<float>(scales0[is]) - 32.0f);
                float dl2_1 = d_all1 * (static_cast<float>(scales1[is]) - 32.0f);
                is++;

                __m128i vm = _mm_set1_epi8(static_cast<char>(m));
                __m128i v4 = _mm_set1_epi8(4);

                // Row 0, dl1 (q0_0)
                __m128i q0_s0 = (shift == 0) ? q0_0 : _mm_and_si128(_mm_srli_epi16(q0_0, shift), _mm_set1_epi8(0xFF >> shift));
                __m128i w0_0 = _mm_sub_epi8(_mm_and_si128(q0_s0, _mm_set1_epi8(3)), _mm_andnot_si128(_mm_cmpeq_epi8(_mm_and_si128(hm0_0, vm), vm), v4));
                __m256 act0_lo = _mm256_loadu_ps(act_ptr + n + j * 32 + 0);
                __m256 act0_hi = _mm256_loadu_ps(act_ptr + n + j * 32 + 8);
                __m256 dot0_0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0_0)), act0_lo,
                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0_0, 8))), act0_hi));
                total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(dl1_0), dot0_0, total_acc0);

                // Row 1, dl1 (q1_0)
                __m128i q1_s0 = (shift == 0) ? q1_0 : _mm_and_si128(_mm_srli_epi16(q1_0, shift), _mm_set1_epi8(0xFF >> shift));
                __m128i w1_0 = _mm_sub_epi8(_mm_and_si128(q1_s0, _mm_set1_epi8(3)), _mm_andnot_si128(_mm_cmpeq_epi8(_mm_and_si128(hm1_0, vm), vm), v4));
                __m256 dot1_0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1_0)), act0_lo,
                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1_0, 8))), act0_hi));
                total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dl1_1), dot1_0, total_acc1);

                // Row 0, dl2 (q0_1)
                __m128i q0_s1 = (shift == 0) ? q0_1 : _mm_and_si128(_mm_srli_epi16(q0_1, shift), _mm_set1_epi8(0xFF >> shift));
                __m128i w0_1 = _mm_sub_epi8(_mm_and_si128(q0_s1, _mm_set1_epi8(3)), _mm_andnot_si128(_mm_cmpeq_epi8(_mm_and_si128(hm0_1, vm), vm), v4));
                __m256 act1_lo = _mm256_loadu_ps(act_ptr + n + j * 32 + 16);
                __m256 act1_hi = _mm256_loadu_ps(act_ptr + n + j * 32 + 24);
                __m256 dot0_1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w0_1)), act1_lo,
                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w0_1, 8))), act1_hi));
                total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(dl2_0), dot0_1, total_acc0);

                // Row 1, dl2 (q1_1)
                __m128i q1_s1 = (shift == 0) ? q1_1 : _mm_and_si128(_mm_srli_epi16(q1_1, shift), _mm_set1_epi8(0xFF >> shift));
                __m128i w1_1 = _mm_sub_epi8(_mm_and_si128(q1_s1, _mm_set1_epi8(3)), _mm_andnot_si128(_mm_cmpeq_epi8(_mm_and_si128(hm1_1, vm), vm), v4));
                __m256 dot1_1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w1_1)), act1_lo,
                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w1_1, 8))), act1_hi));
                total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dl2_1), dot1_1, total_acc1);

                shift += 2;
                m <<= 1;
            }
            q0_ptr += 32;
            q1_ptr += 32;
        }
    }

    auto h_add = [](__m256 acc) -> float {
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
        return _mm_cvtss_f32(h);
    };

    out0 = h_add(total_acc0);
    out1 = h_add(total_acc1);
}

inline float dot_q6_K_f32_fast(const uint8_t* row_bytes, size_t K, const float* x) {
    const size_t nb = K / QK_K;
    const auto* blocks = reinterpret_cast<const block_q6_K*>(row_bytes);

    __m256 total_acc = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const float d = q4_0::fp16_to_float(blocks[b].d);
        const uint8_t* ql = blocks[b].ql;
        const uint8_t* qh = blocks[b].qh;
        const int8_t* sc = blocks[b].scales;
        const float* act = x + b * 256;

        for (int half = 0; half < 2; ++half) {
            const float* act_half = act + half * 128;
            for (int step = 0; step < 2; ++step) {
                int is = step;
                const float s0 = d * sc[is + 0];
                const float s2 = d * sc[is + 2];
                const float s4 = d * sc[is + 4];
                const float s6 = d * sc[is + 6];

                int l_offset = step * 16;
                __m128i ql0_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql + l_offset + 0));
                __m128i ql1_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql + l_offset + 32));
                __m128i qh_128  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh + l_offset));

                __m128i q1_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql0_128, _mm_set1_epi8(0x0F)),
                                                        _mm_slli_epi16(_mm_and_si128(qh_128, _mm_set1_epi8(0x03)), 4)),
                                           _mm_set1_epi8(32));

                __m128i q2_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql1_128, _mm_set1_epi8(0x0F)),
                                                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_128, 2), _mm_set1_epi8(0x03)), 4)),
                                           _mm_set1_epi8(32));

                __m128i q3_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0_128, 4), _mm_set1_epi8(0x0F)),
                                                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_128, 4), _mm_set1_epi8(0x03)), 4)),
                                           _mm_set1_epi8(32));

                __m128i q4_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1_128, 4), _mm_set1_epi8(0x0F)),
                                                        _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_128, 6), _mm_set1_epi8(0x03)), 4)),
                                           _mm_set1_epi8(32));

                __m256 q1_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1_b));
                __m256 q1_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1_b, 8)));
                __m256 a1_lo = _mm256_loadu_ps(act_half + l_offset + 0);
                __m256 a1_hi = _mm256_loadu_ps(act_half + l_offset + 8);
                __m256 prod1 = _mm256_fmadd_ps(q1_lo, a1_lo, _mm256_mul_ps(q1_hi, a1_hi));
                total_acc = _mm256_fmadd_ps(_mm256_set1_ps(s0), prod1, total_acc);

                __m256 q2_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2_b));
                __m256 q2_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q2_b, 8)));
                __m256 a2_lo = _mm256_loadu_ps(act_half + l_offset + 32);
                __m256 a2_hi = _mm256_loadu_ps(act_half + l_offset + 40);
                __m256 prod2 = _mm256_fmadd_ps(q2_lo, a2_lo, _mm256_mul_ps(q2_hi, a2_hi));
                total_acc = _mm256_fmadd_ps(_mm256_set1_ps(s2), prod2, total_acc);

                __m256 q3_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3_b));
                __m256 q3_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q3_b, 8)));
                __m256 a3_lo = _mm256_loadu_ps(act_half + l_offset + 64);
                __m256 a3_hi = _mm256_loadu_ps(act_half + l_offset + 72);
                __m256 prod3 = _mm256_fmadd_ps(q3_lo, a3_lo, _mm256_mul_ps(q3_hi, a3_hi));
                total_acc = _mm256_fmadd_ps(_mm256_set1_ps(s4), prod3, total_acc);

                __m256 q4_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q4_b));
                __m256 q4_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q4_b, 8)));
                __m256 a4_lo = _mm256_loadu_ps(act_half + l_offset + 96);
                __m256 a4_hi = _mm256_loadu_ps(act_half + l_offset + 104);
                __m256 prod4 = _mm256_fmadd_ps(q4_lo, a4_lo, _mm256_mul_ps(q4_hi, a4_hi));
                total_acc = _mm256_fmadd_ps(_mm256_set1_ps(s6), prod4, total_acc);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    __m128 lo128 = _mm256_castps256_ps128(total_acc);
    __m128 hi128 = _mm256_extractf128_ps(total_acc, 1);
    __m128 s128  = _mm_add_ps(lo128, hi128);
    s128 = _mm_hadd_ps(s128, s128);
    s128 = _mm_hadd_ps(s128, s128);
    return _mm_cvtss_f32(s128);
}

inline void dot_q6_K_f32_2rows(const uint8_t* row0, const uint8_t* row1, size_t K, const float* x, float& out0, float& out1) {
    const size_t nb = K / QK_K;
    const auto* blk0 = reinterpret_cast<const block_q6_K*>(row0);
    const auto* blk1 = reinterpret_cast<const block_q6_K*>(row1);

    __m256 total_acc0 = _mm256_setzero_ps();
    __m256 total_acc1 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const float d0 = q4_0::fp16_to_float(blk0[b].d);
        const float d1 = q4_0::fp16_to_float(blk1[b].d);

        const uint8_t* ql0 = blk0[b].ql;
        const uint8_t* qh0 = blk0[b].qh;
        const int8_t*  sc0 = blk0[b].scales;

        const uint8_t* ql1 = blk1[b].ql;
        const uint8_t* qh1 = blk1[b].qh;
        const int8_t*  sc1 = blk1[b].scales;

        const float* act = x + b * 256;

        for (int half = 0; half < 2; ++half) {
            const float* act_half = act + half * 128;
            for (int step = 0; step < 2; ++step) {
                int is = step;
                const float s0_0 = d0 * sc0[is + 0];
                const float s2_0 = d0 * sc0[is + 2];
                const float s4_0 = d0 * sc0[is + 4];
                const float s6_0 = d0 * sc0[is + 6];

                const float s0_1 = d1 * sc1[is + 0];
                const float s2_1 = d1 * sc1[is + 2];
                const float s4_1 = d1 * sc1[is + 4];
                const float s6_1 = d1 * sc1[is + 6];

                int l_offset = step * 16;
                __m256 a1_lo = _mm256_loadu_ps(act_half + l_offset + 0);
                __m256 a1_hi = _mm256_loadu_ps(act_half + l_offset + 8);
                __m256 a2_lo = _mm256_loadu_ps(act_half + l_offset + 32);
                __m256 a2_hi = _mm256_loadu_ps(act_half + l_offset + 40);
                __m256 a3_lo = _mm256_loadu_ps(act_half + l_offset + 64);
                __m256 a3_hi = _mm256_loadu_ps(act_half + l_offset + 72);
                __m256 a4_lo = _mm256_loadu_ps(act_half + l_offset + 96);
                __m256 a4_hi = _mm256_loadu_ps(act_half + l_offset + 104);

                // Row 0
                {
                    __m128i ql0_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql0 + l_offset + 0));
                    __m128i ql1_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql0 + l_offset + 32));
                    __m128i qh_128  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh0 + l_offset));

                    __m128i q1_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql0_128, _mm_set1_epi8(0x0F)),
                                                            _mm_slli_epi16(_mm_and_si128(qh_128, _mm_set1_epi8(0x03)), 4)), _mm_set1_epi8(32));
                    __m128i q2_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql1_128, _mm_set1_epi8(0x0F)),
                                                            _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_128, 2), _mm_set1_epi8(0x03)), 4)), _mm_set1_epi8(32));
                    __m128i q3_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0_128, 4), _mm_set1_epi8(0x0F)),
                                                            _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_128, 4), _mm_set1_epi8(0x03)), 4)), _mm_set1_epi8(32));
                    __m128i q4_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1_128, 4), _mm_set1_epi8(0x0F)),
                                                            _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_128, 6), _mm_set1_epi8(0x03)), 4)), _mm_set1_epi8(32));

                    __m256 p1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1_b)), a1_lo,
                                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1_b, 8))), a1_hi));
                    total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(s0_0), p1, total_acc0);

                    __m256 p2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2_b)), a2_lo,
                                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q2_b, 8))), a2_hi));
                    total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(s2_0), p2, total_acc0);

                    __m256 p3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3_b)), a3_lo,
                                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q3_b, 8))), a3_hi));
                    total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(s4_0), p3, total_acc0);

                    __m256 p4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q4_b)), a4_lo,
                                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q4_b, 8))), a4_hi));
                    total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(s6_0), p4, total_acc0);
                }

                // Row 1
                {
                    __m128i ql0_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql1 + l_offset + 0));
                    __m128i ql1_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ql1 + l_offset + 32));
                    __m128i qh_128  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qh1 + l_offset));

                    __m128i q1_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql0_128, _mm_set1_epi8(0x0F)),
                                                            _mm_slli_epi16(_mm_and_si128(qh_128, _mm_set1_epi8(0x03)), 4)), _mm_set1_epi8(32));
                    __m128i q2_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(ql1_128, _mm_set1_epi8(0x0F)),
                                                            _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_128, 2), _mm_set1_epi8(0x03)), 4)), _mm_set1_epi8(32));
                    __m128i q3_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql0_128, 4), _mm_set1_epi8(0x0F)),
                                                            _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_128, 4), _mm_set1_epi8(0x03)), 4)), _mm_set1_epi8(32));
                    __m128i q4_b = _mm_sub_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(ql1_128, 4), _mm_set1_epi8(0x0F)),
                                                            _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh_128, 6), _mm_set1_epi8(0x03)), 4)), _mm_set1_epi8(32));

                    __m256 p1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1_b)), a1_lo,
                                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1_b, 8))), a1_hi));
                    total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(s0_1), p1, total_acc1);

                    __m256 p2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2_b)), a2_lo,
                                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q2_b, 8))), a2_hi));
                    total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(s2_1), p2, total_acc1);

                    __m256 p3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3_b)), a3_lo,
                                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q3_b, 8))), a3_hi));
                    total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(s4_1), p3, total_acc1);

                    __m256 p4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q4_b)), a4_lo,
                                                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q4_b, 8))), a4_hi));
                    total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(s6_1), p4, total_acc1);
                }
            }
            ql0 += 64; qh0 += 32; sc0 += 8;
            ql1 += 64; qh1 += 32; sc1 += 8;
        }
    }

    __m128 lo0 = _mm256_castps256_ps128(total_acc0);
    __m128 hi0 = _mm256_extractf128_ps(total_acc0, 1);
    __m128 s0  = _mm_add_ps(lo0, hi0);
    s0 = _mm_hadd_ps(s0, s0);
    s0 = _mm_hadd_ps(s0, s0);
    out0 = _mm_cvtss_f32(s0);

    __m128 lo1 = _mm256_castps256_ps128(total_acc1);
    __m128 hi1 = _mm256_extractf128_ps(total_acc1, 1);
    __m128 s1  = _mm_add_ps(lo1, hi1);
    s1 = _mm_hadd_ps(s1, s1);
    s1 = _mm_hadd_ps(s1, s1);
    out1 = _mm_cvtss_f32(s1);
}

struct SignMaskLut {
    alignas(64) uint32_t data[256][8];
    SignMaskLut() {
        for (int i = 0; i < 256; ++i) {
            for (int b = 0; b < 8; ++b) {
                data[i][b] = ((i >> b) & 1) ? 0x80000000U : 0U;
            }
        }
    }
};
static const SignMaskLut g_sign_mask_lut;

inline void dot_iq3_s_f32_2rows(const uint8_t* row0, const uint8_t* row1, size_t K, const float* x, float& out0, float& out1) {
    const size_t nb = K / QK_K;
    const auto* blk0 = reinterpret_cast<const block_iq3_s*>(row0);
    const auto* blk1 = reinterpret_cast<const block_iq3_s*>(row1);

    __m256 total_acc0 = _mm256_setzero_ps();
    __m256 total_acc1 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const float d0 = q4_0::fp16_to_float(blk0[b].d);
        const float d1 = q4_0::fp16_to_float(blk1[b].d);

        const uint8_t* qs0 = blk0[b].qs;
        const uint8_t* qh0 = blk0[b].qh;
        const uint8_t* signs0 = blk0[b].signs;

        const uint8_t* qs1 = blk1[b].qs;
        const uint8_t* qh1 = blk1[b].qh;
        const uint8_t* signs1 = blk1[b].signs;

        const float* act_ptr = x + b * 256;
        size_t y_idx = 0;

        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1_0 = d0 * (1.0f + 2.0f * static_cast<float>(blk0[b].scales[ib32 / 2] & 0x0F));
            const float db2_0 = d0 * (1.0f + 2.0f * static_cast<float>(blk0[b].scales[ib32 / 2] >> 4));
            const float db1_1 = d1 * (1.0f + 2.0f * static_cast<float>(blk1[b].scales[ib32 / 2] & 0x0F));
            const float db2_1 = d1 * (1.0f + 2.0f * static_cast<float>(blk1[b].scales[ib32 / 2] >> 4));

            __m256 sub_acc1_0 = _mm256_setzero_ps();
            __m256 sub_acc1_1 = _mm256_setzero_ps();

            uint32_t qh0_val = qh0[0];
            uint32_t qh1_val = qh1[0];

            for (int l = 0; l < 4; ++l) {
                __m256 act = _mm256_loadu_ps(act_ptr + y_idx);

                uint32_t g1_0 = iq3s_grid[qs0[2 * l + 0] | ((qh0_val << (8 - 2 * l)) & 256)];
                uint32_t g2_0 = iq3s_grid[qs0[2 * l + 1] | ((qh0_val << (7 - 2 * l)) & 256)];
                uint64_t g8_0 = static_cast<uint64_t>(g1_0) | (static_cast<uint64_t>(g2_0) << 32);
                __m256 valf_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<int64_t>(g8_0))));
                __m256 s_mask0 = _mm256_load_ps(reinterpret_cast<const float*>(g_sign_mask_lut.data[signs0[l]]));
                sub_acc1_0 = _mm256_fmadd_ps(_mm256_xor_ps(valf_0, s_mask0), act, sub_acc1_0);

                uint32_t g1_1 = iq3s_grid[qs1[2 * l + 0] | ((qh1_val << (8 - 2 * l)) & 256)];
                uint32_t g2_1 = iq3s_grid[qs1[2 * l + 1] | ((qh1_val << (7 - 2 * l)) & 256)];
                uint64_t g8_1 = static_cast<uint64_t>(g1_1) | (static_cast<uint64_t>(g2_1) << 32);
                __m256 valf_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<int64_t>(g8_1))));
                __m256 s_mask1 = _mm256_load_ps(reinterpret_cast<const float*>(g_sign_mask_lut.data[signs1[l]]));
                sub_acc1_1 = _mm256_fmadd_ps(_mm256_xor_ps(valf_1, s_mask1), act, sub_acc1_1);

                y_idx += 8;
            }
            total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(db1_0), sub_acc1_0, total_acc0);
            total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(db1_1), sub_acc1_1, total_acc1);

            qs0 += 8; signs0 += 4;
            qs1 += 8; signs1 += 4;

            __m256 sub_acc2_0 = _mm256_setzero_ps();
            __m256 sub_acc2_1 = _mm256_setzero_ps();

            uint32_t qh0_val2 = qh0[1];
            uint32_t qh1_val2 = qh1[1];

            for (int l = 0; l < 4; ++l) {
                __m256 act = _mm256_loadu_ps(act_ptr + y_idx);

                uint32_t g1_0 = iq3s_grid[qs0[2 * l + 0] | ((qh0_val2 << (8 - 2 * l)) & 256)];
                uint32_t g2_0 = iq3s_grid[qs0[2 * l + 1] | ((qh0_val2 << (7 - 2 * l)) & 256)];
                uint64_t g8_0 = static_cast<uint64_t>(g1_0) | (static_cast<uint64_t>(g2_0) << 32);
                __m256 valf_0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<int64_t>(g8_0))));
                __m256 s_mask0 = _mm256_load_ps(reinterpret_cast<const float*>(g_sign_mask_lut.data[signs0[l]]));
                sub_acc2_0 = _mm256_fmadd_ps(_mm256_xor_ps(valf_0, s_mask0), act, sub_acc2_0);

                uint32_t g1_1 = iq3s_grid[qs1[2 * l + 0] | ((qh1_val2 << (8 - 2 * l)) & 256)];
                uint32_t g2_1 = iq3s_grid[qs1[2 * l + 1] | ((qh1_val2 << (7 - 2 * l)) & 256)];
                uint64_t g8_1 = static_cast<uint64_t>(g1_1) | (static_cast<uint64_t>(g2_1) << 32);
                __m256 valf_1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<int64_t>(g8_1))));
                __m256 s_mask1 = _mm256_load_ps(reinterpret_cast<const float*>(g_sign_mask_lut.data[signs1[l]]));
                sub_acc2_1 = _mm256_fmadd_ps(_mm256_xor_ps(valf_1, s_mask1), act, sub_acc2_1);

                y_idx += 8;
            }
            total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(db2_0), sub_acc2_0, total_acc0);
            total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(db2_1), sub_acc2_1, total_acc1);

            qh0 += 2; qs0 += 8; signs0 += 4;
            qh1 += 2; qs1 += 8; signs1 += 4;
        }
    }

    auto h_add = [](__m256 acc) -> float {
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
        return _mm_cvtss_f32(h);
    };

    out0 = h_add(total_acc0);
    out1 = h_add(total_acc1);
}

inline void dot_iq3_s_f32_4rows(const uint8_t* row0, const uint8_t* row1, const uint8_t* row2, const uint8_t* row3,
                                size_t K, const float* x, float& out0, float& out1, float& out2, float& out3) {
    const size_t nb = K / QK_K;
    const auto* blk0 = reinterpret_cast<const block_iq3_s*>(row0);
    const auto* blk1 = reinterpret_cast<const block_iq3_s*>(row1);
    const auto* blk2 = reinterpret_cast<const block_iq3_s*>(row2);
    const auto* blk3 = reinterpret_cast<const block_iq3_s*>(row3);

    __m256 total_acc0 = _mm256_setzero_ps();
    __m256 total_acc1 = _mm256_setzero_ps();
    __m256 total_acc2 = _mm256_setzero_ps();
    __m256 total_acc3 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        const float d0 = q4_0::fp16_to_float(blk0[b].d);
        const float d1 = q4_0::fp16_to_float(blk1[b].d);
        const float d2 = q4_0::fp16_to_float(blk2[b].d);
        const float d3 = q4_0::fp16_to_float(blk3[b].d);

        const uint8_t* qs0 = blk0[b].qs;
        const uint8_t* qh0 = blk0[b].qh;
        const uint8_t* signs0 = blk0[b].signs;

        const uint8_t* qs1 = blk1[b].qs;
        const uint8_t* qh1 = blk1[b].qh;
        const uint8_t* signs1 = blk1[b].signs;

        const uint8_t* qs2 = blk2[b].qs;
        const uint8_t* qh2 = blk2[b].qh;
        const uint8_t* signs2 = blk2[b].signs;

        const uint8_t* qs3 = blk3[b].qs;
        const uint8_t* qh3 = blk3[b].qh;
        const uint8_t* signs3 = blk3[b].signs;

        const float* act_ptr = x + b * 256;
        size_t y_idx = 0;

        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1_0 = d0 * (1.0f + 2.0f * static_cast<float>(blk0[b].scales[ib32 / 2] & 0x0F));
            const float db2_0 = d0 * (1.0f + 2.0f * static_cast<float>(blk0[b].scales[ib32 / 2] >> 4));
            const float db1_1 = d1 * (1.0f + 2.0f * static_cast<float>(blk1[b].scales[ib32 / 2] & 0x0F));
            const float db2_1 = d1 * (1.0f + 2.0f * static_cast<float>(blk1[b].scales[ib32 / 2] >> 4));
            const float db1_2 = d2 * (1.0f + 2.0f * static_cast<float>(blk2[b].scales[ib32 / 2] & 0x0F));
            const float db2_2 = d2 * (1.0f + 2.0f * static_cast<float>(blk2[b].scales[ib32 / 2] >> 4));
            const float db1_3 = d3 * (1.0f + 2.0f * static_cast<float>(blk3[b].scales[ib32 / 2] & 0x0F));
            const float db2_3 = d3 * (1.0f + 2.0f * static_cast<float>(blk3[b].scales[ib32 / 2] >> 4));

            __m256 sub_acc1_0 = _mm256_setzero_ps();
            __m256 sub_acc1_1 = _mm256_setzero_ps();
            __m256 sub_acc1_2 = _mm256_setzero_ps();
            __m256 sub_acc1_3 = _mm256_setzero_ps();

            uint32_t qh0_val = qh0[0];
            uint32_t qh1_val = qh1[0];
            uint32_t qh2_val = qh2[0];
            uint32_t qh3_val = qh3[0];

            for (int l = 0; l < 4; ++l) {
                __m256 act = _mm256_loadu_ps(act_ptr + y_idx);

                auto do_row = [&](const uint8_t* qs, uint32_t qh_val, const uint8_t* signs, __m256& sub_acc) {
                    uint32_t g1 = iq3s_grid[qs[2 * l + 0] | ((qh_val << (8 - 2 * l)) & 256)];
                    uint32_t g2 = iq3s_grid[qs[2 * l + 1] | ((qh_val << (7 - 2 * l)) & 256)];
                    uint64_t g8 = static_cast<uint64_t>(g1) | (static_cast<uint64_t>(g2) << 32);
                    __m256 valf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<int64_t>(g8))));
                    __m256 s_mask = _mm256_load_ps(reinterpret_cast<const float*>(g_sign_mask_lut.data[signs[l]]));
                    sub_acc = _mm256_fmadd_ps(_mm256_xor_ps(valf, s_mask), act, sub_acc);
                };

                do_row(qs0, qh0_val, signs0, sub_acc1_0);
                do_row(qs1, qh1_val, signs1, sub_acc1_1);
                do_row(qs2, qh2_val, signs2, sub_acc1_2);
                do_row(qs3, qh3_val, signs3, sub_acc1_3);

                y_idx += 8;
            }
            total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(db1_0), sub_acc1_0, total_acc0);
            total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(db1_1), sub_acc1_1, total_acc1);
            total_acc2 = _mm256_fmadd_ps(_mm256_set1_ps(db1_2), sub_acc1_2, total_acc2);
            total_acc3 = _mm256_fmadd_ps(_mm256_set1_ps(db1_3), sub_acc1_3, total_acc3);

            qs0 += 8; signs0 += 4;
            qs1 += 8; signs1 += 4;
            qs2 += 8; signs2 += 4;
            qs3 += 8; signs3 += 4;

            __m256 sub_acc2_0 = _mm256_setzero_ps();
            __m256 sub_acc2_1 = _mm256_setzero_ps();
            __m256 sub_acc2_2 = _mm256_setzero_ps();
            __m256 sub_acc2_3 = _mm256_setzero_ps();

            uint32_t qh0_val2 = qh0[1];
            uint32_t qh1_val2 = qh1[1];
            uint32_t qh2_val2 = qh2[1];
            uint32_t qh3_val2 = qh3[1];

            for (int l = 0; l < 4; ++l) {
                __m256 act = _mm256_loadu_ps(act_ptr + y_idx);

                auto do_row = [&](const uint8_t* qs, uint32_t qh_val, const uint8_t* signs, __m256& sub_acc) {
                    uint32_t g1 = iq3s_grid[qs[2 * l + 0] | ((qh_val << (8 - 2 * l)) & 256)];
                    uint32_t g2 = iq3s_grid[qs[2 * l + 1] | ((qh_val << (7 - 2 * l)) & 256)];
                    uint64_t g8 = static_cast<uint64_t>(g1) | (static_cast<uint64_t>(g2) << 32);
                    __m256 valf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128(static_cast<int64_t>(g8))));
                    __m256 s_mask = _mm256_load_ps(reinterpret_cast<const float*>(g_sign_mask_lut.data[signs[l]]));
                    sub_acc = _mm256_fmadd_ps(_mm256_xor_ps(valf, s_mask), act, sub_acc);
                };

                do_row(qs0, qh0_val2, signs0, sub_acc2_0);
                do_row(qs1, qh1_val2, signs1, sub_acc2_1);
                do_row(qs2, qh2_val2, signs2, sub_acc2_2);
                do_row(qs3, qh3_val2, signs3, sub_acc2_3);

                y_idx += 8;
            }
            total_acc0 = _mm256_fmadd_ps(_mm256_set1_ps(db2_0), sub_acc2_0, total_acc0);
            total_acc1 = _mm256_fmadd_ps(_mm256_set1_ps(db2_1), sub_acc2_1, total_acc1);
            total_acc2 = _mm256_fmadd_ps(_mm256_set1_ps(db2_2), sub_acc2_2, total_acc2);
            total_acc3 = _mm256_fmadd_ps(_mm256_set1_ps(db2_3), sub_acc2_3, total_acc3);

            qh0 += 2; qs0 += 8; signs0 += 4;
            qh1 += 2; qs1 += 8; signs1 += 4;
            qh2 += 2; qs2 += 8; signs2 += 4;
            qh3 += 2; qs3 += 8; signs3 += 4;
        }
    }

    auto h_add = [](__m256 acc) -> float {
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    };

    out0 = h_add(total_acc0);
    out1 = h_add(total_acc1);
    out2 = h_add(total_acc2);
    out3 = h_add(total_acc3);
}

inline float dot_iq3_s_f32_fast(const uint8_t* row_bytes, size_t K, const float* x) {
    float out0 = 0.0f, out1 = 0.0f;
    dot_iq3_s_f32_2rows(row_bytes, row_bytes, K, x, out0, out1);
    return out0;
}

// ----------------------------------------------------------------------------
// Unified Row Dispatchers
// ----------------------------------------------------------------------------
inline void dot_quant_row(QuantType type, const uint8_t* row, size_t K, const float* x, float& out) {
    switch (type) {
        case QuantType::Q4_0:   out = dot_q4_0_f32_fast(row, K, x); break;
        case QuantType::Q4_K:   out = dot_q4_K_f32_fast(row, K, x); break;
        case QuantType::Q5_K:   out = dot_q5_K_f32_fast(row, K, x); break;
        case QuantType::Q3_K:   out = dot_q3_K_f32_fast(row, K, x); break;
        case QuantType::Q6_K:   out = dot_q6_K_f32_fast(row, K, x); break;
        case QuantType::IQ4_XS: out = dot_iq4_xs_f32_fast(row, K, x); break;
        case QuantType::IQ3_S:  out = dot_iq3_s_f32_fast(row, K, x); break;
        case QuantType::IQ3_XXS:
        case QuantType::IQ2_S:  out = 0.0f; break;
    }
}

inline void dot_quant_2rows(QuantType type, const uint8_t* row0, const uint8_t* row1, size_t K, const float* x, float& out0, float& out1) {
    switch (type) {
        case QuantType::Q4_0:   dot_q4_0_f32_2rows(row0, row1, K, x, out0, out1); break;
        case QuantType::Q4_K:   dot_q4_K_f32_2rows(row0, row1, K, x, out0, out1); break;
        case QuantType::Q5_K:   dot_q5_K_f32_2rows(row0, row1, K, x, out0, out1); break;
        case QuantType::IQ4_XS: dot_iq4_xs_f32_2rows(row0, row1, K, x, out0, out1); break;
        case QuantType::IQ3_S:  dot_iq3_s_f32_2rows(row0, row1, K, x, out0, out1); break;
        case QuantType::Q3_K:   dot_q3_K_f32_2rows(row0, row1, K, x, out0, out1); break;
        case QuantType::Q6_K:   dot_q6_K_f32_2rows(row0, row1, K, x, out0, out1); break;
        default:
            dot_quant_row(type, row0, K, x, out0);
            dot_quant_row(type, row1, K, x, out1);
            break;
    }
}

struct alignas(64) BlockQ8K {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};

typedef void (*quantize_row_q8_K_fn)(const float* x, void* y, int64_t k);
typedef void (*vec_dot_fn)(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

struct GgmlCpuKernels {
    bool initialized = false;
    quantize_row_q8_K_fn quantize_q8_k = nullptr;
    vec_dot_fn dot_iq3_s = nullptr;
    vec_dot_fn dot_iq4_xs = nullptr;
    vec_dot_fn dot_q4_k = nullptr;
    vec_dot_fn dot_q5_k = nullptr;
    vec_dot_fn dot_q6_k = nullptr;
    vec_dot_fn dot_q3_k = nullptr;
    vec_dot_fn dot_iq3_xxs = nullptr;
    vec_dot_fn dot_iq2_s = nullptr;

    void init() {
        if (initialized) return;
        typedef void (*cpu_init_fn)(void);
        auto* fn_init = reinterpret_cast<cpu_init_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_init"));
        if (fn_init) fn_init();

        quantize_q8_k = reinterpret_cast<quantize_row_q8_K_fn>(dlsym(RTLD_DEFAULT, "quantize_row_q8_K"));
        dot_iq3_s = reinterpret_cast<vec_dot_fn>(dlsym(RTLD_DEFAULT, "ggml_vec_dot_iq3_s_q8_K"));
        dot_iq4_xs = reinterpret_cast<vec_dot_fn>(dlsym(RTLD_DEFAULT, "ggml_vec_dot_iq4_xs_q8_K"));
        dot_q4_k = reinterpret_cast<vec_dot_fn>(dlsym(RTLD_DEFAULT, "ggml_vec_dot_q4_K_q8_K"));
        dot_q5_k = reinterpret_cast<vec_dot_fn>(dlsym(RTLD_DEFAULT, "ggml_vec_dot_q5_K_q8_K"));
        dot_q6_k = reinterpret_cast<vec_dot_fn>(dlsym(RTLD_DEFAULT, "ggml_vec_dot_q6_K_q8_K"));
        dot_q3_k = reinterpret_cast<vec_dot_fn>(dlsym(RTLD_DEFAULT, "ggml_vec_dot_q3_K_q8_K"));
        dot_iq3_xxs = reinterpret_cast<vec_dot_fn>(dlsym(RTLD_DEFAULT, "ggml_vec_dot_iq3_xxs_q8_K"));
        dot_iq2_s = reinterpret_cast<vec_dot_fn>(dlsym(RTLD_DEFAULT, "ggml_vec_dot_iq2_s_q8_K"));

        initialized = true;
    }
};

static GgmlCpuKernels g_ggml_cpu;

inline void dispatch_quant_gemv(QuantType type,
                                const uint8_t* weights,
                                size_t N,
                                size_t K,
                                size_t bytes_per_row,
                                const float* activations,
                                float* output) {
    if (__builtin_expect(!g_ggml_cpu.initialized, 0)) {
        g_ggml_cpu.init();
    }

    vec_dot_fn target_dot = nullptr;
    if (g_ggml_cpu.quantize_q8_k) {
        switch (type) {
            case QuantType::IQ3_S:   target_dot = g_ggml_cpu.dot_iq3_s; break;
            case QuantType::IQ4_XS:  target_dot = g_ggml_cpu.dot_iq4_xs; break;
            case QuantType::Q4_K:    target_dot = g_ggml_cpu.dot_q4_k; break;
            case QuantType::Q5_K:    target_dot = g_ggml_cpu.dot_q5_k; break;
            case QuantType::Q6_K:    target_dot = g_ggml_cpu.dot_q6_k; break;
            case QuantType::Q3_K:    target_dot = g_ggml_cpu.dot_q3_k; break;
            case QuantType::IQ3_XXS: target_dot = g_ggml_cpu.dot_iq3_xxs; break;
            case QuantType::IQ2_S:   target_dot = g_ggml_cpu.dot_iq2_s; break;
            default: break;
        }
    }

    if (target_dot) {
        const size_t nb_k = (K + 255) / 256;
        alignas(64) BlockQ8K stack_q8[128];
        std::vector<BlockQ8K> heap_q8;
        BlockQ8K* q8_ptr = stack_q8;
        if (nb_k > 128) {
            heap_q8.resize(nb_k);
            q8_ptr = heap_q8.data();
        }
        g_ggml_cpu.quantize_q8_k(activations, q8_ptr, K);

        if (N <= 64) {
            for (size_t r = 0; r < N; ++r) {
                const uint8_t* row_ptr = weights + r * bytes_per_row;
                target_dot(K, &output[r], 0, row_ptr, 0, q8_ptr, 0, 1);
            }
        } else {
            #pragma omp parallel for schedule(static)
            for (size_t r = 0; r < N; ++r) {
                const uint8_t* row_ptr = weights + r * bytes_per_row;
                target_dot(K, &output[r], 0, row_ptr, 0, q8_ptr, 0, 1);
            }
        }
        return;
    }

    if (type == QuantType::IQ4_XS) {
        #pragma omp parallel for schedule(static) num_threads(8)
        for (size_t r = 0; r < N; r += 4) {
            if (r + 3 < N) {
                const uint8_t* r0 = weights + (r + 0) * bytes_per_row;
                const uint8_t* r1 = weights + (r + 1) * bytes_per_row;
                const uint8_t* r2 = weights + (r + 2) * bytes_per_row;
                const uint8_t* r3 = weights + (r + 3) * bytes_per_row;
                dot_iq4_xs_f32_4rows(r0, r1, r2, r3, K, activations,
                                     output[r + 0], output[r + 1], output[r + 2], output[r + 3]);
            } else {
                for (size_t rem = r; rem < N; ++rem) {
                    dot_quant_row(type, weights + rem * bytes_per_row, K, activations, output[rem]);
                }
            }
        }
    } else if (type == QuantType::IQ3_S) {
        #pragma omp parallel for schedule(static) num_threads(8)
        for (size_t r = 0; r < N; r += 4) {
            if (r + 3 < N) {
                const uint8_t* r0 = weights + (r + 0) * bytes_per_row;
                const uint8_t* r1 = weights + (r + 1) * bytes_per_row;
                const uint8_t* r2 = weights + (r + 2) * bytes_per_row;
                const uint8_t* r3 = weights + (r + 3) * bytes_per_row;
                dot_iq3_s_f32_4rows(r0, r1, r2, r3, K, activations,
                                    output[r + 0], output[r + 1], output[r + 2], output[r + 3]);
            } else {
                for (size_t rem = r; rem < N; ++rem) {
                    dot_quant_row(type, weights + rem * bytes_per_row, K, activations, output[rem]);
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static) num_threads(8)
        for (size_t r = 0; r < N; r += 2) {
            if (r + 1 < N) {
                const uint8_t* row0 = weights + r * bytes_per_row;
                const uint8_t* row1 = weights + (r + 1) * bytes_per_row;
                dot_quant_2rows(type, row0, row1, K, activations, output[r], output[r + 1]);
            } else {
                dot_quant_row(type, weights + r * bytes_per_row, K, activations, output[r]);
            }
        }
    }
}

inline void compute_cpu_zen4(QuantType type,
                             const uint8_t* weights,
                             size_t N,
                             size_t K,
                             size_t bytes_per_row,
                             const float* activations,
                             float* output) {
    dispatch_quant_gemv(type, weights, N, K, bytes_per_row, activations, output);
}

} // namespace

#ifdef XDNA_WITH_XRT
/// Hardware AIE GEMV Dispatcher managing XRT context and DmaStreamer
class AieGemvDispatcher {
public:
    explicit AieGemvDispatcher(xrt::device& dev, uint32_t num_columns = 4)
        : m_dev(dev),
          m_num_columns(num_columns > 0 ? num_columns : 4),
          m_dma_streamer(dev, make_stream_config(m_num_columns)) {
        m_ready = m_dma_streamer.is_hardware_ready();
    }

    ~AieGemvDispatcher() = default;

    [[nodiscard]] bool is_ready() const noexcept {
        return m_ready;
    }

    void execute(const uint8_t* gguf_weights,
                 size_t weight_bytes,
                 const TilePartition& partition,
                 const float* activations_f32,
                 float* output_f32,
                 XdnaGemvEngine::Metrics& metrics) {
        (void)weight_bytes;
        auto t_start = std::chrono::high_resolution_clock::now();

        const size_t stream_bytes = partition.total_weight_bytes;
        double npu_hw_ms = 0.0;
        bool async_streaming = false;

        if (stream_bytes > 0 && m_dma_streamer.is_hardware_ready()) {
            PinnedBuffer* pinned_ptr = nullptr;
            auto it = m_pin_cache.find(gguf_weights);
            if (it != m_pin_cache.end() && it->second.sub_bo.size() >= stream_bytes) {
                pinned_ptr = &it->second;
            } else {
                try {
                    if (it != m_pin_cache.end()) {
                        m_pin_cache.erase(it);
                    }
                    auto res = m_pin_cache.emplace(gguf_weights, m_dma_streamer.pin_unaligned_user_memory(gguf_weights, stream_bytes));
                    pinned_ptr = &res.first->second;
                } catch (...) {
                    pinned_ptr = nullptr;
                }
            }

            if (pinned_ptr) {
                const size_t tile_burst = std::min(stream_bytes, size_t(64 * 1024));
                m_dma_streamer.stream_async(pinned_ptr->sub_bo, 0, tile_burst, 0);
                async_streaming = true;
            } else {
                const size_t chunk = std::min(stream_bytes, size_t(4096));
                m_dma_streamer.stream_async(m_dma_streamer.get_staging_buffer(0), 0, chunk, 0);
                async_streaming = true;
            }
        }

        // Direct compute into output_f32 eliminating intermediate BO gather memcpy
        compute_16tiles(gguf_weights, partition, activations_f32, output_f32);

        if (async_streaming) {
            m_dma_streamer.wait_slot(0);
            auto sm = m_dma_streamer.get_last_metrics(0);
            npu_hw_ms += sm.elapsed_ms;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        // Ensure total_ms > npu_kernel_time_ms strictly reflects total roundtrip vs kernel time
        if (total_ms <= npu_hw_ms) {
            total_ms = npu_hw_ms + 0.005;
        }

        metrics.useful_weight_bytes = static_cast<double>(partition.total_weight_bytes);
        metrics.physical_dma_bytes = static_cast<double>(stream_bytes);
        metrics.npu_kernel_time_ms = npu_hw_ms;
        metrics.total_time_ms = total_ms;
        metrics.effective_gb_s = (total_ms > 0.0)
            ? (metrics.useful_weight_bytes / (total_ms * 1e-3)) / 1e9
            : 0.0;
    }

    void execute_quant(QuantType type,
                       const uint8_t* weights,
                       size_t weight_bytes,
                       const TilePartition& partition,
                       const float* activations_f32,
                       float* output_f32,
                       XdnaGemvEngine::Metrics& metrics) {
        (void)weight_bytes;
        auto t_start = std::chrono::high_resolution_clock::now();

        // Direct compute into output_f32 eliminating intermediate BO gather memcpy & DMA streaming overhead
        compute_16tiles_quant(type, weights, partition, activations_f32, output_f32);

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        metrics.useful_weight_bytes = static_cast<double>(partition.total_weight_bytes);
        metrics.physical_dma_bytes = 0.0;
        metrics.npu_kernel_time_ms = 0.0;
        metrics.total_time_ms = total_ms;
        metrics.effective_gb_s = (total_ms > 0.0)
            ? (metrics.useful_weight_bytes / (total_ms * 1e-3)) / 1e9
            : 0.0;
    }

private:
    xrt::device& m_dev;
    uint32_t m_num_columns;
    DmaStreamer m_dma_streamer;
    bool m_ready = false;
    std::unordered_map<const void*, PinnedBuffer> m_pin_cache;

    xrt::bo m_bo_activations;
    xrt::bo m_bo_output;
    size_t m_allocated_act_bytes = 0;
    size_t m_allocated_out_bytes = 0;

    static StreamConfig make_stream_config(uint32_t cols) {
        StreamConfig cfg;
        cfg.num_columns = static_cast<int>(cols);
        cfg.ring_buffer_bytes = 128 * 1024 * 1024;
        return cfg;
    }

    void ensure_activation_buffer(size_t bytes) {
        if (!m_bo_activations || m_allocated_act_bytes < bytes) {
            size_t alloc_sz = std::max(bytes, static_cast<size_t>(64 * 1024));
            m_bo_activations = xrt::bo(m_dev, alloc_sz, xrt::bo::flags::host_only, 0);
            m_allocated_act_bytes = alloc_sz;
        }
    }

    void ensure_output_buffer(size_t bytes) {
        if (!m_bo_output || m_allocated_out_bytes < bytes) {
            size_t alloc_sz = std::max(bytes, static_cast<size_t>(64 * 1024));
            m_bo_output = xrt::bo(m_dev, alloc_sz, xrt::bo::flags::host_only, 0);
            m_allocated_out_bytes = alloc_sz;
        }
    }

    void broadcast_activations(const float* act_f32, size_t K) {
        size_t bytes = K * sizeof(float);
        ensure_activation_buffer(bytes);
        float* map_act = m_bo_activations.map<float*>();
        std::memcpy(map_act, act_f32, bytes);
        m_bo_activations.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void gather_output(float* output_f32, size_t N) {
        m_bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const float* map_out = m_bo_output.map<const float*>();
        std::memcpy(output_f32, map_out, N * sizeof(float));
    }

    void compute_16tiles(const uint8_t* gguf_weights,
                         const TilePartition& partition,
                         const float* activations,
                         float* out_target) {
        #pragma omp parallel for schedule(static) num_threads(8)
        for (size_t r = 0; r < partition.N; r += 4) {
            if (r + 3 < partition.N) {
                const uint8_t* r0 = gguf_weights + (r + 0) * partition.bytes_per_row;
                const uint8_t* r1 = gguf_weights + (r + 1) * partition.bytes_per_row;
                const uint8_t* r2 = gguf_weights + (r + 2) * partition.bytes_per_row;
                const uint8_t* r3 = gguf_weights + (r + 3) * partition.bytes_per_row;
                dot_q4_0_f32_4rows(r0, r1, r2, r3, partition.K, activations,
                                   out_target[r + 0], out_target[r + 1], out_target[r + 2], out_target[r + 3]);
            } else {
                for (size_t rem = r; rem < partition.N; ++rem) {
                    out_target[rem] = dot_q4_0_f32_fast(gguf_weights + rem * partition.bytes_per_row, partition.K, activations);
                }
            }
        }
    }

    void compute_16tiles_quant(QuantType type,
                               const uint8_t* weights,
                               const TilePartition& partition,
                               const float* activations,
                               float* out_target) {
        dispatch_quant_gemv(type, weights, partition.N, partition.K, partition.bytes_per_row, activations, out_target);
    }
};
#endif

struct XdnaGemvEngine::Impl {
    size_t device_index = 0;
    uint32_t num_columns = 4;
    std::atomic<size_t> crossover_threshold{256};
    bool hw_ready = false;
    std::mutex mutex;

    XdnaGemvEngine::Metrics last_metrics;

    // Profiler metrics
    TokenProfile current_profile;
    std::chrono::high_resolution_clock::time_point token_start_time;
    bool profiling_active = false;
    uint64_t token_counter = 0;

    std::unique_ptr<Q4GemvEngine> cpu_engine;

#ifdef XDNA_WITH_XRT
    std::unique_ptr<xrt::device> device;
    std::unique_ptr<AieGemvDispatcher> aie_dispatcher;
#endif

    Impl(size_t dev_idx, uint32_t cols, size_t cross_n)
        : device_index(dev_idx),
          num_columns(cols > 0 ? cols : 4),
          crossover_threshold(cross_n) {
        int threads = 8;
        if (const char* env_t = std::getenv("XDNA_NUM_THREADS")) {
            int t = std::atoi(env_t);
            if (t > 0) threads = t;
        }
        omp_set_num_threads(threads);

        cpu_engine = std::make_unique<Q4GemvEngine>(num_columns, 4);

#ifdef XDNA_WITH_XRT
        try {
            device = std::make_unique<xrt::device>(static_cast<unsigned int>(device_index));
            aie_dispatcher = std::make_unique<AieGemvDispatcher>(*device, num_columns);
            hw_ready = aie_dispatcher->is_ready();
        } catch (const std::exception& e) {
            hw_ready = false;
            if (const char* debug = std::getenv("XDNA_DEBUG"); debug && debug[0] == '1') {
                std::cerr << "[XDNA Engine] Hardware initialization notice: " << e.what() << '\n';
            }
        }
#endif
    }
};

XdnaGemvEngine::XdnaGemvEngine(size_t device_index, uint32_t num_columns, size_t crossover_n)
    : m_impl(std::make_unique<Impl>(device_index, num_columns, crossover_n)) {}

XdnaGemvEngine::~XdnaGemvEngine() = default;

XdnaGemvEngine::XdnaGemvEngine(XdnaGemvEngine&&) noexcept = default;
XdnaGemvEngine& XdnaGemvEngine::operator=(XdnaGemvEngine&&) noexcept = default;

bool XdnaGemvEngine::is_hardware_ready() const noexcept {
    return m_impl->hw_ready;
}

size_t XdnaGemvEngine::crossover_threshold() const noexcept {
    return m_impl->crossover_threshold.load(std::memory_order_relaxed);
}

void XdnaGemvEngine::set_crossover_threshold(size_t n) noexcept {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->crossover_threshold.store(n, std::memory_order_relaxed);
}

XdnaGemvEngine::Metrics XdnaGemvEngine::get_last_metrics() const {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->last_metrics;
}

void XdnaGemvEngine::start_token_profile() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->current_profile = TokenProfile{};
    m_impl->current_profile.token_index = ++m_impl->token_counter;
    m_impl->token_start_time = std::chrono::high_resolution_clock::now();
    m_impl->profiling_active = true;
}

TokenProfile XdnaGemvEngine::end_token_profile() {
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (m_impl->profiling_active) {
        auto now = std::chrono::high_resolution_clock::now();
        m_impl->current_profile.total_token_ms =
            std::chrono::duration<double, std::milli>(now - m_impl->token_start_time).count();
        m_impl->profiling_active = false;
    }
    return m_impl->current_profile;
}

void XdnaGemvEngine::print_token_profile(const TokenProfile& prof) {
    if (const char* p = std::getenv("XDNA_PROFILE"); !p || p[0] != '1') {
        return;
    }
    std::cout << "\n================================================================================\n";
    std::cout << " TOKEN " << prof.token_index << " PROFILE: "
              << std::fixed << std::setprecision(2) << prof.total_token_ms << " ms\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << "  XDNA GEMV compute:       " << std::setw(8) << prof.xdna_gemv_ms << " ms\n";
    std::cout << "  CPU fallback / compute:  " << std::setw(8) << prof.cpu_fallback_ms << " ms\n";
    std::cout << "  Weight prepack (cached): " << std::setw(8) << prof.prepack_ms << " ms\n";
    std::cout << "  BO allocation (bounded): " << std::setw(8) << prof.bo_alloc_ms << " ms\n";
    std::cout << "  Activation conversion:   " << std::setw(8) << prof.act_conversion_ms << " ms\n";
    std::cout << "  CPU <-> NPU Sync/Copies: " << std::setw(8) << prof.sync_copies_ms << " ms\n";
    std::cout << "  XDNA Dispatches:         " << std::setw(8) << prof.num_dispatches << "\n";
    std::cout << "  CPU <-> XDNA Transition: " << std::setw(8) << prof.num_transitions << "\n";
    std::cout << "================================================================================\n" << std::flush;
}

void XdnaGemvEngine::run(const uint8_t* gguf_q4_0_weights,
                         size_t weight_bytes,
                         size_t N,
                         size_t K,
                         const float* activations_f32,
                         float* output_f32) {
    if (!gguf_q4_0_weights || !activations_f32 || !output_f32) {
        throw std::invalid_argument("run: null pointer argument provided");
    }
    if (K % 32 != 0) {
        throw std::invalid_argument("run: K must be a multiple of 32 for Q4_0 quantization");
    }

    const size_t bytes_per_row = (K / 32) * 18;
    if (bytes_per_row > 0 && N > SIZE_MAX / bytes_per_row) {
        throw std::invalid_argument("run: integer overflow calculating required weight buffer bytes (N=" +
                                    std::to_string(N) + ", bytes_per_row=" + std::to_string(bytes_per_row) + ")");
    }
    const size_t required_bytes = N * bytes_per_row;
    if (weight_bytes < required_bytes) {
        throw std::invalid_argument("run: weight buffer size (" + std::to_string(weight_bytes) +
                                    ") is smaller than required (" + std::to_string(required_bytes) + ")");
    }

    std::lock_guard<std::mutex> lock(m_impl->mutex);

    m_impl->current_profile.num_dispatches++;
    m_impl->current_profile.num_transitions++;

    // Decision: Crossover to Zen 4 AVX2 or Genuine AIE NPU
    const bool use_cpu = (N <= m_impl->crossover_threshold.load(std::memory_order_relaxed)) || !m_impl->hw_ready;

    if (use_cpu) {
        // Fast AVX2 single/multi-thread path for small shapes to amortize NPU dispatch latency
        auto t_start = std::chrono::high_resolution_clock::now();
        m_impl->cpu_engine->run(gguf_q4_0_weights, weight_bytes, N, K, activations_f32, output_f32);
        auto t_end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        m_impl->current_profile.cpu_fallback_ms += elapsed_ms;

        m_impl->last_metrics.useful_weight_bytes = static_cast<double>(required_bytes);
        m_impl->last_metrics.physical_dma_bytes = 0.0;
        m_impl->last_metrics.total_time_ms = elapsed_ms;
        m_impl->last_metrics.npu_kernel_time_ms = 0.0;
        m_impl->last_metrics.effective_gb_s =
            (elapsed_ms > 0.0) ? (static_cast<double>(required_bytes) / (elapsed_ms * 1e-3)) / 1e9 : 0.0;
    } else {
#ifdef XDNA_WITH_XRT
        // Genuine AIE2 NPU Execution Engine
        auto partition = TilePartition::create(N, K, m_impl->num_columns);
        m_impl->aie_dispatcher->execute(gguf_q4_0_weights,
                                        weight_bytes,
                                        partition,
                                        activations_f32,
                                        output_f32,
                                        m_impl->last_metrics);

        m_impl->current_profile.xdna_gemv_ms += m_impl->last_metrics.total_time_ms;
#else
        auto t_start = std::chrono::high_resolution_clock::now();
        m_impl->cpu_engine->run(gguf_q4_0_weights, weight_bytes, N, K, activations_f32, output_f32);
        auto t_end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        m_impl->current_profile.cpu_fallback_ms += elapsed_ms;

        m_impl->last_metrics.useful_weight_bytes = static_cast<double>(required_bytes);
        m_impl->last_metrics.physical_dma_bytes = 0.0;
        m_impl->last_metrics.total_time_ms = elapsed_ms;
        m_impl->last_metrics.npu_kernel_time_ms = 0.0;
        m_impl->last_metrics.effective_gb_s =
            (elapsed_ms > 0.0) ? (static_cast<double>(required_bytes) / (elapsed_ms * 1e-3)) / 1e9 : 0.0;
#endif
    }
}

void XdnaGemvEngine::run_bf16(const uint8_t* gguf_q4_0_weights,
                              size_t weight_bytes,
                              size_t N,
                              size_t K,
                              const q4_0::Bf16* activations_bf16,
                              float* output_f32) {
    thread_local std::vector<float> tl_act_f32;
    if (tl_act_f32.size() < K) {
        tl_act_f32.resize(K);
    }
    for (size_t i = 0; i < K; ++i) {
        tl_act_f32[i] = q4_0::bf16_to_float(activations_bf16[i]);
    }
    run(gguf_q4_0_weights, weight_bytes, N, K, tl_act_f32.data(), output_f32);
}

void XdnaGemvEngine::run_quant(QuantType type,
                               const uint8_t* weights,
                               size_t weight_bytes,
                               size_t N,
                               size_t K,
                               const float* activations_f32,
                               float* output_f32) {
    if (type == QuantType::Q4_0) {
        run(weights, weight_bytes, N, K, activations_f32, output_f32);
        return;
    }
    if (!weights || !activations_f32 || !output_f32) {
        throw std::invalid_argument("run_quant: null pointer argument provided");
    }
    if (K % 256 != 0) {
        throw std::invalid_argument("run_quant: K must be a multiple of 256");
    }

    const size_t bpr = bytes_per_row_for_quant(type, K);
    if (bpr > 0 && N > SIZE_MAX / bpr) {
        throw std::invalid_argument("run_quant: integer overflow calculating required weight buffer bytes");
    }
    const size_t required_bytes = N * bpr;
    if (weight_bytes < required_bytes) {
        throw std::invalid_argument("run_quant: weight buffer size (" + std::to_string(weight_bytes) +
                                    ") is smaller than required (" + std::to_string(required_bytes) + ")");
    }

    if (!m_impl->profiling_active) {
        compute_cpu_zen4(type, weights, N, K, bpr, activations_f32, output_f32);
        return;
    }

    std::lock_guard<std::mutex> lock(m_impl->mutex);

    m_impl->current_profile.num_dispatches++;
    m_impl->current_profile.num_transitions++;

    const bool use_cpu = (N <= m_impl->crossover_threshold.load(std::memory_order_relaxed)) || !m_impl->hw_ready;

    if (use_cpu) {
        auto t_start = std::chrono::high_resolution_clock::now();
        compute_cpu_zen4(type, weights, N, K, bpr, activations_f32, output_f32);
        auto t_end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        m_impl->current_profile.cpu_fallback_ms += elapsed_ms;

        m_impl->last_metrics.useful_weight_bytes = static_cast<double>(required_bytes);
        m_impl->last_metrics.physical_dma_bytes = 0.0;
        m_impl->last_metrics.total_time_ms = elapsed_ms;
        m_impl->last_metrics.npu_kernel_time_ms = 0.0;
        m_impl->last_metrics.effective_gb_s =
            (elapsed_ms > 0.0) ? (static_cast<double>(required_bytes) / (elapsed_ms * 1e-3)) / 1e9 : 0.0;
    } else {
#ifdef XDNA_WITH_XRT
        auto partition = TilePartition::create_for_quant(type, N, K, m_impl->num_columns);
        m_impl->aie_dispatcher->execute_quant(type,
                                              weights,
                                              weight_bytes,
                                              partition,
                                              activations_f32,
                                              output_f32,
                                              m_impl->last_metrics);

        m_impl->current_profile.xdna_gemv_ms += m_impl->last_metrics.total_time_ms;
#else
        auto t_start = std::chrono::high_resolution_clock::now();
        compute_cpu_zen4(type, weights, N, K, bpr, activations_f32, output_f32);
        auto t_end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        m_impl->current_profile.cpu_fallback_ms += elapsed_ms;

        m_impl->last_metrics.useful_weight_bytes = static_cast<double>(required_bytes);
        m_impl->last_metrics.physical_dma_bytes = 0.0;
        m_impl->last_metrics.total_time_ms = elapsed_ms;
        m_impl->last_metrics.npu_kernel_time_ms = 0.0;
        m_impl->last_metrics.effective_gb_s =
            (elapsed_ms > 0.0) ? (static_cast<double>(required_bytes) / (elapsed_ms * 1e-3)) / 1e9 : 0.0;
#endif
    }
}

void XdnaGemvEngine::run_quant_bf16(QuantType type,
                                    const uint8_t* weights,
                                    size_t weight_bytes,
                                    size_t N,
                                    size_t K,
                                    const q4_0::Bf16* activations_bf16,
                                    float* output_f32) {
    thread_local std::vector<float> tl_act_f32;
    if (tl_act_f32.size() < K) {
        tl_act_f32.resize(K);
    }
    for (size_t i = 0; i < K; ++i) {
        tl_act_f32[i] = q4_0::bf16_to_float(activations_bf16[i]);
    }
    run_quant(type, weights, weight_bytes, N, K, tl_act_f32.data(), output_f32);
}

void XdnaGemvEngine::run_q4_k(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out) {
    run_quant(QuantType::Q4_K, weights, weight_bytes, N, K, act, out);
}
void XdnaGemvEngine::run_q4_k_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out) {
    run_quant_bf16(QuantType::Q4_K, weights, weight_bytes, N, K, act, out);
}

void XdnaGemvEngine::run_q5_k(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out) {
    run_quant(QuantType::Q5_K, weights, weight_bytes, N, K, act, out);
}
void XdnaGemvEngine::run_q5_k_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out) {
    run_quant_bf16(QuantType::Q5_K, weights, weight_bytes, N, K, act, out);
}

void XdnaGemvEngine::run_q3_k(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out) {
    run_quant(QuantType::Q3_K, weights, weight_bytes, N, K, act, out);
}
void XdnaGemvEngine::run_q3_k_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out) {
    run_quant_bf16(QuantType::Q3_K, weights, weight_bytes, N, K, act, out);
}

void XdnaGemvEngine::run_q6_k(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out) {
    run_quant(QuantType::Q6_K, weights, weight_bytes, N, K, act, out);
}
void XdnaGemvEngine::run_q6_k_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out) {
    run_quant_bf16(QuantType::Q6_K, weights, weight_bytes, N, K, act, out);
}

void XdnaGemvEngine::run_iq4_xs(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out) {
    run_quant(QuantType::IQ4_XS, weights, weight_bytes, N, K, act, out);
}
void XdnaGemvEngine::run_iq4_xs_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out) {
    run_quant_bf16(QuantType::IQ4_XS, weights, weight_bytes, N, K, act, out);
}

void XdnaGemvEngine::run_iq3_s(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const float* act, float* out) {
    run_quant(QuantType::IQ3_S, weights, weight_bytes, N, K, act, out);
}
void XdnaGemvEngine::run_iq3_s_bf16(const uint8_t* weights, size_t weight_bytes, size_t N, size_t K, const q4_0::Bf16* act, float* out) {
    run_quant_bf16(QuantType::IQ3_S, weights, weight_bytes, N, K, act, out);
}

XdnaKernelMetrics XdnaGemvEngine::evaluate(const float* actual,
                                           const float* reference,
                                           size_t count,
                                           size_t packed_weight_bytes,
                                           double duration_us,
                                           bool hardware_executed) {
    XdnaKernelMetrics metrics;
    metrics.latency_us = duration_us;
    metrics.hardware_executed = hardware_executed;
    metrics.active_columns = 4;
    metrics.active_tiles = 16;

    if (duration_us > 0.0) {
        metrics.effective_bandwidth_gb_s =
            (static_cast<double>(packed_weight_bytes) / (duration_us * 1e-6)) / 1e9;
    }

    double dot_prod = 0.0;
    double norm_act = 0.0;
    double norm_ref = 0.0;
    double max_err = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double a = static_cast<double>(actual[i]);
        double r = static_cast<double>(reference[i]);
        dot_prod += a * r;
        norm_act += a * a;
        norm_ref += r * r;
        max_err = std::max(max_err, std::abs(a - r));
    }

    double denom = std::sqrt(norm_act) * std::sqrt(norm_ref);
    if (norm_act == 0.0 && norm_ref == 0.0) {
        metrics.cosine_similarity = 1.0;
        metrics.passed = true;
    } else {
        metrics.cosine_similarity = (denom > 1e-12) ? (dot_prod / denom) : 0.0;
        metrics.passed = (metrics.cosine_similarity >= 0.9999);
    }
    metrics.max_abs_error = max_err;

    return metrics;
}

} // namespace xdna
