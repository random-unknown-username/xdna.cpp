#pragma once

#include "xdna/q4_0_reference.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace xdna {

// ============================================================================
// Super-block quantization parameters matching upstream GGML
// ============================================================================
inline constexpr size_t QK_K = 256;
inline constexpr size_t K_SCALE_SIZE = 12;

// ============================================================================
// Quantization Format Enum
// ============================================================================
enum class QuantType {
    Q4_0,
    Q4_K,
    Q5_K,
    Q3_K,
    Q6_K,
    IQ4_XS,
    IQ3_S,
    IQ3_XXS,
    IQ2_S
};

inline const char* quant_type_name(QuantType type) {
    switch (type) {
        case QuantType::Q4_0:    return "Q4_0";
        case QuantType::Q4_K:    return "Q4_K";
        case QuantType::Q5_K:    return "Q5_K";
        case QuantType::Q3_K:    return "Q3_K";
        case QuantType::Q6_K:    return "Q6_K";
        case QuantType::IQ4_XS:  return "IQ4_XS";
        case QuantType::IQ3_S:   return "IQ3_S";
        case QuantType::IQ3_XXS: return "IQ3_XXS";
        case QuantType::IQ2_S:   return "IQ2_S";
    }
    return "UNKNOWN";
}

// ============================================================================
// 1. Q4_K Block Structure (QK_K=256, 144 bytes per block)
// ============================================================================
struct block_q4_K {
    uint16_t d;                 // fp16 super-block scale for quantized scales
    uint16_t dmin;              // fp16 super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // 12 bytes: scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];       // 128 bytes: 4-bit quants
};

static_assert(std::is_standard_layout_v<block_q4_K>, "block_q4_K must remain standard-layout");
static_assert(std::is_trivially_copyable_v<block_q4_K>, "block_q4_K must remain trivially copyable");
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be exactly 144 bytes");
static_assert(offsetof(block_q4_K, d) == 0, "block_q4_K::d offset must be 0");
static_assert(offsetof(block_q4_K, dmin) == 2, "block_q4_K::dmin offset must be 2");
static_assert(offsetof(block_q4_K, scales) == 4, "block_q4_K::scales offset must be 4");
static_assert(offsetof(block_q4_K, qs) == 16, "block_q4_K::qs offset must be 16");

// ============================================================================
// 2. Q5_K Block Structure (QK_K=256, 176 bytes per block)
// ============================================================================
struct block_q5_K {
    uint16_t d;                 // fp16 super-block scale for quantized scales
    uint16_t dmin;              // fp16 super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // 12 bytes: scales and mins, quantized with 6 bits
    uint8_t qh[QK_K / 8];       // 32 bytes: quants, high bit
    uint8_t qs[QK_K / 2];       // 128 bytes: quants, low 4 bits
};

static_assert(std::is_standard_layout_v<block_q5_K>, "block_q5_K must remain standard-layout");
static_assert(std::is_trivially_copyable_v<block_q5_K>, "block_q5_K must remain trivially copyable");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be exactly 176 bytes");
static_assert(offsetof(block_q5_K, d) == 0, "block_q5_K::d offset must be 0");
static_assert(offsetof(block_q5_K, dmin) == 2, "block_q5_K::dmin offset must be 2");
static_assert(offsetof(block_q5_K, scales) == 4, "block_q5_K::scales offset must be 4");
static_assert(offsetof(block_q5_K, qh) == 16, "block_q5_K::qh offset must be 16");
static_assert(offsetof(block_q5_K, qs) == 48, "block_q5_K::qs offset must be 48");

// ============================================================================
// 3. Q3_K Block Structure (QK_K=256, 110 bytes per block)
// ============================================================================
struct block_q3_K {
    uint8_t hmask[QK_K / 8];     // 32 bytes: quants high bit
    uint8_t qs[QK_K / 4];        // 64 bytes: quants low 2 bits
    uint8_t scales[12];          // 12 bytes: scales, quantized with 6 bits
    uint16_t d;                  // fp16 super-block scale
};

static_assert(std::is_standard_layout_v<block_q3_K>, "block_q3_K must remain standard-layout");
static_assert(std::is_trivially_copyable_v<block_q3_K>, "block_q3_K must remain trivially copyable");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be exactly 110 bytes");
static_assert(offsetof(block_q3_K, hmask) == 0, "block_q3_K::hmask offset must be 0");
static_assert(offsetof(block_q3_K, qs) == 32, "block_q3_K::qs offset must be 32");
static_assert(offsetof(block_q3_K, scales) == 96, "block_q3_K::scales offset must be 96");
static_assert(offsetof(block_q3_K, d) == 108, "block_q3_K::d offset must be 108");

// ============================================================================
// 4. Q6_K Block Structure (QK_K=256, 210 bytes per block)
// ============================================================================
struct block_q6_K {
    uint8_t ql[QK_K / 2];        // 128 bytes: quants, lower 4 bits
    uint8_t qh[QK_K / 4];        // 64 bytes: quants, upper 2 bits
    int8_t  scales[QK_K / 16];   // 16 bytes: scales, quantized with 8 bits
    uint16_t d;                  // fp16 super-block scale
};

static_assert(std::is_standard_layout_v<block_q6_K>, "block_q6_K must remain standard-layout");
static_assert(std::is_trivially_copyable_v<block_q6_K>, "block_q6_K must remain trivially copyable");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be exactly 210 bytes");
static_assert(offsetof(block_q6_K, ql) == 0, "block_q6_K::ql offset must be 0");
static_assert(offsetof(block_q6_K, qh) == 128, "block_q6_K::qh offset must be 128");
static_assert(offsetof(block_q6_K, scales) == 192, "block_q6_K::scales offset must be 192");
static_assert(offsetof(block_q6_K, d) == 208, "block_q6_K::d offset must be 208");

// ============================================================================
// 5. IQ4_XS Block Structure (QK_K=256, 136 bytes per block)
// ============================================================================
struct block_iq4_xs {
    uint16_t d;                  // fp16 scale (2 bytes)
    uint16_t scales_h;           // scales high bits (2 bytes)
    uint8_t  scales_l[QK_K / 64]; // scales low bits (4 bytes)
    uint8_t  qs[QK_K / 2];        // quants (128 bytes)
};

static_assert(std::is_standard_layout_v<block_iq4_xs>, "block_iq4_xs must remain standard-layout");
static_assert(std::is_trivially_copyable_v<block_iq4_xs>, "block_iq4_xs must remain trivially copyable");
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs must be exactly 136 bytes");
static_assert(offsetof(block_iq4_xs, d) == 0, "block_iq4_xs::d offset must be 0");
static_assert(offsetof(block_iq4_xs, scales_h) == 2, "block_iq4_xs::scales_h offset must be 2");
static_assert(offsetof(block_iq4_xs, scales_l) == 4, "block_iq4_xs::scales_l offset must be 4");
static_assert(offsetof(block_iq4_xs, qs) == 8, "block_iq4_xs::qs offset must be 8");

// ============================================================================
// 6. IQ3_S Block Structure (QK_K=256, 110 bytes per block)
// ============================================================================
struct block_iq3_s {
    uint16_t d;                  // fp16 scale (2 bytes)
    uint8_t  qs[QK_K / 4];       // quants (64 bytes)
    uint8_t  qh[QK_K / 32];      // quants high bit (8 bytes)
    uint8_t  signs[QK_K / 8];    // signs (32 bytes)
    uint8_t  scales[QK_K / 64];  // scales (4 bytes)
};

static_assert(std::is_standard_layout_v<block_iq3_s>, "block_iq3_s must remain standard-layout");
static_assert(std::is_trivially_copyable_v<block_iq3_s>, "block_iq3_s must remain trivially copyable");
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s must be exactly 110 bytes");
static_assert(offsetof(block_iq3_s, d) == 0, "block_iq3_s::d offset must be 0");
static_assert(offsetof(block_iq3_s, qs) == 2, "block_iq3_s::qs offset must be 2");
static_assert(offsetof(block_iq3_s, qh) == 66, "block_iq3_s::qh offset must be 66");
static_assert(offsetof(block_iq3_s, signs) == 74, "block_iq3_s::signs offset must be 74");
static_assert(offsetof(block_iq3_s, scales) == 106, "block_iq3_s::scales offset must be 106");

// ============================================================================
// Helper functions for block geometry
// ============================================================================
inline size_t block_size_for_quant(QuantType type) {
    switch (type) {
        case QuantType::Q4_0:   return 32;
        case QuantType::Q4_K:
        case QuantType::Q5_K:
        case QuantType::Q3_K:
        case QuantType::Q6_K:
        case QuantType::IQ4_XS:
        case QuantType::IQ3_S:
        case QuantType::IQ3_XXS:
        case QuantType::IQ2_S:  return 256;
    }
    return 32;
}

inline size_t bytes_per_block_for_quant(QuantType type) {
    switch (type) {
        case QuantType::Q4_0:    return 18;
        case QuantType::Q4_K:    return 144;
        case QuantType::Q5_K:    return 176;
        case QuantType::Q3_K:    return 110;
        case QuantType::Q6_K:    return 210;
        case QuantType::IQ4_XS:  return 136;
        case QuantType::IQ3_S:   return 110;
        case QuantType::IQ3_XXS: return 98;
        case QuantType::IQ2_S:   return 82;
    }
    return 18;
}

inline size_t bytes_per_row_for_quant(QuantType type, size_t K) {
    size_t bs = block_size_for_quant(type);
    size_t bpb = bytes_per_block_for_quant(type);
    return (K / bs) * bpb;
}

// ============================================================================
// Scale / Min unpacking helpers (Exact GGML layout)
// ============================================================================
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d_sc, uint8_t& m_sc) {
    if (j < 4) {
        d_sc = q[j] & 63;
        m_sc = q[j + 4] & 63;
    } else {
        d_sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m_sc = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}

// IQ4 Non-linear table
alignas(16) static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// IQ3_S grid table (512 entries)
alignas(16) static const uint32_t iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
};

// ============================================================================
// GGML Reference Scalar Dequantization Routines
// ============================================================================
inline void dequantize_row_q4_K_ref(const block_q4_K* x, float* y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = static_cast<int>(k / QK_K);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* q = x[i].qs;
        const float d   = q4_0::fp16_to_float(x[i].d);
        const float min = q4_0::fp16_to_float(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, sc, m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, sc, m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4)   - m2;
            q += 32;
            is += 2;
        }
    }
}

inline void dequantize_row_q5_K_ref(const block_q5_K* x, float* y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = static_cast<int>(k / QK_K);

    for (int i = 0; i < nb; ++i) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d   = q4_0::fp16_to_float(x[i].d);
        const float min = q4_0::fp16_to_float(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, sc, m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, sc, m);
            const float d2 = d * sc;
            const float m2 = min * m;

            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * ((ql[l] >> 4)   + ((qh[l] & u2) ? 16 : 0)) - m2;
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

inline void dequantize_row_q3_K_ref(const block_q3_K* x, float* y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = static_cast<int>(k / QK_K);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0F0F0F0F;

    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    for (int i = 0; i < nb; ++i) {
        const float d_all = q4_0::fp16_to_float(x[i].d);
        const uint8_t* q = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;

        std::memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (static_cast<float>(scales[is++]) - 32.0f);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * (static_cast<float>((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0.0f : 4.0f));
                }
                dl = d_all * (static_cast<float>(scales[is++]) - 32.0f);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * (static_cast<float>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0.0f : 4.0f));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

inline void dequantize_row_q6_K_ref(const block_q6_K* x, float* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {
        const float d = q4_0::fp16_to_float(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = static_cast<int8_t>((ql[l + 0]  & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = static_cast<int8_t>((ql[l + 0]  >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

inline void dequantize_row_iq4_xs_ref(const block_iq4_xs* x, float* y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* qs = x[i].qs;
        const float d = q4_0::fp16_to_float(x[i].d);

        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) |
                           (((x[i].scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * static_cast<float>(ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0]  = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0x0F]);
                y[j + 16] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
            }
            y += 32;
            qs += 16;
        }
    }
}

inline void dequantize_row_iq3_s_ref(const block_iq3_s* x, float* y, int64_t k) {
    static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {
        const float d = q4_0::fp16_to_float(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;

        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db1 = d * (1.0f + 2.0f * static_cast<float>(x[i].scales[ib32 / 2] & 0x0F));
            const float db2 = d * (1.0f + 2.0f * static_cast<float>(x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(
                    iq3s_grid + (qs[2 * l + 0] | ((static_cast<uint32_t>(qh[0]) << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(
                    iq3s_grid + (qs[2 * l + 1] | ((static_cast<uint32_t>(qh[0]) << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * static_cast<float>(grid1[j]) * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db1 * static_cast<float>(grid2[j]) * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(
                    iq3s_grid + (qs[2 * l + 0] | ((static_cast<uint32_t>(qh[1]) << (8 - 2 * l)) & 256)));
                const uint8_t* grid2 = reinterpret_cast<const uint8_t*>(
                    iq3s_grid + (qs[2 * l + 1] | ((static_cast<uint32_t>(qh[1]) << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * static_cast<float>(grid1[j]) * ((signs[l] & kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db2 * static_cast<float>(grid2[j]) * ((signs[l] & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

// ============================================================================
// Reference GEMV functions
// ============================================================================
inline void gemv_q4_K_ref(const uint8_t* weights, size_t N, size_t K, const float* x, float* y) {
    const size_t bytes_per_row = (K / QK_K) * sizeof(block_q4_K);
    std::vector<float> row_dequant(K);
    for (size_t r = 0; r < N; ++r) {
        const auto* blks = reinterpret_cast<const block_q4_K*>(weights + r * bytes_per_row);
        dequantize_row_q4_K_ref(blks, row_dequant.data(), K);
        float sum = 0.0f;
        for (size_t c = 0; c < K; ++c) {
            sum += row_dequant[c] * x[c];
        }
        y[r] = sum;
    }
}

inline void gemv_q5_K_ref(const uint8_t* weights, size_t N, size_t K, const float* x, float* y) {
    const size_t bytes_per_row = (K / QK_K) * sizeof(block_q5_K);
    std::vector<float> row_dequant(K);
    for (size_t r = 0; r < N; ++r) {
        const auto* blks = reinterpret_cast<const block_q5_K*>(weights + r * bytes_per_row);
        dequantize_row_q5_K_ref(blks, row_dequant.data(), K);
        float sum = 0.0f;
        for (size_t c = 0; c < K; ++c) {
            sum += row_dequant[c] * x[c];
        }
        y[r] = sum;
    }
}

inline void gemv_q3_K_ref(const uint8_t* weights, size_t N, size_t K, const float* x, float* y) {
    const size_t bytes_per_row = (K / QK_K) * sizeof(block_q3_K);
    std::vector<float> row_dequant(K);
    for (size_t r = 0; r < N; ++r) {
        const auto* blks = reinterpret_cast<const block_q3_K*>(weights + r * bytes_per_row);
        dequantize_row_q3_K_ref(blks, row_dequant.data(), K);
        float sum = 0.0f;
        for (size_t c = 0; c < K; ++c) {
            sum += row_dequant[c] * x[c];
        }
        y[r] = sum;
    }
}

inline void gemv_q6_K_ref(const uint8_t* weights, size_t N, size_t K, const float* x, float* y) {
    const size_t bytes_per_row = (K / QK_K) * sizeof(block_q6_K);
    std::vector<float> row_dequant(K);
    for (size_t r = 0; r < N; ++r) {
        const auto* blks = reinterpret_cast<const block_q6_K*>(weights + r * bytes_per_row);
        dequantize_row_q6_K_ref(blks, row_dequant.data(), K);
        float sum = 0.0f;
        for (size_t c = 0; c < K; ++c) {
            sum += row_dequant[c] * x[c];
        }
        y[r] = sum;
    }
}

inline void gemv_iq4_xs_ref(const uint8_t* weights, size_t N, size_t K, const float* x, float* y) {
    const size_t bytes_per_row = (K / QK_K) * sizeof(block_iq4_xs);
    std::vector<float> row_dequant(K);
    for (size_t r = 0; r < N; ++r) {
        const auto* blks = reinterpret_cast<const block_iq4_xs*>(weights + r * bytes_per_row);
        dequantize_row_iq4_xs_ref(blks, row_dequant.data(), K);
        float sum = 0.0f;
        for (size_t c = 0; c < K; ++c) {
            sum += row_dequant[c] * x[c];
        }
        y[r] = sum;
    }
}

inline void gemv_iq3_s_ref(const uint8_t* weights, size_t N, size_t K, const float* x, float* y) {
    const size_t bytes_per_row = (K / QK_K) * sizeof(block_iq3_s);
    std::vector<float> row_dequant(K);
    for (size_t r = 0; r < N; ++r) {
        const auto* blks = reinterpret_cast<const block_iq3_s*>(weights + r * bytes_per_row);
        dequantize_row_iq3_s_ref(blks, row_dequant.data(), K);
        float sum = 0.0f;
        for (size_t c = 0; c < K; ++c) {
            sum += row_dequant[c] * x[c];
        }
        y[r] = sum;
    }
}

} // namespace xdna
