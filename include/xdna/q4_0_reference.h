#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace xdna::q4_0 {

// CPU-only GGML Q4_0 oracle/reference for a future fused XDNA1 kernel.
// This module makes no NPU, XRT, or device-execution claim.
inline constexpr std::size_t kValuesPerBlock = 32;
inline constexpr std::size_t kPackedBytesPerBlock = 16;
inline constexpr std::size_t kEncodedBytesPerBlock = 18;

// In the encoded GGML representation, d is the raw IEEE FP16 bit pattern.
// The serialized byte order is little-endian: d[low byte], d[high byte],
// followed immediately by qs[0..15].  qs[j]'s low nibble is value j and its
// high nibble is value j + 16.
struct BlockQ4_0 {
    std::uint16_t d;
    std::uint8_t qs[kPackedBytesPerBlock];
};

static_assert(std::is_standard_layout_v<BlockQ4_0>,
              "Q4_0 block must remain standard-layout");
static_assert(std::is_trivially_copyable_v<BlockQ4_0>,
              "Q4_0 block must remain trivially copyable");
static_assert(offsetof(BlockQ4_0, d) == 0,
              "Q4_0 scale must be the first field");
static_assert(offsetof(BlockQ4_0, qs) == sizeof(std::uint16_t),
              "Q4_0 nibbles must follow the FP16 scale");
static_assert(sizeof(BlockQ4_0) == kEncodedBytesPerBlock,
              "Q4_0 block must be exactly 18 bytes");

// A BF16 activation is represented by its raw IEEE BF16 bit pattern.  This
// distinct type keeps the BF16 activation APIs unambiguous relative to float.
struct Bf16 {
    std::uint16_t bits;
};

static_assert(std::is_standard_layout_v<Bf16>,
              "BF16 activation must remain standard-layout");
static_assert(sizeof(Bf16) == sizeof(std::uint16_t),
              "BF16 activation must be exactly 16 bits");
static_assert(offsetof(Bf16, bits) == 0,
              "BF16 bits must be the first field");

constexpr Bf16 bf16_from_bits(std::uint16_t bits) noexcept {
    return Bf16{bits};
}

// Exact scalar conversion of a raw IEEE FP16 bit pattern to float32.
float fp16_to_float(std::uint16_t bits) noexcept;

// Exact scalar conversion of a raw IEEE BF16 bit pattern to float32.
float bf16_to_float(Bf16 value) noexcept;

// Exact scalar conversion of float32 to BF16 (with round-to-nearest-even).
Bf16 float_to_bf16(float value) noexcept;

// Throws std::invalid_argument when value_count is not a multiple of 32 or
// when its encoded span cannot be represented by std::size_t.
std::size_t encoded_bytes_for_values(std::size_t value_count);

// Every row operation calls this validation.  The supplied byte count must
// equal the encoded GGML span exactly; trailing or missing bytes are errors.
void validate_row(std::size_t value_count, std::size_t encoded_byte_count);

// Contiguous row-major matrix helpers use exactly row_count encoded rows,
// with no padding between rows.
std::size_t encoded_bytes_for_matrix(std::size_t row_count,
                                     std::size_t values_per_row);
void validate_matrix(std::size_t row_count, std::size_t values_per_row,
                     std::size_t encoded_byte_count);

void dequantize_row(const std::uint8_t* encoded_row,
                    std::size_t encoded_byte_count,
                    std::size_t value_count, float* output);

// Explicit float-activation reference dot product.
float dot_f32(const std::uint8_t* encoded_row,
              std::size_t encoded_byte_count,
              std::size_t value_count,
              const float* activations, std::size_t activation_count);

// Explicit BF16-activation reference dot product.
float dot_bf16(const std::uint8_t* encoded_row,
               std::size_t encoded_byte_count,
               std::size_t value_count,
               const Bf16* activations, std::size_t activation_count);

// Explicit float-activation reference GEMV for a contiguous row-major matrix.
void gemv_f32(const std::uint8_t* encoded_matrix,
              std::size_t encoded_matrix_byte_count,
              std::size_t row_count, std::size_t values_per_row,
              const float* activations, std::size_t activation_count,
              float* output, std::size_t output_count);

// Explicit BF16-activation reference GEMV for a contiguous row-major matrix.
void gemv_bf16(const std::uint8_t* encoded_matrix,
               std::size_t encoded_matrix_byte_count,
               std::size_t row_count, std::size_t values_per_row,
               const Bf16* activations, std::size_t activation_count,
               float* output, std::size_t output_count);

}  // namespace xdna::q4_0
