#include "xdna/q4_0_reference.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace xdna::q4_0 {
namespace {

[[noreturn]] void invalid_argument(const std::string& message) {
    throw std::invalid_argument(message);
}

std::uint16_t load_little_endian_u16(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[0]) |
        (static_cast<std::uint16_t>(bytes[1]) << 8));
}

void require_pointer(const void* pointer, std::size_t count,
                     const char* description) {
    if (count != 0 && pointer == nullptr) {
        invalid_argument(std::string("Q4_0 ") + description +
                         " pointer is null for a non-empty input");
    }
}

void validate_activation_count(std::size_t value_count,
                               std::size_t activation_count,
                               const char* activation_kind) {
    if (value_count != activation_count) {
        invalid_argument(
            std::string("Q4_0 ") + activation_kind +
            " activation length mismatch: expected " +
            std::to_string(value_count) + " values, got " +
            std::to_string(activation_count));
    }
}

std::size_t encoded_bytes_for_values_unchecked(std::size_t value_count) {
    const std::size_t block_count = value_count / kValuesPerBlock;
    if (block_count >
        std::numeric_limits<std::size_t>::max() / kEncodedBytesPerBlock) {
        invalid_argument("Q4_0 encoded byte span overflows size_t for row "
                         "length " +
                         std::to_string(value_count) + " values");
    }
    return block_count * kEncodedBytesPerBlock;
}

std::size_t encoded_bytes_for_matrix_unchecked(std::size_t row_count,
                                               std::size_t row_byte_count) {
    if (row_count != 0 &&
        row_byte_count >
            std::numeric_limits<std::size_t>::max() / row_count) {
        invalid_argument("Q4_0 matrix encoded byte span overflows size_t for " +
                         std::to_string(row_count) + " rows");
    }
    return row_count * row_byte_count;
}

template <typename Activation, typename ConvertActivation>
float dot_unchecked(const std::uint8_t* encoded_row,
                    std::size_t value_count,
                    const Activation* activations,
                    ConvertActivation convert_activation) {
    float sum = 0.0f;
    const std::size_t block_count = value_count / kValuesPerBlock;

    for (std::size_t block_index = 0; block_index < block_count;
         ++block_index) {
        const std::uint8_t* block =
            encoded_row + block_index * kEncodedBytesPerBlock;
        const float d = fp16_to_float(load_little_endian_u16(block));
        const std::uint8_t* qs = block + sizeof(std::uint16_t);
        const std::size_t value_offset = block_index * kValuesPerBlock;

        // This is deliberately the scalar GGML ordering: the low nibble
        // supplies the first 16 values and the high nibble the second 16.
        for (std::size_t j = 0; j < kPackedBytesPerBlock; ++j) {
            const float q0 = static_cast<float>(
                static_cast<int>(qs[j] & 0x0fU) - 8);
            const float w0 = q0 * d;
            sum += w0 * convert_activation(activations[value_offset + j]);
        }
        for (std::size_t j = 0; j < kPackedBytesPerBlock; ++j) {
            const float q1 = static_cast<float>(
                static_cast<int>(qs[j] >> 4U) - 8);
            const float w1 = q1 * d;
            sum += w1 * convert_activation(
                             activations[value_offset + j +
                                        kPackedBytesPerBlock]);
        }
    }
    return sum;
}

}  // namespace

float fp16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign =
        (static_cast<std::uint32_t>(bits & 0x8000U)) << 16;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    std::uint32_t mantissa = bits & 0x03ffU;
    std::uint32_t float_bits = sign;

    if (exponent == 0) {
        if (mantissa == 0) {
            float_bits = sign;
        } else {
            // Normalize a half subnormal into a float32 normal.  The
            // resulting value is exact because the binary significand fits.
            int unbiased_exponent = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --unbiased_exponent;
            }
            mantissa &= 0x03ffU;
            float_bits =
                sign |
                (static_cast<std::uint32_t>(unbiased_exponent + 127) << 23U) |
                (mantissa << 13U);
        }
    } else if (exponent == 0x1fU) {
        // Preserve IEEE infinity/NaN payload bits in the widened mantissa.
        float_bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        const std::uint32_t float_exponent = exponent - 15U + 127U;
        float_bits = sign | (float_exponent << 23U) | (mantissa << 13U);
    }

    float result = 0.0f;
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}

float bf16_to_float(Bf16 value) noexcept {
    const std::uint32_t float_bits =
        static_cast<std::uint32_t>(value.bits) << 16U;
    float result = 0.0f;
    std::memcpy(&result, &float_bits, sizeof(result));
    return result;
}

Bf16 float_to_bf16(float value) noexcept {
    std::uint32_t u = 0;
    std::memcpy(&u, &value, sizeof(u));
    const std::uint32_t lsb = (u >> 16U) & 1U;
    const std::uint32_t rounding_bias = 0x7FFFU + lsb;
    u += rounding_bias;
    return Bf16{static_cast<std::uint16_t>(u >> 16U)};
}

std::size_t encoded_bytes_for_values(std::size_t value_count) {
    if (value_count % kValuesPerBlock != 0) {
        invalid_argument("Q4_0 row length must be a multiple of 32 values "
                         "(got " +
                         std::to_string(value_count) + ")");
    }
    return encoded_bytes_for_values_unchecked(value_count);
}

void validate_row(std::size_t value_count, std::size_t encoded_byte_count) {
    const std::size_t expected = encoded_bytes_for_values(value_count);
    if (encoded_byte_count != expected) {
        invalid_argument(
            "Q4_0 encoded byte span mismatch: row length " +
            std::to_string(value_count) + " values requires exactly " +
            std::to_string(expected) + " bytes, got " +
            std::to_string(encoded_byte_count));
    }
}

std::size_t encoded_bytes_for_matrix(std::size_t row_count,
                                     std::size_t values_per_row) {
    const std::size_t row_byte_count =
        encoded_bytes_for_values(values_per_row);
    return encoded_bytes_for_matrix_unchecked(row_count, row_byte_count);
}

void validate_matrix(std::size_t row_count, std::size_t values_per_row,
                     std::size_t encoded_byte_count) {
    const std::size_t expected =
        encoded_bytes_for_matrix(row_count, values_per_row);
    if (encoded_byte_count != expected) {
        invalid_argument(
            "Q4_0 matrix encoded byte span mismatch: " +
            std::to_string(row_count) + " rows x " +
            std::to_string(values_per_row) + " values requires exactly " +
            std::to_string(expected) + " bytes, got " +
            std::to_string(encoded_byte_count));
    }
}

void dequantize_row(const std::uint8_t* encoded_row,
                    std::size_t encoded_byte_count,
                    std::size_t value_count, float* output) {
    validate_row(value_count, encoded_byte_count);
    require_pointer(encoded_row, encoded_byte_count, "encoded row");
    require_pointer(output, value_count, "output");

    const std::size_t block_count = value_count / kValuesPerBlock;
    for (std::size_t block_index = 0; block_index < block_count;
         ++block_index) {
        const std::uint8_t* block =
            encoded_row + block_index * kEncodedBytesPerBlock;
        const float d = fp16_to_float(load_little_endian_u16(block));
        const std::uint8_t* qs = block + sizeof(std::uint16_t);
        const std::size_t value_offset = block_index * kValuesPerBlock;

        // Matches ggml-quants.c: low nibbles are the first half of the
        // block, high nibbles are the second half, and zero maps to -8.
        for (std::size_t j = 0; j < kPackedBytesPerBlock; ++j) {
            const int x0 = static_cast<int>(qs[j] & 0x0fU) - 8;
            const int x1 = static_cast<int>(qs[j] >> 4U) - 8;
            output[value_offset + j] = static_cast<float>(x0) * d;
            output[value_offset + j + kPackedBytesPerBlock] =
                static_cast<float>(x1) * d;
        }
    }
}

float dot_f32(const std::uint8_t* encoded_row,
              std::size_t encoded_byte_count,
              std::size_t value_count, const float* activations,
              std::size_t activation_count) {
    validate_row(value_count, encoded_byte_count);
    validate_activation_count(value_count, activation_count, "float");
    require_pointer(encoded_row, encoded_byte_count, "encoded row");
    require_pointer(activations, activation_count, "float activation");

    return dot_unchecked(encoded_row, value_count, activations,
                         [](float value) { return value; });
}

float dot_bf16(const std::uint8_t* encoded_row,
               std::size_t encoded_byte_count,
               std::size_t value_count, const Bf16* activations,
               std::size_t activation_count) {
    validate_row(value_count, encoded_byte_count);
    validate_activation_count(value_count, activation_count, "BF16");
    require_pointer(encoded_row, encoded_byte_count, "encoded row");
    require_pointer(activations, activation_count, "BF16 activation");

    return dot_unchecked(encoded_row, value_count, activations,
                         [](Bf16 value) { return bf16_to_float(value); });
}

void gemv_f32(const std::uint8_t* encoded_matrix,
              std::size_t encoded_matrix_byte_count,
              std::size_t row_count, std::size_t values_per_row,
              const float* activations, std::size_t activation_count,
              float* output, std::size_t output_count) {
    validate_matrix(row_count, values_per_row, encoded_matrix_byte_count);
    validate_activation_count(values_per_row, activation_count, "float");
    if (output_count != row_count) {
        invalid_argument("Q4_0 GEMV output length mismatch: expected " +
                         std::to_string(row_count) + " values, got " +
                         std::to_string(output_count));
    }
    require_pointer(encoded_matrix, encoded_matrix_byte_count,
                    "encoded matrix");
    require_pointer(activations, activation_count, "float activation");
    require_pointer(output, output_count, "GEMV output");

    const std::size_t row_byte_count =
        encoded_bytes_for_values(values_per_row);
    for (std::size_t row = 0; row < row_count; ++row) {
        output[row] = dot_unchecked(
            encoded_matrix + row * row_byte_count, values_per_row,
            activations, [](float value) { return value; });
    }
}

void gemv_bf16(const std::uint8_t* encoded_matrix,
               std::size_t encoded_matrix_byte_count,
               std::size_t row_count, std::size_t values_per_row,
               const Bf16* activations, std::size_t activation_count,
               float* output, std::size_t output_count) {
    validate_matrix(row_count, values_per_row, encoded_matrix_byte_count);
    validate_activation_count(values_per_row, activation_count, "BF16");
    if (output_count != row_count) {
        invalid_argument("Q4_0 GEMV output length mismatch: expected " +
                         std::to_string(row_count) + " values, got " +
                         std::to_string(output_count));
    }
    require_pointer(encoded_matrix, encoded_matrix_byte_count,
                    "encoded matrix");
    require_pointer(activations, activation_count, "BF16 activation");
    require_pointer(output, output_count, "GEMV output");

    const std::size_t row_byte_count =
        encoded_bytes_for_values(values_per_row);
    for (std::size_t row = 0; row < row_count; ++row) {
        output[row] = dot_unchecked(
            encoded_matrix + row * row_byte_count, values_per_row, activations,
            [](Bf16 value) { return bf16_to_float(value); });
    }
}

}  // namespace xdna::q4_0
