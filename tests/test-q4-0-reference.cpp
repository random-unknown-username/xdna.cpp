#include "xdna/q4_0_reference.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const char* expression) {
    if (!condition) {
        std::cerr << "FAIL: " << expression << '\n';
        std::exit(1);
    }
}

void expect_float(float actual, float expected, const char* expression) {
    if (actual != expected) {
        std::cerr << "FAIL: " << expression << " (expected " << expected
                  << ", got " << actual << ")\n";
        std::exit(1);
    }
}

template <typename Function>
void expect_invalid_argument(Function&& function, const char* substring,
                             const char* expression) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(substring) != std::string::npos) {
            return;
        }
        std::cerr << "FAIL: " << expression << " (unexpected diagnostic: "
                  << error.what() << ")\n";
        std::exit(1);
    } catch (...) {
        std::cerr << "FAIL: " << expression
                  << " (wrong exception type)\n";
        std::exit(1);
    }
    std::cerr << "FAIL: " << expression << " (no exception)\n";
    std::exit(1);
}

std::array<std::uint8_t, xdna::q4_0::kEncodedBytesPerBlock>
known_edge_block() {
    // d = 1.0 (0x3c00), then low nibbles 0..15 and high nibbles 15..0.
    return {0x00, 0x3c, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96,
            0x87, 0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2d, 0x1e, 0x0f};
}

xdna::q4_0::Bf16 bf16_integer(int value) {
    // All integers in this test are exactly representable in BF16.
    switch (value) {
        case 1:
            return xdna::q4_0::bf16_from_bits(0x3f80);
        case 2:
            return xdna::q4_0::bf16_from_bits(0x4000);
        case 3:
            return xdna::q4_0::bf16_from_bits(0x4040);
        case 4:
            return xdna::q4_0::bf16_from_bits(0x4080);
        case 5:
            return xdna::q4_0::bf16_from_bits(0x40a0);
        case 6:
            return xdna::q4_0::bf16_from_bits(0x40c0);
        case 7:
            return xdna::q4_0::bf16_from_bits(0x40e0);
        case 8:
            return xdna::q4_0::bf16_from_bits(0x4100);
        case 9:
            return xdna::q4_0::bf16_from_bits(0x4110);
        case 10:
            return xdna::q4_0::bf16_from_bits(0x4120);
        case 11:
            return xdna::q4_0::bf16_from_bits(0x4130);
        case 12:
            return xdna::q4_0::bf16_from_bits(0x4140);
        case 13:
            return xdna::q4_0::bf16_from_bits(0x4150);
        case 14:
            return xdna::q4_0::bf16_from_bits(0x4160);
        case 15:
            return xdna::q4_0::bf16_from_bits(0x4170);
        case 16:
            return xdna::q4_0::bf16_from_bits(0x4180);
        case 17:
            return xdna::q4_0::bf16_from_bits(0x4188);
        case 18:
            return xdna::q4_0::bf16_from_bits(0x4190);
        case 19:
            return xdna::q4_0::bf16_from_bits(0x4198);
        case 20:
            return xdna::q4_0::bf16_from_bits(0x41a0);
        case 21:
            return xdna::q4_0::bf16_from_bits(0x41a8);
        case 22:
            return xdna::q4_0::bf16_from_bits(0x41b0);
        case 23:
            return xdna::q4_0::bf16_from_bits(0x41b8);
        case 24:
            return xdna::q4_0::bf16_from_bits(0x41c0);
        case 25:
            return xdna::q4_0::bf16_from_bits(0x41c8);
        case 26:
            return xdna::q4_0::bf16_from_bits(0x41d0);
        case 27:
            return xdna::q4_0::bf16_from_bits(0x41d8);
        case 28:
            return xdna::q4_0::bf16_from_bits(0x41e0);
        case 29:
            return xdna::q4_0::bf16_from_bits(0x41e8);
        case 30:
            return xdna::q4_0::bf16_from_bits(0x41f0);
        case 31:
            return xdna::q4_0::bf16_from_bits(0x41f8);
        case 32:
            return xdna::q4_0::bf16_from_bits(0x4200);
        default:
            std::abort();
    }
}

std::array<std::uint8_t, 2 * xdna::q4_0::kEncodedBytesPerBlock>
gemv_matrix() {
    std::array<std::uint8_t, 2 * xdna::q4_0::kEncodedBytesPerBlock> matrix{};
    matrix[0] = 0x00;
    matrix[1] = 0x3c;  // d = 1.0
    for (std::size_t j = 0; j < xdna::q4_0::kPackedBytesPerBlock; ++j) {
        matrix[2 + j] = 0x99;  // both nibbles decode to +1
    }
    matrix[xdna::q4_0::kEncodedBytesPerBlock] = 0x00;
    matrix[xdna::q4_0::kEncodedBytesPerBlock + 1] = 0x38;  // d = 0.5
    for (std::size_t j = 0; j < xdna::q4_0::kPackedBytesPerBlock; ++j) {
        matrix[xdna::q4_0::kEncodedBytesPerBlock + 2 + j] =
            0x77;  // both nibbles decode to -1
    }
    return matrix;
}

}  // namespace

int main() {
    using namespace xdna::q4_0;

    expect(sizeof(BlockQ4_0) == 18, "Q4_0 block is 18 bytes");
    expect(offsetof(BlockQ4_0, d) == 0, "Q4_0 d offset");
    expect(offsetof(BlockQ4_0, qs) == 2, "Q4_0 qs offset");
    expect(encoded_bytes_for_values(0) == 0, "zero-value row span");
    expect(encoded_bytes_for_values(32) == 18, "one-block row span");
    expect(encoded_bytes_for_matrix(2, 32) == 36, "two-row matrix span");

    expect_float(fp16_to_float(0x3c00), 1.0f, "FP16 1.0 conversion");
    expect_float(fp16_to_float(0xc000), -2.0f, "FP16 -2.0 conversion");
    expect_float(fp16_to_float(0x0001), std::ldexp(1.0f, -24),
                 "FP16 smallest subnormal conversion");
    expect_float(fp16_to_float(0x7bff), 65504.0f,
                 "FP16 largest finite conversion");
    expect_float(bf16_to_float(bf16_from_bits(0x3f80)), 1.0f,
                 "BF16 1.0 conversion");
    expect_float(bf16_to_float(bf16_from_bits(0xc000)), -2.0f,
                 "BF16 -2.0 conversion");

    const auto encoded = known_edge_block();
    std::array<float, kValuesPerBlock> dequantized{};
    dequantize_row(encoded.data(), encoded.size(), dequantized.size(),
                   dequantized.data());
    for (std::size_t j = 0; j < kPackedBytesPerBlock; ++j) {
        expect_float(dequantized[j], static_cast<float>(j) - 8.0f,
                     "low nibble ordering/dequantization");
        expect_float(dequantized[j + kPackedBytesPerBlock],
                     7.0f - static_cast<float>(j),
                     "high nibble ordering/dequantization");
    }

    std::array<float, kValuesPerBlock> float_activations{};
    std::array<Bf16, kValuesPerBlock> bf16_activations{};
    for (std::size_t j = 0; j < kValuesPerBlock; ++j) {
        float_activations[j] = static_cast<float>(j + 1);
        bf16_activations[j] = bf16_integer(static_cast<int>(j + 1));
    }
    expect_float(dot_f32(encoded.data(), encoded.size(), kValuesPerBlock,
                         float_activations.data(), float_activations.size()),
                 -264.0f, "Q4_0 dot against float activations");
    expect_float(dot_bf16(encoded.data(), encoded.size(), kValuesPerBlock,
                          bf16_activations.data(), bf16_activations.size()),
                 -264.0f, "Q4_0 dot against BF16 activations");

    const auto matrix = gemv_matrix();
    std::array<float, 2> gemv_float_output{};
    gemv_f32(matrix.data(), matrix.size(), 2, kValuesPerBlock,
             float_activations.data(), float_activations.size(),
             gemv_float_output.data(), gemv_float_output.size());
    expect_float(gemv_float_output[0], 528.0f,
                 "Q4_0 GEMV against float activations row 0");
    expect_float(gemv_float_output[1], -264.0f,
                 "Q4_0 GEMV against float activations row 1");

    std::array<float, 2> gemv_bf16_output{};
    gemv_bf16(matrix.data(), matrix.size(), 2, kValuesPerBlock,
              bf16_activations.data(), bf16_activations.size(),
              gemv_bf16_output.data(), gemv_bf16_output.size());
    expect_float(gemv_bf16_output[0], 528.0f,
                 "Q4_0 GEMV against BF16 activations row 0");
    expect_float(gemv_bf16_output[1], -264.0f,
                 "Q4_0 GEMV against BF16 activations row 1");

    expect_invalid_argument(
        [] { validate_row(31, 18); }, "multiple of 32",
        "reject non-multiple-of-32 row length");
    expect_invalid_argument(
        [] { validate_row(32, 17); }, "requires exactly 18 bytes",
        "reject truncated encoded row");
    expect_invalid_argument(
        [] { validate_row(32, 19); }, "requires exactly 18 bytes",
        "reject trailing encoded row bytes");
    expect_invalid_argument(
        [] { validate_matrix(2, 32, 35); }, "matrix encoded byte span",
        "reject non-exact encoded matrix span");
    expect_invalid_argument(
        [&] {
            dot_f32(encoded.data(), encoded.size(), kValuesPerBlock,
                    float_activations.data(), kValuesPerBlock - 1);
        },
        "float activation length mismatch", "reject float activation length");
    expect_invalid_argument(
        [&] {
            gemv_bf16(matrix.data(), matrix.size(), 2, kValuesPerBlock,
                      bf16_activations.data(), bf16_activations.size(),
                      gemv_bf16_output.data(), 1);
        },
        "GEMV output length mismatch", "reject GEMV output length");

    std::cout << "xdna Q4_0 CPU reference tests: PASS\n";
    return 0;
}
