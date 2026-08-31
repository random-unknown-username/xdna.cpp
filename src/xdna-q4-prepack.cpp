#include "xdna/q4_prepack.h"
#include "xdna/q4_0_reference.h"

#include <cstring>
#include <stdexcept>

namespace xdna {

PackedTileLayout Q4Prepacker::compute_layout(size_t M, size_t K, size_t num_columns, size_t group_size) {
    if (num_columns == 0) num_columns = 4;
    if (group_size == 0) group_size = 32;
    if (K % group_size != 0) {
        throw std::invalid_argument("K must be a multiple of group_size");
    }
    if (M % num_columns != 0) {
        throw std::invalid_argument("M must be divisible by num_columns");
    }

    PackedTileLayout layout;
    layout.M = M;
    layout.K = K;
    layout.num_columns = num_columns;
    layout.rows_per_column = M / num_columns;
    layout.group_size = group_size;
    layout.groups_per_row = K / group_size;
    layout.weight_bytes_per_row = K / 2;
    layout.scale_bytes_per_row = layout.groups_per_row * sizeof(uint16_t);
    layout.packed_bytes_per_row = layout.weight_bytes_per_row + layout.scale_bytes_per_row;
    layout.total_packed_bytes = M * layout.packed_bytes_per_row;

    return layout;
}

void Q4Prepacker::pack(const uint8_t* gguf_q4_0_data,
                       size_t M,
                       size_t K,
                       uint8_t* packed_output,
                       size_t num_columns,
                       size_t group_size) {
    auto layout = compute_layout(M, K, num_columns, group_size);
    const size_t gguf_bytes_per_row = layout.groups_per_row * sizeof(q4_0::BlockQ4_0); // 18 bytes * (K/32)

    const size_t col_weight_bytes = layout.rows_per_column * layout.weight_bytes_per_row;
    const size_t col_scale_bytes = layout.rows_per_column * layout.scale_bytes_per_row;
    const size_t col_total_bytes = col_weight_bytes + col_scale_bytes;

    for (size_t col = 0; col < num_columns; ++col) {
        uint8_t* col_base = packed_output + col * col_total_bytes;
        uint8_t* col_weights = col_base;
        auto* col_scales = reinterpret_cast<uint16_t*>(col_base + col_weight_bytes);

        for (size_t r = 0; r < layout.rows_per_column; ++r) {
            size_t global_row = col * layout.rows_per_column + r;
            const auto* gguf_blocks = reinterpret_cast<const q4_0::BlockQ4_0*>(
                gguf_q4_0_data + global_row * gguf_bytes_per_row);

            uint8_t* row_weight_ptr = col_weights + r * layout.weight_bytes_per_row;
            uint16_t* row_scale_ptr = col_scales + r * layout.groups_per_row;

            for (size_t g = 0; g < layout.groups_per_row; ++g) {
                // Copy 16 bytes of packed nibbles (32 weights)
                std::memcpy(row_weight_ptr + g * 16, gguf_blocks[g].qs, 16);

                // Convert FP16 scale to BF16
                float scale_f32 = q4_0::fp16_to_float(gguf_blocks[g].d);
                row_scale_ptr[g] = q4_0::float_to_bf16(scale_f32).bits;
            }
        }
    }
}

void Q4Prepacker::dequantize_packed(const uint8_t* packed_data,
                                    size_t M,
                                    size_t K,
                                    float* output_f32,
                                    size_t num_columns,
                                    size_t group_size) {
    auto layout = compute_layout(M, K, num_columns, group_size);
    const size_t col_weight_bytes = layout.rows_per_column * layout.weight_bytes_per_row;
    const size_t col_scale_bytes = layout.rows_per_column * layout.scale_bytes_per_row;
    const size_t col_total_bytes = col_weight_bytes + col_scale_bytes;

    for (size_t col = 0; col < num_columns; ++col) {
        const uint8_t* col_base = packed_data + col * col_total_bytes;
        const uint8_t* col_weights = col_base;
        const auto* col_scales = reinterpret_cast<const uint16_t*>(col_base + col_weight_bytes);

        for (size_t r = 0; r < layout.rows_per_column; ++r) {
            size_t global_row = col * layout.rows_per_column + r;
            float* row_out = output_f32 + global_row * K;

            const uint8_t* row_weight_ptr = col_weights + r * layout.weight_bytes_per_row;
            const uint16_t* row_scale_ptr = col_scales + r * layout.groups_per_row;

            for (size_t g = 0; g < layout.groups_per_row; ++g) {
                q4_0::Bf16 scale_bf16;
                scale_bf16.bits = row_scale_ptr[g];
                float d = q4_0::bf16_to_float(scale_bf16);

                const uint8_t* qs = row_weight_ptr + g * 16;
                for (size_t j = 0; j < 16; ++j) {
                    uint8_t byte = qs[j];
                    int v0 = (byte & 0x0F) - 8;
                    int v1 = (byte >> 4) - 8;

                    row_out[g * 32 + j] = v0 * d;
                    row_out[g * 32 + j + 16] = v1 * d;
                }
            }
        }
    }
}

} // namespace xdna
