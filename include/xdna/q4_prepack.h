#pragma once

#include "xdna/q4_0_reference.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xdna {

struct PackedTileLayout {
    size_t M;               // Total rows
    size_t K;               // Total columns
    size_t num_columns;     // AIE column count (e.g. 4)
    size_t rows_per_column; // M / num_columns
    size_t group_size;      // 32
    size_t groups_per_row;  // K / 32
    size_t weight_bytes_per_row; // K / 2
    size_t scale_bytes_per_row;  // (K / 32) * sizeof(uint16_t)
    size_t packed_bytes_per_row; // (K / 2) + (K / 32) * 2
    size_t total_packed_bytes;   // M * packed_bytes_per_row
};

// Prepack standard GGUF block_q4_0 interleaved tensor into AIE column-partitioned planar layout:
// Per column:
//   [Rows 0..M_col-1 uint4 packed weights (M_col * K / 2 bytes)]
//   [Rows 0..M_col-1 BF16 scale factors (M_col * (K / 32) * 2 bytes)]
class Q4Prepacker {
public:
    static PackedTileLayout compute_layout(size_t M, size_t K, size_t num_columns = 4, size_t group_size = 32);

    // Prepack GGUF Q4_0 interleaved blocks into AIE tile format
    static void pack(const uint8_t* gguf_q4_0_data,
                     size_t M,
                     size_t K,
                     uint8_t* packed_output,
                     size_t num_columns = 4,
                     size_t group_size = 32);

    // Unpack / reconstruct matrix for bit-exact verification against GGML reference
    static void dequantize_packed(const uint8_t* packed_data,
                                  size_t M,
                                  size_t K,
                                  float* output_f32,
                                  size_t num_columns = 4,
                                  size_t group_size = 32);
};

} // namespace xdna
