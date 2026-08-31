#include "xdna/ggml-xdna.h"
#include "xdna/q4_0_reference.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    std::cout << "--- Testing GGML XDNA Backend Registration & Offload Logic ---\n";

    ggml_backend_t backend = ggml_backend_xdna_init(0);
    assert(backend != nullptr);
    assert(ggml_backend_is_xdna(backend) == true);
    std::cout << "  Backend init & identification: PASS\n";

    // 1. Check op support for Q4_0 MUL_MAT
    ggml_tensor_xdna w_q4_0{};
    w_q4_0.type = GGML_TYPE_Q4_0;
    w_q4_0.ne[0] = 5120;
    w_q4_0.ne[1] = 17408;
    w_q4_0.ne[2] = 1;
    w_q4_0.ne[3] = 1;

    ggml_tensor_xdna act_f32{};
    act_f32.type = GGML_TYPE_F32;
    act_f32.ne[0] = 5120;
    act_f32.ne[1] = 1;
    act_f32.ne[2] = 1;
    act_f32.ne[3] = 1;

    ggml_tensor_xdna op_mul_mat{};
    op_mul_mat.op = GGML_OP_MUL_MAT;
    op_mul_mat.type = GGML_TYPE_F32;
    op_mul_mat.ne[0] = 17408;
    op_mul_mat.ne[1] = 1;
    op_mul_mat.src[0] = &w_q4_0;
    op_mul_mat.src[1] = &act_f32;

    bool supported = ggml_backend_xdna_supports_op(backend, &op_mul_mat);
    assert(supported == true);
    std::cout << "  Q4_0 MUL_MAT (5120 x 17408) op support: PASS (Claimed for XDNA)\n";

    // 2. Check fallback for F32 weights
    ggml_tensor_xdna w_f32 = w_q4_0;
    w_f32.type = GGML_TYPE_F32;
    op_mul_mat.src[0] = &w_f32;
    assert(ggml_backend_xdna_supports_op(backend, &op_mul_mat) == false);
    std::cout << "  F32 MUL_MAT fallback to CPU: PASS\n";

    // 3. Check fallback for non-supported op (RMS_NORM)
    ggml_tensor_xdna op_norm{};
    op_norm.op = GGML_OP_RMS_NORM;
    assert(ggml_backend_xdna_supports_op(backend, &op_norm) == false);
    std::cout << "  RMS_NORM fallback to CPU: PASS\n";

    // 4. Test execution through compute graph API
    const size_t N = 128;
    const size_t K = 256;
    size_t weight_bytes = (N * (K / 32)) * 18;
    std::vector<uint8_t> weight_data(weight_bytes);

    for (size_t r = 0; r < N; ++r) {
        auto* blks = reinterpret_cast<xdna::q4_0::BlockQ4_0*>(weight_data.data() + r * (K / 32) * 18);
        for (size_t b = 0; b < K / 32; ++b) {
            blks[b].d = 0x3C00; // FP16 1.0
            for (size_t j = 0; j < 16; ++j) {
                blks[b].qs[j] = static_cast<uint8_t>(0x99); // (9-8=1, 9-8=1)
            }
        }
    }

    std::vector<float> act_data(K, 0.5f);
    std::vector<float> out_data(N, 0.0f);

    w_q4_0.ne[0] = K;
    w_q4_0.ne[1] = N;
    w_q4_0.data = weight_data.data();
    w_q4_0.type = GGML_TYPE_Q4_0;

    act_f32.ne[0] = K;
    act_f32.ne[1] = 1;
    act_f32.data = act_data.data();

    op_mul_mat.src[0] = &w_q4_0;
    op_mul_mat.src[1] = &act_f32;
    op_mul_mat.data = out_data.data();

    struct ggml_tensor_xdna* nodes[1] = { &op_mul_mat };
    bool ok = ggml_backend_xdna_compute_graph(backend, nodes, 1);
    assert(ok == true);

    // Verify output: each row dot product = 256 * (1.0 * 0.5) = 128.0
    for (size_t i = 0; i < N; ++i) {
        assert(std::abs(out_data[i] - 128.0f) < 1e-3);
    }
    std::cout << "  Compute graph execution & accuracy: PASS (out[0..N-1] = 128.0)\n";

    ggml_backend_xdna_free(backend);
    std::cout << "  Backend cleanup: PASS\n";

    std::cout << "ALL BACKEND TESTS PASSED!\n";
    return 0;
}
