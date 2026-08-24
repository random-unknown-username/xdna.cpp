#include "xdna/ggml-xdna.h"
#include "xdna/xdna_gemv_engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

static const uint32_t XDNA_BACKEND_MAGIC = 0x58444E41; // "XDNA"

struct XdnaBackendContext {
    uint32_t magic = XDNA_BACKEND_MAGIC;
    size_t device_index = 0;
    std::unique_ptr<xdna::XdnaGemvEngine> engine;
    std::mutex mtx;
};

} // namespace

extern "C" {

ggml_backend_t ggml_backend_xdna_init(size_t dev_num) {
    auto ctx = std::make_unique<XdnaBackendContext>();
    ctx->device_index = dev_num;
    ctx->engine = std::make_unique<xdna::XdnaGemvEngine>(dev_num, 4);

    return reinterpret_cast<ggml_backend_t>(ctx.release());
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    if (!backend) return false;
    auto* ctx = reinterpret_cast<const XdnaBackendContext*>(backend);
    return ctx->magic == XDNA_BACKEND_MAGIC;
}

void ggml_backend_xdna_free(ggml_backend_t backend) {
    if (!backend) return;
    auto* ctx = reinterpret_cast<XdnaBackendContext*>(backend);
    delete ctx;
}

ggml_backend_buffer_type_t ggml_backend_xdna_buffer_type(size_t dev_num) {
    (void)dev_num;
    return reinterpret_cast<ggml_backend_buffer_type_t>(0x1);
}

bool ggml_backend_xdna_supports_op(ggml_backend_t backend, const struct ggml_tensor_xdna* op) {
    (void)backend;
    if (!op) return false;

    if (op->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor_xdna* src0 = op->src[0]; // Weights [K, N]
        const struct ggml_tensor_xdna* src1 = op->src[1]; // Activations [K, M]

        if (src0 && src1 && src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
            int64_t K = src0->ne[0];
            int64_t N = src0->ne[1];
            int64_t M = src1->ne[1];

            if (M == 1 && K % 32 == 0 && N % 4 == 0) {
                return true;
            }
        }
    }

    return false;
}

bool ggml_backend_xdna_compute_graph(ggml_backend_t backend, struct ggml_tensor_xdna** nodes, size_t n_nodes) {
    if (!backend || !nodes || n_nodes == 0) return false;
    auto* ctx = reinterpret_cast<XdnaBackendContext*>(backend);
    std::lock_guard<std::mutex> lock(ctx->mtx);

    for (size_t i = 0; i < n_nodes; ++i) {
        auto* node = nodes[i];
        if (!node) continue;

        if (node->op == GGML_OP_MUL_MAT) {
            auto* src0 = node->src[0]; // Weights [K, N]
            auto* src1 = node->src[1]; // Activations [K, M]

            if (src0 && src1 && src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
                int64_t K = src0->ne[0];
                int64_t N = src0->ne[1];
                int64_t M = src1->ne[1];

                if (M == 1 && node->data && src0->data && src1->data) {
                    size_t bytes_per_row = (K / 32) * 18;
                    size_t total_weight_bytes = N * bytes_per_row;

                    const auto* weights = reinterpret_cast<const uint8_t*>(src0->data);
                    const auto* act = reinterpret_cast<const float*>(src1->data);
                    auto* out = reinterpret_cast<float*>(node->data);

                    ctx->engine->run(weights, total_weight_bytes, N, K, act, out);
                    continue;
                }
            }
        }
    }

    return true;
}

int ggml_backend_score(void) {
    return 100;
}

} // extern "C"
