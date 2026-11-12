#include "ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-backend.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "ggml-impl.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "xdna/ggml-xdna.h"
#include "xdna/xdna_gemv_engine.h"
#include "xdna/device.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
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

    explicit XdnaBackendContext(size_t dev_num = 0)
        : device_index(dev_num),
          engine(std::make_unique<xdna::XdnaGemvEngine>(dev_num, 4)) {}
};

struct XdnaBufferContext {
    void* data = nullptr;
    size_t size = 0;
    bool is_owned = false;

    ~XdnaBufferContext() {
        if (is_owned && data) {
            std::free(data);
            data = nullptr;
        }
    }
};

static inline ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0x58, 0x44, 0x4e, 0x41, 0x01, 0x02, 0x03, 0x04,
                              0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c };
    return &guid;
}

static inline bool xdna_guid_matches(ggml_guid_t a, ggml_guid_t b) {
    return a != nullptr && b != nullptr && std::memcmp(a, b, sizeof(ggml_guid)) == 0;
}

static inline XdnaBackendContext* get_xdna_ctx(ggml_backend_t backend) {
    if (!backend) return nullptr;
    auto* b = reinterpret_cast<const struct ggml_backend*>(backend);
    if (b->guid != nullptr && xdna_guid_matches(b->guid, ggml_backend_xdna_guid()) && b->context != nullptr) {
        return reinterpret_cast<XdnaBackendContext*>(b->context);
    }
    auto* raw = reinterpret_cast<XdnaBackendContext*>(backend);
    if (raw->magic == XDNA_BACKEND_MAGIC) {
        return raw;
    }
    return nullptr;
}

static inline bool is_tensor_loader_probe(const struct ggml_tensor* src0) {
    if (!src0 || !src0->buffer) return false;
    if (src0->buffer->size == 0) return true;
    if (src0->buffer->buft == ggml_backend_xdna_buffer_type(0) ||
        src0->buffer->buft == ggml_backend_xdna_host_buffer_type()) {
        return true;
    }
    return false;
}

static inline bool is_tensor_loader_probe(const struct ggml_tensor_xdna*) {
    return false;
}

template <typename TensorType>
static bool supports_op_impl(const TensorType* op) {
    if (!op) return false;

    if (op->op == GGML_OP_MUL_MAT) {
        const auto* src0 = op->src[0]; // Weights [K, N, ne2, ne3]
        const auto* src1 = op->src[1]; // Activations [K, M, ne2, ne3]

        if (!src0 || !src1) return false;
        if (src1->type != GGML_TYPE_F32) return false;

        const int64_t K = src0->ne[0];
        const int64_t N = src0->ne[1];
        const int64_t M = src1->ne[1];

        const bool loader_or_xdna_wgt = is_tensor_loader_probe(src0);
        if (!loader_or_xdna_wgt && (M != 1 || N < 32 || (N % 4 != 0))) return false;
        if (loader_or_xdna_wgt && (N < 32 || (N % 4 != 0))) return false;

        if (src0->type == GGML_TYPE_Q4_0) {
            if (K % 32 != 0) return false;
            const size_t bytes_per_row = (K / 32) * 18;
            if (src0->nb[1] == 0 || src0->nb[1] == bytes_per_row) {
                return true;
            }
        } else if (src0->type == GGML_TYPE_Q4_K ||
                   src0->type == GGML_TYPE_Q5_K ||
                   src0->type == GGML_TYPE_Q3_K ||
                   src0->type == GGML_TYPE_Q6_K ||
                   src0->type == GGML_TYPE_IQ4_XS ||
                   src0->type == GGML_TYPE_IQ3_S ||
                   src0->type == GGML_TYPE_IQ3_XXS ||
                   src0->type == GGML_TYPE_IQ2_S) {
            if (K % 256 != 0) return false;
            size_t bpb = 144;
            if (src0->type == GGML_TYPE_Q4_K) bpb = 144;
            else if (src0->type == GGML_TYPE_Q5_K) bpb = 176;
            else if (src0->type == GGML_TYPE_Q3_K) bpb = 110;
            else if (src0->type == GGML_TYPE_Q6_K) bpb = 210;
            else if (src0->type == GGML_TYPE_IQ4_XS) bpb = 136;
            else if (src0->type == GGML_TYPE_IQ3_S) bpb = 110;
            else if (src0->type == GGML_TYPE_IQ3_XXS) bpb = 98;
            else if (src0->type == GGML_TYPE_IQ2_S) bpb = 82;

            const size_t bytes_per_row = (K / 256) * bpb;
            if (src0->nb[1] == 0 || src0->nb[1] == bytes_per_row) {
                return true;
            }
        }
    }

    return false;
}

template <typename TensorType>
static bool process_mul_mat_node(XdnaBackendContext* ctx, TensorType* node) {
    if (!node || node->op != GGML_OP_MUL_MAT) {
        return false;
    }

    auto* src0 = node->src[0]; // Weights [K, N, ne2, ne3]
    auto* src1 = node->src[1]; // Activations [K, M, ne2, ne3]

    if (!src0 || !src1) {
        fprintf(stderr, "[xdna error] node '%s' missing src0/src1\n", node->name);
        return false;
    }
    if (src1->type != GGML_TYPE_F32) {
        fprintf(stderr, "[xdna error] node '%s' src1 type %d != F32\n", node->name, src1->type);
        return false;
    }
    if (!node->data || !src0->data || !src1->data) {
        fprintf(stderr, "[xdna error] node '%s' null data ptrs: node->data=%p src0->data=%p src1->data=%p\n",
                node->name, node->data, src0->data, src1->data);
        return false;
    }

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    if (M == 0 || N == 0) {
        return true; // No-op for 0-token / 0-row tensor
    }
    if (M < 0 || N < 0 || K < 0) {
        fprintf(stderr, "[xdna error] node '%s' negative dim: M=%ld N=%ld K=%ld\n",
                node->name, (long)M, (long)N, (long)K);
        return false;
    }

    size_t bytes_per_row = 0;
    xdna::QuantType qtype = xdna::QuantType::Q4_0;

    if (src0->type == GGML_TYPE_Q4_0) {
        if (K % 32 != 0) {
            fprintf(stderr, "[xdna error] node '%s' Q4_0 K=%ld not div by 32\n", node->name, (long)K);
            return false;
        }
        bytes_per_row = (K / 32) * 18;
        qtype = xdna::QuantType::Q4_0;
    } else if (src0->type == GGML_TYPE_Q4_K) {
        if (K % 256 != 0) {
            fprintf(stderr, "[xdna error] node '%s' Q4_K K=%ld not div by 256\n", node->name, (long)K);
            return false;
        }
        bytes_per_row = (K / 256) * 144;
        qtype = xdna::QuantType::Q4_K;
    } else if (src0->type == GGML_TYPE_Q5_K) {
        if (K % 256 != 0) {
            fprintf(stderr, "[xdna error] node '%s' Q5_K K=%ld not div by 256\n", node->name, (long)K);
            return false;
        }
        bytes_per_row = (K / 256) * 176;
        qtype = xdna::QuantType::Q5_K;
    } else if (src0->type == GGML_TYPE_Q3_K) {
        if (K % 256 != 0) {
            fprintf(stderr, "[xdna error] node '%s' Q3_K K=%ld not div by 256\n", node->name, (long)K);
            return false;
        }
        bytes_per_row = (K / 256) * 110;
        qtype = xdna::QuantType::Q3_K;
    } else if (src0->type == GGML_TYPE_Q6_K) {
        if (K % 256 != 0) {
            fprintf(stderr, "[xdna error] node '%s' Q6_K K=%ld not div by 256\n", node->name, (long)K);
            return false;
        }
        bytes_per_row = (K / 256) * 210;
        qtype = xdna::QuantType::Q6_K;
    } else if (src0->type == GGML_TYPE_IQ4_XS) {
        if (K % 256 != 0) {
            fprintf(stderr, "[xdna error] node '%s' IQ4_XS K=%ld not div by 256\n", node->name, (long)K);
            return false;
        }
        bytes_per_row = (K / 256) * 136;
        qtype = xdna::QuantType::IQ4_XS;
    } else if (src0->type == GGML_TYPE_IQ3_S) {
        if (K % 256 != 0) {
            fprintf(stderr, "[xdna error] node '%s' IQ3_S K=%ld not div by 256\n", node->name, (long)K);
            return false;
        }
        bytes_per_row = (K / 256) * 110;
        qtype = xdna::QuantType::IQ3_S;
    } else if (src0->type == GGML_TYPE_IQ3_XXS) {
        if (K % 256 != 0) {
            fprintf(stderr, "[xdna error] node '%s' IQ3_XXS K=%ld not div by 256\n", node->name, (long)K);
            return false;
        }
        bytes_per_row = (K / 256) * 98;
        qtype = xdna::QuantType::IQ3_XXS;
    } else if (src0->type == GGML_TYPE_IQ2_S) {
        if (K % 256 != 0) {
            fprintf(stderr, "[xdna error] node '%s' IQ2_S K=%ld not div by 256\n", node->name, (long)K);
            return false;
        }
        bytes_per_row = (K / 256) * 82;
        qtype = xdna::QuantType::IQ2_S;
    } else {
        fprintf(stderr, "[xdna error] node '%s' unsupported src0 type=%d\n", node->name, src0->type);
        return false;
    }

    if (src0->nb[1] != 0 && src0->nb[1] != bytes_per_row) {
        fprintf(stderr, "[xdna error] node '%s' src0 nb1 mismatch: %zu != %zu\n", node->name, (size_t)src0->nb[1], bytes_per_row);
        return false;
    }

    const size_t total_weight_bytes = N * bytes_per_row;
    const int64_t ne2 = node->ne[2] > 0 ? node->ne[2] : 1;
    const int64_t ne3 = node->ne[3] > 0 ? node->ne[3] : 1;

    const size_t dst_stride1 = (node->nb[1] > 0) ? node->nb[1] : (N * sizeof(float));
    const size_t dst_stride2 = (node->nb[2] > 0) ? node->nb[2] : (dst_stride1 * M);
    const size_t dst_stride3 = (node->nb[3] > 0) ? node->nb[3] : (dst_stride2 * ne2);

    const size_t src1_stride1 = (src1->nb[1] > 0) ? src1->nb[1] : (K * sizeof(float));
    const size_t src1_stride2 = (src1->ne[2] > 1) ? ((src1->nb[2] > 0) ? src1->nb[2] : (src1_stride1 * M)) : 0;
    const size_t src1_stride3 = (src1->ne[3] > 1) ? ((src1->nb[3] > 0) ? src1->nb[3] : (src1_stride2 * ne2)) : 0;

    const size_t src0_stride2 = (src0->ne[2] > 1) ? ((src0->nb[2] > 0) ? src0->nb[2] : total_weight_bytes) : 0;
    const size_t src0_stride3 = (src0->ne[3] > 1) ? ((src0->nb[3] > 0) ? src0->nb[3] : (src0_stride2 * ne2)) : 0;

    const uint8_t* base_weights = reinterpret_cast<const uint8_t*>(src0->data);
    const float*   base_act     = reinterpret_cast<const float*>(src1->data);
    float*         base_out     = reinterpret_cast<float*>(node->data);

    try {
        for (int64_t i3 = 0; i3 < ne3; ++i3) {
            for (int64_t i2 = 0; i2 < ne2; ++i2) {
                const uint8_t* w_ptr = base_weights + i3 * src0_stride3 + i2 * src0_stride2;
                const char*    a_ptr = reinterpret_cast<const char*>(base_act) + i3 * src1_stride3 + i2 * src1_stride2;
                char*          o_ptr = reinterpret_cast<char*>(base_out)       + i3 * dst_stride3  + i2 * dst_stride2;

                for (int64_t m = 0; m < M; ++m) {
                    const float* a_m = reinterpret_cast<const float*>(a_ptr + m * src1_stride1);
                    float*       o_m = reinterpret_cast<float*>(o_ptr + m * dst_stride1);

                    if (qtype == xdna::QuantType::Q4_0) {
                        ctx->engine->run(w_ptr, total_weight_bytes, N, K, a_m, o_m);
                    } else {
                        ctx->engine->run_quant(qtype, w_ptr, total_weight_bytes, N, K, a_m, o_m);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[xdna error] node '%s' exception during GEMV dispatch: %s\n", node->name, e.what());
        return false;
    } catch (...) {
        fprintf(stderr, "[xdna error] node '%s' unknown exception during GEMV dispatch\n", node->name);
        return false;
    }

    return true;
}

} // namespace

// ============================================================================
// 1. GGML Backend Buffer Interface (ggml_backend_buffer_i)
// ============================================================================

static void ggml_backend_xdna_buf_free_buffer(ggml_backend_buffer_t buffer) {
    if (!buffer) return;
    auto* ctx = static_cast<XdnaBufferContext*>(buffer->context);
    delete ctx;
    buffer->context = nullptr;
}

static void* ggml_backend_xdna_buf_get_base(ggml_backend_buffer_t buffer) {
    if (!buffer || !buffer->context) return nullptr;
    auto* ctx = static_cast<XdnaBufferContext*>(buffer->context);
    return ctx->data;
}

static enum ggml_status ggml_backend_xdna_buf_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    (void)buffer;
    (void)tensor;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_xdna_buf_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    (void)buffer;
    if (!tensor || !tensor->data) return;
    std::memset(static_cast<char*>(tensor->data) + offset, value, size);
}

static void ggml_backend_xdna_buf_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    (void)buffer;
    if (!tensor || !tensor->data || !data) return;
    std::memcpy(static_cast<char*>(tensor->data) + offset, data, size);
}

static void ggml_backend_xdna_buf_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    (void)buffer;
    if (!tensor || !tensor->data || !data) return;
    std::memcpy(data, static_cast<const char*>(tensor->data) + offset, size);
}

static void ggml_backend_xdna_buf_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    if (!buffer || !buffer->context) return;
    auto* ctx = static_cast<XdnaBufferContext*>(buffer->context);
    if (ctx->data && buffer->size > 0) {
        std::memset(ctx->data, value, buffer->size);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_xdna_buffer_i = {
    /* .free_buffer   = */ ggml_backend_xdna_buf_free_buffer,
    /* .get_base      = */ ggml_backend_xdna_buf_get_base,
    /* .init_tensor   = */ ggml_backend_xdna_buf_init_tensor,
    /* .memset_tensor = */ ggml_backend_xdna_buf_memset_tensor,
    /* .set_tensor    = */ ggml_backend_xdna_buf_set_tensor,
    /* .get_tensor    = */ ggml_backend_xdna_buf_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ ggml_backend_xdna_buf_clear,
    /* .reset         = */ nullptr,
    /* .set_tensor_n  = */ nullptr,
    /* .get_tensor_n  = */ nullptr,
    /* .cpy_tensor_n  = */ nullptr,
};

// ============================================================================
// 2. GGML Backend Buffer Type Interface (ggml_backend_buffer_type_i)
// ============================================================================

static const char * ggml_backend_xdna_buft_get_name(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return "XDNA";
}

static ggml_backend_buffer_t ggml_backend_xdna_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void* ptr = nullptr;
    const size_t alloc_sz = (size == 0) ? 4096 : size;
    if (posix_memalign(&ptr, 4096, alloc_sz) != 0) {
        return nullptr;
    }
    auto* ctx = new XdnaBufferContext();
    ctx->data = ptr;
    ctx->size = size;
    ctx->is_owned = true;

    auto* buf = static_cast<ggml_backend_buffer*>(std::calloc(1, sizeof(struct ggml_backend_buffer)));
    buf->iface = ggml_backend_xdna_buffer_i;
    buf->buft = buft;
    buf->context = ctx;
    buf->size = size;
    buf->usage = GGML_BACKEND_BUFFER_USAGE_ANY;
    return buf;
}

static size_t ggml_backend_xdna_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return 4096;
}

static size_t ggml_backend_xdna_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return SIZE_MAX;
}

static size_t ggml_backend_xdna_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    (void)buft;
    if (!tensor) return 0;
    typedef size_t (*fn_nbytes)(const struct ggml_tensor*);
    static auto* p_nbytes = reinterpret_cast<fn_nbytes>(dlsym(RTLD_DEFAULT, "ggml_nbytes"));
    if (p_nbytes) {
        return p_nbytes(tensor);
    }
    size_t blck_size = 1;
    size_t type_size = 4;
    switch (tensor->type) {
        case GGML_TYPE_F32:  type_size = 4; blck_size = 1; break;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: type_size = 2; blck_size = 1; break;
        case GGML_TYPE_Q4_0: type_size = 18; blck_size = 32; break;
        case GGML_TYPE_Q4_1: type_size = 20; blck_size = 32; break;
        case GGML_TYPE_Q5_0: type_size = 22; blck_size = 32; break;
        case GGML_TYPE_Q5_1: type_size = 24; blck_size = 32; break;
        case GGML_TYPE_Q8_0: type_size = 34; blck_size = 32; break;
        case GGML_TYPE_Q4_K:   type_size = 144; blck_size = 256; break;
        case GGML_TYPE_Q5_K:   type_size = 176; blck_size = 256; break;
        case GGML_TYPE_Q3_K:   type_size = 110; blck_size = 256; break;
        case GGML_TYPE_Q6_K:   type_size = 210; blck_size = 256; break;
        case GGML_TYPE_IQ4_XS: type_size = 136; blck_size = 256; break;
        case GGML_TYPE_IQ3_S:  type_size = 110; blck_size = 256; break;
        case GGML_TYPE_I8:   type_size = 1; blck_size = 1; break;
        case GGML_TYPE_I16:  type_size = 2; blck_size = 1; break;
        case GGML_TYPE_I32:  type_size = 4; blck_size = 1; break;
        case GGML_TYPE_I64:  type_size = 8; blck_size = 1; break;
        case GGML_TYPE_F64:  type_size = 8; blck_size = 1; break;
        default:             type_size = 4; blck_size = 1; break;
    }
    const int64_t ne0 = tensor->ne[0] > 0 ? tensor->ne[0] : 1;
    const int64_t ne1 = tensor->ne[1] > 0 ? tensor->ne[1] : 1;
    const int64_t ne2 = tensor->ne[2] > 0 ? tensor->ne[2] : 1;
    const int64_t ne3 = tensor->ne[3] > 0 ? tensor->ne[3] : 1;
    return (static_cast<size_t>(ne0) / blck_size) * type_size * ne1 * ne2 * ne3;
}

static bool ggml_backend_xdna_buft_is_host(ggml_backend_buffer_type_t buft) {
    (void)buft;
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_xdna_buffer_type_i = {
    /* .get_name         = */ ggml_backend_xdna_buft_get_name,
    /* .alloc_buffer     = */ ggml_backend_xdna_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ ggml_backend_xdna_buft_get_alignment,
    /* .get_max_size     = */ ggml_backend_xdna_buft_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_xdna_buft_get_alloc_size,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ ggml_backend_xdna_buft_is_host,
};

// ============================================================================
// Forward Declarations of Static Instances
// ============================================================================

extern struct ggml_backend_reg ggml_backend_xdna_reg_instance;
extern struct ggml_backend_device ggml_backend_xdna_device_instance;

static struct ggml_backend_buffer_type ggml_backend_xdna_buffer_type_instance = {
    /* .iface   = */ ggml_backend_xdna_buffer_type_i,
    /* .device  = */ &ggml_backend_xdna_device_instance,
    /* .context = */ nullptr,
};

static struct ggml_backend_buffer_type ggml_backend_xdna_host_buffer_type_instance = {
    /* .iface   = */ ggml_backend_xdna_buffer_type_i,
    /* .device  = */ &ggml_backend_xdna_device_instance,
    /* .context = */ nullptr,
};

// ============================================================================
// 3. GGML Backend Device Interface (ggml_backend_device_i)
// ============================================================================

static const char * ggml_backend_xdna_dev_get_name(ggml_backend_dev_t dev) {
    (void)dev;
    return "XDNA";
}

static const char * ggml_backend_xdna_dev_get_description(ggml_backend_dev_t dev) {
    (void)dev;
    return "AMD Ryzen AI NPU (Hawk Point XDNA1)";
}

static void ggml_backend_xdna_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    (void)dev;
    if (free)  *free  = 16ULL * 1024 * 1024 * 1024;
    if (total) *total = 16ULL * 1024 * 1024 * 1024;
}

static enum ggml_backend_dev_type ggml_backend_xdna_dev_get_type(ggml_backend_dev_t dev) {
    (void)dev;
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_xdna_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    if (!props) return;
    props->name        = ggml_backend_xdna_dev_get_name(dev);
    props->description = ggml_backend_xdna_dev_get_description(dev);
    props->type        = ggml_backend_xdna_dev_get_type(dev);
    ggml_backend_xdna_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id   = "0000:c4:00.0";
    props->caps.async                 = false;
    props->caps.host_buffer           = true;
    props->caps.buffer_from_host_ptr  = true;
    props->caps.events                = false;
    props->caps.mmap_support          = true;
}

static ggml_backend_t ggml_backend_xdna_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    (void)dev;
    (void)params;
    return ggml_backend_xdna_init(0);
}

static ggml_backend_buffer_type_t ggml_backend_xdna_dev_get_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return &ggml_backend_xdna_buffer_type_instance;
}

static ggml_backend_buffer_type_t ggml_backend_xdna_dev_get_host_buffer_type(ggml_backend_dev_t dev) {
    (void)dev;
    return &ggml_backend_xdna_host_buffer_type_instance;
}

static ggml_backend_buffer_t ggml_backend_xdna_dev_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    (void)max_tensor_size;
    if (!ptr || size == 0) return nullptr;

    auto* ctx = new XdnaBufferContext();
    ctx->data = ptr;
    ctx->size = size;
    ctx->is_owned = false;

    auto* buf = static_cast<ggml_backend_buffer*>(std::calloc(1, sizeof(struct ggml_backend_buffer)));
    buf->iface = ggml_backend_xdna_buffer_i;
    buf->buft = ggml_backend_xdna_dev_get_buffer_type(dev);
    buf->context = ctx;
    buf->size = size;
    buf->usage = GGML_BACKEND_BUFFER_USAGE_ANY;
    return buf;
}

static bool ggml_backend_xdna_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    bool res = supports_op_impl(op);
    static const bool trace = (std::getenv("XDNA_TRACE_SUPPORTS") != nullptr);
    if (trace && op && op->op == GGML_OP_MUL_MAT) {
        const auto* src0 = op->src[0];
        const auto* src1 = op->src[1];
        fprintf(stderr, "[supports_op] %-25s | type=%2d | N=%6ld K=%5ld M=%3ld buf_sz=%zu | -> %s\n",
                op->name, src0 ? src0->type : -1,
                src0 ? (long)src0->ne[1] : 0,
                src0 ? (long)src0->ne[0] : 0,
                src1 ? (long)src1->ne[1] : 0,
                (src0 && src0->buffer) ? src0->buffer->size : 0,
                res ? "YES" : "NO");
    }
    return res;
}

static bool ggml_backend_xdna_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    (void)dev;
    if (!buft) return false;
    if (buft == &ggml_backend_xdna_buffer_type_instance || buft == &ggml_backend_xdna_host_buffer_type_instance) {
        return true;
    }
    if (buft->iface.is_host && buft->iface.is_host(buft)) {
        return true;
    }
    return false;
}

static bool ggml_backend_xdna_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    (void)dev;
    return supports_op_impl(op);
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_dev_get_name,
    /* .get_description      = */ ggml_backend_xdna_dev_get_description,
    /* .get_memory           = */ ggml_backend_xdna_dev_get_memory,
    /* .get_type             = */ ggml_backend_xdna_dev_get_type,
    /* .get_props            = */ ggml_backend_xdna_dev_get_props,
    /* .init_backend         = */ ggml_backend_xdna_dev_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna_dev_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_xdna_dev_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_dev_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_dev_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_dev_supports_buft,
    /* .offload_op           = */ ggml_backend_xdna_dev_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

struct ggml_backend_device ggml_backend_xdna_device_instance = {
    /* .iface   = */ ggml_backend_xdna_device_i,
    /* .reg     = */ &ggml_backend_xdna_reg_instance,
    /* .context = */ nullptr,
};

// ============================================================================
// 4. GGML Backend Registry Interface (ggml_backend_reg_i)
// ============================================================================

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "XDNA";
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    (void)reg;
    return 1;
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    (void)reg;
    if (index == 0) {
        return &ggml_backend_xdna_device_instance;
    }
    return nullptr;
}

static void * ggml_backend_xdna_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void)reg;
    if (!name) return nullptr;
    if (std::strcmp(name, "ggml_backend_xdna_init") == 0) {
        return reinterpret_cast<void*>(ggml_backend_xdna_init);
    }
    if (std::strcmp(name, "ggml_backend_is_xdna") == 0) {
        return reinterpret_cast<void*>(ggml_backend_is_xdna);
    }
    if (std::strcmp(name, "ggml_backend_xdna_free") == 0) {
        return reinterpret_cast<void*>(ggml_backend_xdna_free);
    }
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_reg_get_proc_address,
};

struct ggml_backend_reg ggml_backend_xdna_reg_instance = {
    /* .api_version = */ GGML_BACKEND_API_VERSION, // 3
    /* .iface       = */ ggml_backend_xdna_reg_i,
    /* .context     = */ nullptr,
};

// ============================================================================
// 5. GGML Backend Stream Interface (ggml_backend_i)
// ============================================================================

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    (void)backend;
    return "XDNA";
}

static void ggml_backend_xdna_free_backend(ggml_backend_t backend) {
    if (!backend) return;
    auto* b = reinterpret_cast<struct ggml_backend*>(backend);
    if (b->guid != nullptr && xdna_guid_matches(b->guid, ggml_backend_xdna_guid())) {
        auto* ctx = reinterpret_cast<XdnaBackendContext*>(b->context);
        delete ctx;
        b->context = nullptr;
    }
}

static void ggml_backend_xdna_synchronize(ggml_backend_t backend) {
    (void)backend;
}

static const struct ggml_backend_i ggml_backend_xdna_i = {
    /* .get_name            = */ ggml_backend_xdna_get_name,
    /* .free                = */ ggml_backend_xdna_free_backend,
    /* .set_tensor_async    = */ nullptr,
    /* .get_tensor_async    = */ nullptr,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ nullptr,
    /* .synchronize         = */ ggml_backend_xdna_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ ggml_backend_xdna_graph_compute,
    /* .event_record        = */ nullptr,
    /* .event_wait          = */ nullptr,
    /* .graph_optimize      = */ nullptr,
};

// ============================================================================
// 6. Official GGML Backend API v3 Entry Points & Backwards-Compatibility Layer
// ============================================================================

extern "C" {

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void) {
    return &ggml_backend_xdna_reg_instance;
}

GGML_BACKEND_API int ggml_backend_score(void) {
    if (std::filesystem::exists("/dev/accel/accel0")) {
        return 100;
    }
    return 100;
}

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    return &ggml_backend_xdna_reg_instance;
}

ggml_backend_t ggml_backend_xdna_init(size_t dev_num) {
    try {
        auto ctx = std::make_unique<XdnaBackendContext>(dev_num);
        auto* backend = static_cast<ggml_backend*>(std::calloc(1, sizeof(struct ggml_backend)));
        backend->guid = ggml_backend_xdna_guid();
        backend->iface = ggml_backend_xdna_i;
        backend->device = &ggml_backend_xdna_device_instance;
        backend->context = ctx.release();
        return backend;
    } catch (...) {
        return nullptr;
    }
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return get_xdna_ctx(backend) != nullptr;
}

void ggml_backend_xdna_free(ggml_backend_t backend) {
    if (!backend) return;
    auto* b = reinterpret_cast<struct ggml_backend*>(backend);
    if (b->guid != nullptr && xdna_guid_matches(b->guid, ggml_backend_xdna_guid())) {
        auto* ctx = reinterpret_cast<XdnaBackendContext*>(b->context);
        delete ctx;
        delete b;
    } else {
        auto* raw = reinterpret_cast<XdnaBackendContext*>(backend);
        if (raw->magic == XDNA_BACKEND_MAGIC) {
            delete raw;
        }
    }
}

ggml_backend_buffer_type_t ggml_backend_xdna_buffer_type(size_t dev_num) {
    (void)dev_num;
    return &ggml_backend_xdna_buffer_type_instance;
}

ggml_backend_buffer_type_t ggml_backend_xdna_host_buffer_type(void) {
    return &ggml_backend_xdna_host_buffer_type_instance;
}

bool ggml_backend_xdna_supports_op(ggml_backend_t backend, const struct ggml_tensor_xdna* op) {
    (void)backend;
    return supports_op_impl(op);
}

enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    if (!backend || !cgraph) return GGML_STATUS_FAILED;
    if (cgraph->n_nodes > 0 && !cgraph->nodes) return GGML_STATUS_FAILED;
    auto* ctx = get_xdna_ctx(backend);
    if (!ctx) return GGML_STATUS_FAILED;

    std::lock_guard<std::mutex> lock(ctx->mtx);

    static const bool trace = (std::getenv("XDNA_TRACE_NODES") != nullptr);
    double total_ms = 0.0;
    int node_count = 0;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (!cgraph->nodes) return GGML_STATUS_FAILED;
        struct ggml_tensor* node = cgraph->nodes[i];
        if (!node) continue;

        if (node->op == GGML_OP_MUL_MAT) {
            std::chrono::high_resolution_clock::time_point t0;
            if (trace) {
                t0 = std::chrono::high_resolution_clock::now();
            }
            if (!process_mul_mat_node(ctx, node)) {
                return GGML_STATUS_FAILED;
            }
            if (trace) {
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_ms += ms;
                node_count++;
                fprintf(stderr, "[xdna trace] node %3d: %-30s | type=%2d | N=%6ld K=%5ld | %6.2f ms\n",
                        i, node->name[0] ? node->name : "unnamed",
                        node->src[0]->type, (long)node->src[0]->ne[1], (long)node->src[0]->ne[0], ms);
            }
        }
    }
    if (trace && node_count > 0) {
        fprintf(stderr, "[xdna trace] TOTAL: %d nodes, %6.2f ms\n", node_count, total_ms);
    }
    return GGML_STATUS_SUCCESS;
}

bool ggml_backend_xdna_compute_graph(ggml_backend_t backend, struct ggml_tensor_xdna** nodes, size_t n_nodes) {
    if (!backend || !nodes || n_nodes == 0) return false;
    auto* ctx = get_xdna_ctx(backend);
    if (!ctx) return false;

    std::lock_guard<std::mutex> lock(ctx->mtx);

    for (size_t i = 0; i < n_nodes; ++i) {
        if (!nodes) return false;
        auto* node = nodes[i];
        if (!node) continue;

        if (node->op == GGML_OP_MUL_MAT) {
            if (!process_mul_mat_node(ctx, node)) {
                return false;
            }
        }
    }
    return true;
}

} // extern "C"
