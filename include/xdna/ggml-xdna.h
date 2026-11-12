#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(GGML_BACKEND_H) || defined(GGML_MAX_DIMS) || defined(GGML_API)
    typedef enum ggml_type ggml_type_xdna;
    typedef enum ggml_op   ggml_op_xdna;
#elif __has_include("ggml.h") && __has_include("ggml-backend.h")
#   include "ggml.h"
#   include "ggml-backend.h"
    typedef enum ggml_type ggml_type_xdna;
    typedef enum ggml_op   ggml_op_xdna;
#else
    // Standalone fallback definitions if upstream GGML headers are not available
    typedef enum {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        GGML_TYPE_COUNT,
    } ggml_type_xdna;

    typedef enum {
        GGML_OP_NONE = 0,
        GGML_OP_DUP,
        GGML_OP_ADD,
        GGML_OP_ADD1,
        GGML_OP_ACC,
        GGML_OP_SUB,
        GGML_OP_MUL,
        GGML_OP_DIV,
        GGML_OP_SQR,
        GGML_OP_SQRT,
        GGML_OP_LOG,
        GGML_OP_SIN,
        GGML_OP_COS,
        GGML_OP_CLAMP,
        GGML_OP_NORM,
        GGML_OP_RMS_NORM,
        GGML_OP_RMS_NORM_BACK,
        GGML_OP_GROUP_NORM,
        GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT_ID,
        GGML_OP_OUT_PROD,
        GGML_OP_SCALE,
        GGML_OP_SET,
        GGML_OP_CPY,
        GGML_OP_CONT,
        GGML_OP_RESHAPE,
        GGML_OP_VIEW,
        GGML_OP_PERMUTE,
        GGML_OP_TRANSPOSE,
        GGML_OP_GET_ROWS,
        GGML_OP_GET_ROWS_BACK,
        GGML_OP_DIAG,
        GGML_OP_DIAG_MASK_INF,
        GGML_OP_DIAG_MASK_ZERO,
        GGML_OP_SOFT_MAX,
        GGML_OP_SOFT_MAX_BACK,
        GGML_OP_ROPE,
        GGML_OP_ROPE_BACK,
        GGML_OP_ALIBI,
        GGML_OP_SILU,
        GGML_OP_COUNT
    } ggml_op_xdna;

    // Opaque backend types
    typedef void* ggml_backend_t;
    typedef void* ggml_backend_buffer_t;
    typedef void* ggml_backend_buffer_type_t;
    typedef void* ggml_backend_reg_t;
    typedef void* ggml_backend_dev_t;

    enum ggml_status {
        GGML_STATUS_ALLOC_FAILED = -2,
        GGML_STATUS_FAILED       = -1,
        GGML_STATUS_SUCCESS      = 0,
        GGML_STATUS_ABORTED      = 1,
    };

    struct ggml_cgraph;
    struct ggml_tensor;
#endif

#ifdef GGML_BACKEND_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BACKEND_BUILD
#            define GGML_BACKEND_API __declspec(dllexport) extern
#        else
#            define GGML_BACKEND_API __declspec(dllimport) extern
#        endif
#    else
#        define GGML_BACKEND_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define GGML_BACKEND_API extern
#endif

#ifdef __cplusplus
extern "C" {
#endif

// GGML Tensor ABI Representation for backwards-compatible test harness
struct ggml_tensor_xdna {
    ggml_type_xdna type;
    int64_t ne[4];     // number of elements
    size_t  nb[4];     // stride in bytes
    uint32_t op;       // ggml_op
    int32_t op_params[16];
    int32_t flags;
    struct ggml_tensor_xdna* src[10];
    void* data;
    char name[64];
    void* extra;
};

// ============================================================================
// Official GGML Backend API v3 Entry Points (Dynamic Loading)
// ============================================================================
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);
GGML_BACKEND_API int                ggml_backend_score(void);

// Official GGML v3 Graph Compute
enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph);

// ============================================================================
// Backwards-Compatibility Layer APIs (Internal C ABI)
// ============================================================================
ggml_backend_t             ggml_backend_xdna_init(size_t dev_num);
bool                       ggml_backend_is_xdna(ggml_backend_t backend);
void                       ggml_backend_xdna_free(ggml_backend_t backend);

ggml_backend_buffer_type_t ggml_backend_xdna_buffer_type(size_t dev_num);
ggml_backend_buffer_type_t ggml_backend_xdna_host_buffer_type(void);

bool                       ggml_backend_xdna_supports_op(ggml_backend_t backend, const struct ggml_tensor_xdna* op);
bool                       ggml_backend_xdna_compute_graph(ggml_backend_t backend, struct ggml_tensor_xdna** nodes, size_t n_nodes);
ggml_backend_reg_t         ggml_backend_xdna_reg(void);

#ifdef __cplusplus
}
#endif
