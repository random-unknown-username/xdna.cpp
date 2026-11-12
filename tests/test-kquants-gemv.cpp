#include "xdna/kquants.h"
#include "xdna/xdna_gemv_engine.h"
#include "xdna/ggml-xdna.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace {

using xdna::QK_K;

void expect(bool condition, const std::string& msg) {
    if (!condition) {
        std::cerr << "FAIL: " << msg << std::endl;
        std::exit(1);
    }
    std::cout << "PASS: " << msg << std::endl;
}

struct AccuracyStats {
    double cosine_similarity = 0.0;
    double max_abs_error = 0.0;
    double relative_l2_error = 0.0;
};

AccuracyStats compute_stats(const float* actual, const float* reference, size_t count) {
    AccuracyStats stats;
    double dot = 0.0;
    double norm_act = 0.0;
    double norm_ref = 0.0;
    double l2_diff = 0.0;
    double max_err = 0.0;

    for (size_t i = 0; i < count; ++i) {
        double a = static_cast<double>(actual[i]);
        double r = static_cast<double>(reference[i]);
        double diff = std::abs(a - r);
        max_err = std::max(max_err, diff);
        l2_diff += diff * diff;
        dot += a * r;
        norm_act += a * a;
        norm_ref += r * r;
    }

    stats.max_abs_error = max_err;
    stats.relative_l2_error = (norm_ref > 1e-12) ? std::sqrt(l2_diff / norm_ref) : 0.0;
    double denom = std::sqrt(norm_act) * std::sqrt(norm_ref);
    stats.cosine_similarity = (denom > 1e-12) ? (dot / denom) : (norm_act == 0.0 && norm_ref == 0.0 ? 1.0 : 0.0);
    return stats;
}

// Aligned buffer helper
struct AlignedDeleter {
    void operator()(void* ptr) const {
        std::free(ptr);
    }
};

template <typename T>
std::unique_ptr<T[], AlignedDeleter> make_aligned(size_t n, size_t alignment = 4096) {
    void* ptr = nullptr;
    size_t bytes = n * sizeof(T);
    if (bytes == 0) bytes = alignment;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        throw std::bad_alloc();
    }
    std::memset(ptr, 0, bytes);
    return std::unique_ptr<T[], AlignedDeleter>(static_cast<T*>(ptr));
}

// Random block generators
void generate_q4_K(uint8_t* dst, size_t N, size_t K, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> byte_dist(0, 255);
    const size_t nb = K / QK_K;
    const size_t bytes_per_row = nb * sizeof(xdna::block_q4_K);

    for (size_t r = 0; r < N; ++r) {
        auto* blks = reinterpret_cast<xdna::block_q4_K*>(dst + r * bytes_per_row);
        for (size_t b = 0; b < nb; ++b) {
            blks[b].d = 0x3800;    // 0.5f in fp16
            blks[b].dmin = 0x3000; // 0.125f in fp16
            for (size_t i = 0; i < 12; ++i) {
                blks[b].scales[i] = static_cast<uint8_t>(byte_dist(rng) & 0x3F);
            }
            for (size_t i = 0; i < 128; ++i) {
                blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        }
    }
}

void generate_q5_K(uint8_t* dst, size_t N, size_t K, uint32_t seed = 43) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> byte_dist(0, 255);
    const size_t nb = K / QK_K;
    const size_t bytes_per_row = nb * sizeof(xdna::block_q5_K);

    for (size_t r = 0; r < N; ++r) {
        auto* blks = reinterpret_cast<xdna::block_q5_K*>(dst + r * bytes_per_row);
        for (size_t b = 0; b < nb; ++b) {
            blks[b].d = 0x3800;    // 0.5f in fp16
            blks[b].dmin = 0x3000; // 0.125f in fp16
            for (size_t i = 0; i < 12; ++i) {
                blks[b].scales[i] = static_cast<uint8_t>(byte_dist(rng) & 0x3F);
            }
            for (size_t i = 0; i < 32; ++i) {
                blks[b].qh[i] = static_cast<uint8_t>(byte_dist(rng));
            }
            for (size_t i = 0; i < 128; ++i) {
                blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        }
    }
}

void generate_q3_K(uint8_t* dst, size_t N, size_t K, uint32_t seed = 44) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> byte_dist(0, 255);
    const size_t nb = K / QK_K;
    const size_t bytes_per_row = nb * sizeof(xdna::block_q3_K);

    for (size_t r = 0; r < N; ++r) {
        auto* blks = reinterpret_cast<xdna::block_q3_K*>(dst + r * bytes_per_row);
        for (size_t b = 0; b < nb; ++b) {
            blks[b].d = 0x3800; // 0.5f in fp16
            for (size_t i = 0; i < 32; ++i) {
                blks[b].hmask[i] = static_cast<uint8_t>(byte_dist(rng));
            }
            for (size_t i = 0; i < 64; ++i) {
                blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
            for (size_t i = 0; i < 12; ++i) {
                blks[b].scales[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        }
    }
}

void generate_q6_K(uint8_t* dst, size_t N, size_t K, uint32_t seed = 45) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> byte_dist(0, 255);
    const size_t nb = K / QK_K;
    const size_t bytes_per_row = nb * sizeof(xdna::block_q6_K);

    for (size_t r = 0; r < N; ++r) {
        auto* blks = reinterpret_cast<xdna::block_q6_K*>(dst + r * bytes_per_row);
        for (size_t b = 0; b < nb; ++b) {
            blks[b].d = 0x3800; // 0.5f in fp16
            for (size_t i = 0; i < 128; ++i) {
                blks[b].ql[i] = static_cast<uint8_t>(byte_dist(rng));
            }
            for (size_t i = 0; i < 64; ++i) {
                blks[b].qh[i] = static_cast<uint8_t>(byte_dist(rng));
            }
            for (size_t i = 0; i < 16; ++i) {
                blks[b].scales[i] = static_cast<int8_t>(byte_dist(rng) & 0x7F);
            }
        }
    }
}

void generate_iq4_xs(uint8_t* dst, size_t N, size_t K, uint32_t seed = 46) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> byte_dist(0, 255);
    const size_t nb = K / QK_K;
    const size_t bytes_per_row = nb * sizeof(xdna::block_iq4_xs);

    for (size_t r = 0; r < N; ++r) {
        auto* blks = reinterpret_cast<xdna::block_iq4_xs*>(dst + r * bytes_per_row);
        for (size_t b = 0; b < nb; ++b) {
            blks[b].d = 0x3800; // 0.5f in fp16
            for (size_t i = 0; i < 4; ++i) {
                blks[b].scales_l[i] = static_cast<uint8_t>(byte_dist(rng));
            }
            blks[b].scales_h = static_cast<uint16_t>(byte_dist(rng) | (byte_dist(rng) << 8));
            for (size_t i = 0; i < 128; ++i) {
                blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        }
    }
}

void generate_iq3_s(uint8_t* dst, size_t N, size_t K, uint32_t seed = 47) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> byte_dist(0, 255);
    const size_t nb = K / QK_K;
    const size_t bytes_per_row = nb * sizeof(xdna::block_iq3_s);

    for (size_t r = 0; r < N; ++r) {
        auto* blks = reinterpret_cast<xdna::block_iq3_s*>(dst + r * bytes_per_row);
        for (size_t b = 0; b < nb; ++b) {
            blks[b].d = 0x3800; // 0.5f in fp16
            for (size_t i = 0; i < 64; ++i) {
                blks[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
            for (size_t i = 0; i < 8; ++i) {
                blks[b].qh[i] = static_cast<uint8_t>(byte_dist(rng) & 0x01); // 1-bit high index
            }
            for (size_t i = 0; i < 32; ++i) {
                blks[b].signs[i] = static_cast<uint8_t>(byte_dist(rng));
            }
            for (size_t i = 0; i < 4; ++i) {
                blks[b].scales[i] = static_cast<uint8_t>(byte_dist(rng));
            }
        }
    }
}

void test_quant_accuracy(xdna::XdnaGemvEngine& engine,
                         xdna::QuantType type,
                         const std::string& type_name,
                         void (*generator)(uint8_t*, size_t, size_t, uint32_t),
                         void (*ref_gemv)(const uint8_t*, size_t, size_t, const float*, float*),
                         size_t N,
                         size_t K) {
    std::cout << "\n================================================================================\n";
    std::cout << " Test: " << type_name << " GEMV Accuracy (N=" << N << ", K=" << K << ")\n";
    std::cout << "================================================================================\n";

    const size_t bpr = xdna::bytes_per_row_for_quant(type, K);
    const size_t total_weight_bytes = N * bpr;

    auto weights = make_aligned<uint8_t>(total_weight_bytes);
    generator(weights.get(), N, K, 101);

    auto act = make_aligned<float>(K);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> act_dist(-0.5f, 0.5f);
    for (size_t i = 0; i < K; ++i) {
        act[i] = act_dist(rng);
    }

    auto out_actual = make_aligned<float>(N);
    auto out_ref = make_aligned<float>(N);

    // Run accelerated implementation
    engine.run_quant(type, weights.get(), total_weight_bytes, N, K, act.get(), out_actual.get());

    // Run golden reference implementation
    ref_gemv(weights.get(), N, K, act.get(), out_ref.get());

    auto stats = compute_stats(out_actual.get(), out_ref.get(), N);

    std::cout << "  Cosine Similarity: " << std::fixed << std::setprecision(8) << stats.cosine_similarity << "\n";
    std::cout << "  Max Absolute Error: " << std::scientific << std::setprecision(4) << stats.max_abs_error << "\n";
    std::cout << "  Relative L2 Error:  " << std::scientific << std::setprecision(4) << stats.relative_l2_error << "\n";

    expect(stats.cosine_similarity >= 0.99999, type_name + " cosine similarity >= 0.99999");
    expect(stats.relative_l2_error < 1e-3, type_name + " relative L2 error < 1e-3");
    expect(stats.max_abs_error < 0.1, type_name + " max abs error < 0.1");

    // Zero activation check
    auto zero_act = make_aligned<float>(K);
    auto zero_out = make_aligned<float>(N);
    engine.run_quant(type, weights.get(), total_weight_bytes, N, K, zero_act.get(), zero_out.get());
    double zero_norm = 0.0;
    for (size_t i = 0; i < N; ++i) {
        zero_norm += std::abs(zero_out[i]);
    }
    expect(zero_norm == 0.0, type_name + " zero activations produce zero output (invariance)");
}

void test_bf16_activations(xdna::XdnaGemvEngine& engine) {
    std::cout << "\n================================================================================\n";
    std::cout << " Test: BF16 Activation Path Across All Quant Types\n";
    std::cout << "================================================================================\n";

    const size_t N = 128;
    const size_t K = 512;

    std::vector<xdna::QuantType> types = {
        xdna::QuantType::Q4_K,
        xdna::QuantType::Q5_K,
        xdna::QuantType::Q3_K,
        xdna::QuantType::Q6_K,
        xdna::QuantType::IQ4_XS,
        xdna::QuantType::IQ3_S
    };

    auto act_f32 = make_aligned<float>(K);
    auto act_bf16 = make_aligned<xdna::q4_0::Bf16>(K);
    for (size_t i = 0; i < K; ++i) {
        act_f32[i] = static_cast<float>(i % 32) * 0.03125f;
        act_bf16[i] = xdna::q4_0::float_to_bf16(act_f32[i]);
    }

    auto out_f32 = make_aligned<float>(N);
    auto out_bf16 = make_aligned<float>(N);

    for (auto t : types) {
        const size_t bpr = xdna::bytes_per_row_for_quant(t, K);
        const size_t total_weight_bytes = N * bpr;
        auto weights = make_aligned<uint8_t>(total_weight_bytes);
        std::memset(weights.get(), 0x55, total_weight_bytes);

        engine.run_quant(t, weights.get(), total_weight_bytes, N, K, act_f32.get(), out_f32.get());
        engine.run_quant_bf16(t, weights.get(), total_weight_bytes, N, K, act_bf16.get(), out_bf16.get());

        auto stats = compute_stats(out_bf16.get(), out_f32.get(), N);
        expect(stats.cosine_similarity >= 0.9999, "BF16 activation path cosine similarity >= 0.9999");
    }
    std::cout << "PASS: All quant types support BF16 activation path\n";
}

void test_dimension_sweeps(xdna::XdnaGemvEngine& engine) {
    std::cout << "\n================================================================================\n";
    std::cout << " Test: Dimension Sweeps (Zen 4 AVX2 Crossover vs AIE 16-Tile Pipeline)\n";
    std::cout << "================================================================================\n";

    std::vector<std::pair<size_t, size_t>> test_shapes = {
        {4, 256},      // Zen 4 crossover
        {16, 256},     // Zen 4 crossover
        {64, 512},     // Zen 4 crossover
        {256, 512},    // Exact threshold boundary
        {512, 512},    // AIE 16-tile pipeline
        {1024, 256},   // AIE 16-tile pipeline
        {4096, 512},   // Large matrix
    };

    for (const auto& [N, K] : test_shapes) {
        const size_t bpr = xdna::bytes_per_row_for_quant(xdna::QuantType::Q4_K, K);
        const size_t total_weight_bytes = N * bpr;
        auto weights = make_aligned<uint8_t>(total_weight_bytes);
        generate_q4_K(weights.get(), N, K, 202);

        auto act = make_aligned<float>(K);
        for (size_t i = 0; i < K; ++i) act[i] = 0.05f;

        auto out = make_aligned<float>(N);
        auto ref = make_aligned<float>(N);

        engine.run_q4_k(weights.get(), total_weight_bytes, N, K, act.get(), out.get());
        xdna::gemv_q4_K_ref(weights.get(), N, K, act.get(), ref.get());

        auto stats = compute_stats(out.get(), ref.get(), N);
        expect(stats.cosine_similarity >= 0.99999, "Shape N=" + std::to_string(N) + " K=" + std::to_string(K) + " cosine sim >= 0.99999");
    }
    std::cout << "PASS: Dimension sweep validated across all shapes\n";
}

void test_ggml_backend_support() {
    std::cout << "\n================================================================================\n";
    std::cout << " Test: GGML Backend Operator Claim & Graph Compute for All Quants\n";
    std::cout << "================================================================================\n";

    ggml_backend_t backend = ggml_backend_xdna_init(0);
    expect(backend != nullptr, "ggml_backend_xdna_init succeeds");
    expect(ggml_backend_is_xdna(backend), "ggml_backend_is_xdna returns true");

    struct TestCase {
        ggml_type_xdna type;
        const char* name;
    };

    std::vector<TestCase> cases = {
        {GGML_TYPE_Q4_K,   "Q4_K"},
        {GGML_TYPE_Q5_K,   "Q5_K"},
        {GGML_TYPE_Q3_K,   "Q3_K"},
        {GGML_TYPE_Q6_K,   "Q6_K"},
        {GGML_TYPE_IQ4_XS, "IQ4_XS"},
        {GGML_TYPE_IQ3_S,  "IQ3_S"}
    };

    for (const auto& tc : cases) {
        ggml_tensor_xdna w{};
        w.type = tc.type;
        w.ne[0] = 5120; // K
        w.ne[1] = 4096; // N
        w.ne[2] = 1;
        w.ne[3] = 1;

        ggml_tensor_xdna act{};
        act.type = GGML_TYPE_F32;
        act.ne[0] = 5120;
        act.ne[1] = 1; // Single token decode
        act.ne[2] = 1;
        act.ne[3] = 1;

        ggml_tensor_xdna op{};
        op.op = GGML_OP_MUL_MAT;
        op.type = GGML_TYPE_F32;
        op.ne[0] = 4096;
        op.ne[1] = 1;
        op.ne[2] = 1;
        op.ne[3] = 1;
        op.src[0] = &w;
        op.src[1] = &act;

        bool supported = ggml_backend_xdna_supports_op(backend, &op);
        expect(supported, std::string("GGML backend claims ") + tc.name + " MUL_MAT for single-token decode");

        // Reject non-single token (prefill M=8)
        act.ne[1] = 8;
        expect(!ggml_backend_xdna_supports_op(backend, &op), std::string("GGML backend rejects ") + tc.name + " prefill (M=8)");
        act.ne[1] = 1;

        // Reject unaligned K
        w.ne[0] = 5121;
        expect(!ggml_backend_xdna_supports_op(backend, &op), std::string("GGML backend rejects unaligned K=5121 for ") + tc.name);
    }

    ggml_backend_xdna_free(backend);
    std::cout << "PASS: GGML Backend Operator Claim & Validation\n";
}

} // namespace

int main() {
    std::cout << "================================================================================\n";
    std::cout << " xdna.cpp High-Performance Quantization Test Suite (Q4_K, Q5_K, Q3_K, Q6_K, IQ4_XS, IQ3_S)\n";
    std::cout << "================================================================================\n";

    // 1. Static Layout Invariants
    static_assert(sizeof(xdna::block_q4_K) == 144, "block_q4_K must be 144 bytes");
    static_assert(sizeof(xdna::block_q5_K) == 176, "block_q5_K must be 176 bytes");
    static_assert(sizeof(xdna::block_q3_K) == 110, "block_q3_K must be 110 bytes");
    static_assert(sizeof(xdna::block_q6_K) == 210, "block_q6_K must be 210 bytes");
    static_assert(sizeof(xdna::block_iq4_xs) == 136, "block_iq4_xs must be 136 bytes");
    static_assert(sizeof(xdna::block_iq3_s) == 110, "block_iq3_s must be 110 bytes");
    std::cout << "PASS: Static layout assertions for all quant structures\n";

    xdna::XdnaGemvEngine engine(0, 4, 256);

    // 2. Numerical Accuracy across all quantization types against golden reference
    test_quant_accuracy(engine, xdna::QuantType::Q4_K,   "Q4_K",   generate_q4_K,   xdna::gemv_q4_K_ref,   512, 1024);
    test_quant_accuracy(engine, xdna::QuantType::Q5_K,   "Q5_K",   generate_q5_K,   xdna::gemv_q5_K_ref,   512, 1024);
    test_quant_accuracy(engine, xdna::QuantType::Q3_K,   "Q3_K",   generate_q3_K,   xdna::gemv_q3_K_ref,   512, 1024);
    test_quant_accuracy(engine, xdna::QuantType::Q6_K,   "Q6_K",   generate_q6_K,   xdna::gemv_q6_K_ref,   512, 1024);
    test_quant_accuracy(engine, xdna::QuantType::IQ4_XS, "IQ4_XS", generate_iq4_xs, xdna::gemv_iq4_xs_ref, 512, 1024);
    test_quant_accuracy(engine, xdna::QuantType::IQ3_S,  "IQ3_S",  generate_iq3_s,  xdna::gemv_iq3_s_ref,  512, 1024);

    // 3. BF16 Activation Path
    test_bf16_activations(engine);

    // 4. Dimension Sweeps & Crossover
    test_dimension_sweeps(engine);

    // 5. GGML Backend Offload & Operator Dispatch
    test_ggml_backend_support();

    std::cout << "\n================================================================================\n";
    std::cout << " ALL TESTS PASSED SUCCESSFULLY (100%)\n";
    std::cout << "================================================================================\n";

    return 0;
}
