#include "xdna/q4_0_reference.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <functional>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace {

// Persistent Thread Pool to eliminate thread spawn/join overhead
class PersistentWorkerPool {
public:
    explicit PersistentWorkerPool(size_t num_workers) : m_stop(false), m_working(0) {
        for (size_t i = 0; i < num_workers; ++i) {
            m_workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_cv_task.wait(lock, [this]() { return m_stop || !m_tasks.empty(); });
                        if (m_stop && m_tasks.empty()) return;
                        if (!m_tasks.empty()) {
                            task = std::move(m_tasks.back());
                            m_tasks.pop_back();
                            m_working++;
                        }
                    }
                    if (task) {
                        task();
                        {
                            std::unique_lock<std::mutex> lock(m_mutex);
                            m_working--;
                            if (m_tasks.empty() && m_working == 0) {
                                m_cv_done.notify_all();
                            }
                        }
                    }
                }
            });
        }
    }

    ~PersistentWorkerPool() {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv_task.notify_all();
        for (auto& t : m_workers) {
            if (t.joinable()) t.join();
        }
    }

    void parallel_for(size_t total, size_t chunks, const std::function<void(size_t, size_t)>& func) {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            size_t step = (total + chunks - 1) / chunks;
            for (size_t c = 0; c < chunks; ++c) {
                size_t start = c * step;
                size_t end = std::min(start + step, total);
                if (start < total) {
                    m_tasks.push_back([=]() { func(start, end); });
                }
            }
        }
        m_cv_task.notify_all();
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv_done.wait(lock, [this]() { return m_tasks.empty() && m_working == 0; });
        }
    }

private:
    std::vector<std::thread> m_workers;
    std::vector<std::function<void()>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_cv_task;
    std::condition_variable m_cv_done;
    bool m_stop;
    int m_working;
};

// Optimized unrolled AVX2 Q4_0 dot product
inline float dot_q4_0_f32_unrolled(const uint8_t* row_bytes, size_t K, const float* x) {
    const size_t nb = K / 32;
    const auto* blocks = reinterpret_cast<const xdna::q4_0::BlockQ4_0*>(row_bytes);

    __m256 acc0 = _mm256_setzero_ps();

    for (size_t b = 0; b < nb; ++b) {
        float d = xdna::q4_0::fp16_to_float(blocks[b].d);
        __m256 scale = _mm256_set1_ps(d);

        __m128i raw16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blocks[b].qs));
        __m128i low_nibbles = _mm_and_si128(raw16, _mm_set1_epi8(0x0F));
        __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw16, 4), _mm_set1_epi8(0x0F));

        __m128i v0_8 = _mm_sub_epi16(_mm_cvtepi8_epi16(low_nibbles), _mm_set1_epi16(8));
        __m256 v0_8_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v0_8));

        __m128i v8_16 = _mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(low_nibbles, 8)), _mm_set1_epi16(8));
        __m256 v8_16_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v8_16));

        __m128i v16_24 = _mm_sub_epi16(_mm_cvtepi8_epi16(high_nibbles), _mm_set1_epi16(8));
        __m256 v16_24_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v16_24));

        __m128i v24_32 = _mm_sub_epi16(_mm_cvtepi8_epi16(_mm_srli_si128(high_nibbles, 8)), _mm_set1_epi16(8));
        __m256 v24_32_f = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v24_32));

        const float* act_ptr = x + b * 32;
        __m256 x0 = _mm256_loadu_ps(act_ptr + 0);
        __m256 x1 = _mm256_loadu_ps(act_ptr + 8);
        __m256 x2 = _mm256_loadu_ps(act_ptr + 16);
        __m256 x3 = _mm256_loadu_ps(act_ptr + 24);

        __m256 prod0 = _mm256_mul_ps(v0_8_f, x0);
        __m256 prod1 = _mm256_mul_ps(v8_16_f, x1);
        __m256 prod2 = _mm256_mul_ps(v16_24_f, x2);
        __m256 prod3 = _mm256_mul_ps(v24_32_f, x3);

        __m256 sum_blk = _mm256_add_ps(_mm256_add_ps(prod0, prod1), _mm256_add_ps(prod2, prod3));
        acc0 = _mm256_fmadd_ps(sum_blk, scale, acc0);
    }

    __m128 h0 = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    h0 = _mm_add_ps(h0, _mm_movehl_ps(h0, h0));
    h0 = _mm_add_ss(h0, _mm_shuffle_ps(h0, h0, 1));
    return _mm_cvtss_f32(h0);
}

} // namespace

int main() {
    std::cout << "================================================================================\n";
    std::cout << " Testing Persistent Worker Pool & Worker Scaling for Q4_0 GEMV\n";
    std::cout << "================================================================================\n";

    const size_t N = 17408;
    const size_t K = 5120;
    const size_t bytes_per_row = (K / 32) * 18;
    const size_t total_bytes = N * bytes_per_row; // 47.81 MB

    std::vector<uint8_t> weights(total_bytes, 0x55);
    for (size_t r = 0; r < N; ++r) {
        auto* blks = reinterpret_cast<xdna::q4_0::BlockQ4_0*>(weights.data() + r * bytes_per_row);
        for (size_t b = 0; b < K / 32; ++b) {
            blks[b].d = 0x3C00;
        }
    }

    std::vector<float> x(K, 0.5f);
    std::vector<float> y(N, 0.0f);

    // Test across worker counts (Experiment A: Tile / Worker Scaling)
    const std::vector<size_t> worker_counts = {1, 2, 4, 8, 16};

    std::cout << "\nMatrix: " << N << " x " << K << " (" << total_bytes / (1024.0 * 1024.0) << " MB)\n\n";
    std::cout << std::left << std::setw(12) << "Workers"
              << std::setw(16) << "Latency (ms)"
              << std::setw(20) << "Packed BW (GB/s)"
              << "Speedup vs 1 Worker\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    double base_lat = 0.0;

    for (size_t num_w : worker_counts) {
        PersistentWorkerPool pool(num_w);

        // Warmup
        for (int w = 0; w < 5; ++w) {
            pool.parallel_for(N, num_w, [&](size_t start, size_t end) {
                for (size_t r = start; r < end; ++r) {
                    y[r] = dot_q4_0_f32_unrolled(weights.data() + r * bytes_per_row, K, x.data());
                }
            });
        }

        // Benchmark
        int iters = 50;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            pool.parallel_for(N, num_w, [&](size_t start, size_t end) {
                for (size_t r = start; r < end; ++r) {
                    y[r] = dot_q4_0_f32_unrolled(weights.data() + r * bytes_per_row, K, x.data());
                }
            });
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double dur_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double bw = (static_cast<double>(total_bytes) / (dur_ms * 1e-3)) / 1e9;

        if (num_w == 1) base_lat = dur_ms;
        double speedup = base_lat / dur_ms;

        std::cout << std::left << std::setw(12) << num_w
                  << std::fixed << std::setprecision(2)
                  << std::setw(16) << dur_ms
                  << std::setw(20) << bw
                  << std::setprecision(2) << speedup << "x\n";
    }

    std::cout << "================================================================================\n";
    return 0;
}
