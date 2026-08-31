#include "xdna/cpu_q4_gemv_engine.h"
#include "xdna/q4_prepack.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#ifdef XDNA_WITH_XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#endif

struct ShapeInfo {
    size_t N;
    size_t K;
    size_t count;
    size_t bytes_per_tensor;
    double avg_latency_ms = 0.0;
    double effective_gb_s = 0.0;
};

int main() {
    std::cout << "========================================================================================================\n";
    std::cout << " PRODUCTION TENSOR SHAPES & PIPELINE FILL/DRAIN AUDIT (Qwen3.8-27B Inventory)\n";
    std::cout << "========================================================================================================\n";

    // Tensor shape inventory of the 369 XDNA-offloaded Q4_0 matrices in Qwen3.8-27B
    std::vector<ShapeInfo> shapes = {
        {17408, 5120, 130, 17408 * (5120 / 32) * 18}, // ffn_gate, ffn_up (65 layers x 2)
        {10240, 5120, 48,  10240 * (5120 / 32) * 18}, // attn_qkv (48 layers)
        {6144,  5120, 48,  6144  * (5120 / 32) * 18}, // attn_gate (48 layers)
        {1024,  5120, 34,  1024  * (5120 / 32) * 18}, // attn_k, attn_v (17 layers x 2)
        {12288, 5120, 17,  12288 * (5120 / 32) * 18}, // SSM layers
        {5120,  5120, 24,  5120  * (5120 / 32) * 18}, // attn_output
        {248320, 5120, 1,  248320 * (5120 / 32) * 18},// token_embd (1 matrix)
    };

    // Calculate total count and total bytes
    size_t total_tensors = 0;
    size_t total_weight_bytes = 0;
    for (const auto& s : shapes) {
        total_tensors += s.count;
        total_weight_bytes += s.count * s.bytes_per_tensor;
    }

    std::cout << "Offloaded Tensors:  " << total_tensors << " matrices\n";
    std::cout << "Total Weight Bytes: " << (total_weight_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB ("
              << total_weight_bytes << " bytes)\n\n";

    xdna::Q4GemvEngine engine(4, 4);

    std::cout << std::left << std::setw(20) << "Shape (NxK)"
              << std::setw(10) << "Count"
              << std::setw(14) << "Size (MB)"
              << std::setw(16) << "Total MB"
              << std::setw(16) << "Avg Latency (ms)"
              << std::setw(16) << "Bandwidth (GB/s)"
              << "Total Time (ms)\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    double sum_total_time_ms = 0.0;
    std::vector<double> x_bytes, y_ms;

    for (auto& s : shapes) {
        std::vector<uint8_t> w(s.bytes_per_tensor, 0x33);
        std::vector<float> x(s.K, 0.01f);
        std::vector<float> y(s.N, 0.0f);

        // Warmup
        engine.run(w.data(), s.bytes_per_tensor, s.N, s.K, x.data(), y.data());

        int iters = 30;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            engine.run(w.data(), s.bytes_per_tensor, s.N, s.K, x.data(), y.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double gb_s = (static_cast<double>(s.bytes_per_tensor) / (avg_ms * 1e-3)) / 1e9;
        double group_total_ms = avg_ms * s.count;
        sum_total_time_ms += group_total_ms;

        s.avg_latency_ms = avg_ms;
        s.effective_gb_s = gb_s;

        x_bytes.push_back(static_cast<double>(s.bytes_per_tensor) / (1024.0 * 1024.0));
        y_ms.push_back(avg_ms);

        std::cout << std::left << std::setw(20) << (std::to_string(s.N) + "x" + std::to_string(s.K))
                  << std::setw(10) << s.count
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << (s.bytes_per_tensor / (1024.0 * 1024.0))
                  << std::setw(16) << ((s.bytes_per_tensor * s.count) / (1024.0 * 1024.0))
                  << std::setw(16) << avg_ms
                  << std::setw(16) << gb_s
                  << std::setw(16) << group_total_ms << "\n";
    }

    std::cout << "--------------------------------------------------------------------------------------------------------\n";
    std::cout << " SUM OF ALL INDIVIDUAL ISOLATED TENSORS: " << std::fixed << std::setprecision(2)
              << sum_total_time_ms << " ms\n";
    std::cout << " OVERALL EFFECTIVE STREAMING RATE:        " << std::setprecision(2)
              << ((static_cast<double>(total_weight_bytes) / 1e9) / (sum_total_time_ms * 1e-3)) << " GB/s\n";
    std::cout << "========================================================================================================\n\n";

    // -------------------------------------------------------------------------
    // CPU DDR Contention Experiment: Idle vs Heavy Background Memory Traffic
    // -------------------------------------------------------------------------
    std::cout << "========================================================================================================\n";
    std::cout << " MEMORY CONTROLLER CONTENTION EXPERIMENT\n";
    std::cout << "========================================================================================================\n";

    // Test A: AIE sequence with CPU Idle
    double time_idle_ms = 0.0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (const auto& s : shapes) {
            std::vector<uint8_t> w(s.bytes_per_tensor, 0x11);
            std::vector<float> x(s.K, 0.01f);
            std::vector<float> y(s.N, 0.0f);
            for (size_t c = 0; c < s.count; ++c) {
                engine.run(w.data(), s.bytes_per_tensor, s.N, s.K, x.data(), y.data());
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        time_idle_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    double gb_idle = (static_cast<double>(total_weight_bytes) / 1e9) / (time_idle_ms * 1e-3);

    // Test B: AIE sequence with Heavy CPU Memory Load (8 threads doing streaming memory reads)
    std::atomic<bool> stop_load{false};
    std::vector<std::thread> load_threads;
    const size_t buf_size = 64 * 1024 * 1024;
    std::vector<std::vector<uint8_t>> load_buffers(4, std::vector<uint8_t>(buf_size, 0x77));

    for (int t = 0; t < 4; ++t) {
        load_threads.emplace_back([&, t]() {
            volatile uint64_t sum = 0;
            const uint64_t* ptr = reinterpret_cast<const uint64_t*>(load_buffers[t].data());
            size_t num_words = buf_size / sizeof(uint64_t);
            while (!stop_load.load(std::memory_order_relaxed)) {
                for (size_t i = 0; i < num_words; i += 8) {
                    sum += ptr[i];
                }
            }
        });
    }

    double time_contended_ms = 0.0;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (const auto& s : shapes) {
            std::vector<uint8_t> w(s.bytes_per_tensor, 0x11);
            std::vector<float> x(s.K, 0.01f);
            std::vector<float> y(s.N, 0.0f);
            for (size_t c = 0; c < s.count; ++c) {
                engine.run(w.data(), s.bytes_per_tensor, s.N, s.K, x.data(), y.data());
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        time_contended_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    stop_load.store(true);
    for (auto& th : load_threads) th.join();

    double gb_contended = (static_cast<double>(total_weight_bytes) / 1e9) / (time_contended_ms * 1e-3);

    std::cout << "Test A (CPU Memory Idle):        " << std::fixed << std::setprecision(2)
              << time_idle_ms << " ms (" << gb_idle << " GB/s)\n";
    std::cout << "Test B (CPU Memory Contended):   " << std::setprecision(2)
              << time_contended_ms << " ms (" << gb_contended << " GB/s)\n";
    std::cout << "DDR Contention Slowdown:         " << std::setprecision(2)
              << ((time_contended_ms / time_idle_ms - 1.0) * 100.0) << " %\n";
    std::cout << "========================================================================================================\n";

    return 0;
}
