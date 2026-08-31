#include "xdna/xdna_gemv_engine.h"
#include "xdna/q4_prepack.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sys/resource.h>
#include <vector>

struct RusageStats {
    long minflt = 0;
    long majflt = 0;
};

RusageStats get_rusage_stats() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return {usage.ru_minflt, usage.ru_majflt};
}

void benchmark_size(xdna::XdnaGemvEngine& engine, size_t target_bytes, size_t K = 5120) {
    // Number of rows N:
    // bytes_per_row = (K / 32) * 18 = 160 * 18 = 2880 bytes
    size_t bytes_per_row = (K / 32) * 18;
    size_t N = (target_bytes + bytes_per_row - 1) / bytes_per_row;
    // Align N to 4 for column partitioning
    N = ((N + 3) / 4) * 4;
    size_t actual_bytes = N * bytes_per_row;

    // Generate deterministic synthetic Q4_0 data
    std::vector<uint8_t> weights(actual_bytes);
    for (size_t i = 0; i < actual_bytes; ++i) {
        weights[i] = static_cast<uint8_t>((i * 137 + 73) & 0xFF);
    }

    std::vector<float> activations(K, 0.01f);
    std::vector<float> output(N, 0.0f);

    // Warmup
    engine.run(weights.data(), actual_bytes, N, K, activations.data(), output.data());

    int iters = (actual_bytes <= 64 * 1024 * 1024) ? 20 : (actual_bytes <= 256 * 1024 * 1024 ? 10 : 5);

    auto ru_before = get_rusage_stats();
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iters; ++it) {
        engine.run(weights.data(), actual_bytes, N, K, activations.data(), output.data());
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto ru_after = get_rusage_stats();

    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double avg_ms = total_ms / iters;
    double gb_s = (static_cast<double>(actual_bytes) / (avg_ms * 1e-3)) / 1e9;

    long delta_minflt = (ru_after.minflt - ru_before.minflt) / iters;
    long delta_majflt = (ru_after.majflt - ru_before.majflt) / iters;

    std::cout << std::left << std::setw(12) << (std::to_string(actual_bytes / (1024 * 1024)) + " MB")
              << std::setw(14) << (std::to_string(N) + "x" + std::to_string(K))
              << std::fixed << std::setprecision(2)
              << std::setw(14) << avg_ms
              << std::setw(16) << gb_s
              << std::setw(10) << "4 / 16"
              << std::setw(14) << delta_minflt
              << std::setw(14) << delta_majflt
              << "\n";
}

int main() {
    std::cout << "================================================================================================\n";
    std::cout << " STANDALONE AIE Q4 GEMV BENCHMARK (OUTSIDE MODEL / RUNTIME)\n";
    std::cout << " Target: AMD XDNA1 NPU (4 Columns x 4 Rows = 16 Tiles), K=5120\n";
    std::cout << "================================================================================================\n";
    std::cout << std::left << std::setw(12) << "Size (MB)"
              << std::setw(14) << "Shape (NxK)"
              << std::setw(14) << "Avg Time (ms)"
              << std::setw(16) << "Bandwidth (GB/s)"
              << std::setw(10) << "Cols/Tiles"
              << std::setw(14) << "Minor Faults"
              << std::setw(14) << "Major Faults"
              << "\n";
    std::cout << "------------------------------------------------------------------------------------------------\n";

    xdna::XdnaGemvEngine engine(0, 4);

    const std::vector<size_t> sizes_mb = {16, 32, 48, 64, 128, 256, 512, 1024};
    for (size_t mb : sizes_mb) {
        benchmark_size(engine, mb * 1024 * 1024);
    }

    std::cout << "================================================================================================\n";
    return 0;
}
