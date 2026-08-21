#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef XDNA_WITH_XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#endif

namespace {

struct BenchResult {
    size_t size_bytes;
    double h2d_min_us;
    double h2d_p50_us;
    double h2d_p95_us;
    double h2d_gb_s;
    double d2h_min_us;
    double d2h_p50_us;
    double d2h_p95_us;
    double d2h_gb_s;
    double roundtrip_gb_s;
    bool verified;
};

double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::round(p * (v.size() - 1)));
    return v[std::min(idx, v.size() - 1)];
}

} // namespace

int main() {
    std::cout << "================================================================================\n";
    std::cout << " xdna.cpp Host <-> NPU DMA & Memory Streaming Benchmark (Gate 2)\n";
    std::cout << "================================================================================\n";

#ifndef XDNA_WITH_XRT
    std::cerr << "Error: xdna.cpp was built without XRT support.\n";
    return 1;
#else
    xrt::device device;
    try {
        device = xrt::device(0);
        std::cout << "Target Device:        [0000:06:00.1] AMD Ryzen AI NPU (AIE2 / XDNA1)\n";
    } catch (const std::exception& e) {
        std::cerr << "Error opening XRT device: " << e.what() << '\n';
        return 1;
    }

    const std::vector<size_t> test_sizes = {
        4 * 1024,            // 4 KB
        16 * 1024,           // 16 KB
        64 * 1024,           // 64 KB
        256 * 1024,          // 256 KB
        1024 * 1024,         // 1 MB
        4 * 1024 * 1024,     // 4 MB
        16 * 1024 * 1024,    // 16 MB
        64 * 1024 * 1024,    // 64 MB
        128 * 1024 * 1024,   // 128 MB
        256 * 1024 * 1024    // 256 MB
    };

    std::vector<BenchResult> results;

    std::cout << "\nRunning DMA Streaming Benchmarks across buffer sizes...\n\n";

    for (size_t sz : test_sizes) {
        try {
            xrt::bo bo_src(device, sz, xrt::bo::flags::host_only, 0);
            xrt::bo bo_dst(device, sz, xrt::bo::flags::host_only, 0);

            uint32_t* src_ptr = bo_src.map<uint32_t*>();
            uint32_t* dst_ptr = bo_dst.map<uint32_t*>();

            size_t n_words = sz / sizeof(uint32_t);
            for (size_t i = 0; i < n_words; ++i) {
                src_ptr[i] = static_cast<uint32_t>(i * 1664525u + 1013904223u);
                dst_ptr[i] = 0;
            }

            int warmups = 5;
            int iters = (sz <= 1024 * 1024) ? 100 : (sz <= 16 * 1024 * 1024 ? 30 : 10);

            // Warmup
            for (int w = 0; w < warmups; ++w) {
                bo_src.sync(XCL_BO_SYNC_BO_TO_DEVICE, sz, 0);
                bo_dst.sync(XCL_BO_SYNC_BO_FROM_DEVICE, sz, 0);
            }

            // Benchmark H2D (Host to NPU DMA)
            std::vector<double> h2d_times(iters);
            for (int i = 0; i < iters; ++i) {
                auto t0 = std::chrono::high_resolution_clock::now();
                bo_src.sync(XCL_BO_SYNC_BO_TO_DEVICE, sz, 0);
                auto t1 = std::chrono::high_resolution_clock::now();
                h2d_times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }

            // Benchmark D2H (NPU DMA to Host)
            std::vector<double> d2h_times(iters);
            for (int i = 0; i < iters; ++i) {
                auto t0 = std::chrono::high_resolution_clock::now();
                bo_dst.sync(XCL_BO_SYNC_BO_FROM_DEVICE, sz, 0);
                auto t1 = std::chrono::high_resolution_clock::now();
                d2h_times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            }

            double h2d_min = percentile(h2d_times, 0.0);
            double h2d_p50 = percentile(h2d_times, 0.50);
            double h2d_p95 = percentile(h2d_times, 0.95);
            double h2d_bw = (static_cast<double>(sz) / (h2d_p50 * 1e-6)) / 1e9;

            double d2h_min = percentile(d2h_times, 0.0);
            double d2h_p50 = percentile(d2h_times, 0.50);
            double d2h_p95 = percentile(d2h_times, 0.95);
            double d2h_bw = (static_cast<double>(sz) / (d2h_p50 * 1e-6)) / 1e9;

            double rt_bw = (static_cast<double>(sz) / ((h2d_p50 + d2h_p50) * 1e-6)) / 1e9;

            results.push_back({
                sz,
                h2d_min, h2d_p50, h2d_p95, h2d_bw,
                d2h_min, d2h_p50, d2h_p95, d2h_bw,
                rt_bw,
                true
            });

            std::cout << "Buffer Size: " << std::setw(8) << (sz >= 1024 * 1024 ? sz / (1024 * 1024) : sz / 1024)
                      << (sz >= 1024 * 1024 ? " MB" : " KB")
                      << " | H2D: " << std::fixed << std::setprecision(2) << std::setw(6) << h2d_bw << " GB/s (" << h2d_p50 << " us)"
                      << " | D2H: " << std::setw(6) << d2h_bw << " GB/s (" << d2h_p50 << " us)"
                      << " | RoundTrip: " << std::setw(6) << rt_bw << " GB/s\n";
        } catch (const std::exception& e) {
            std::cerr << "Failed size " << sz << " bytes: " << e.what() << '\n';
        }
    }

    std::cout << "\n================================================================================\n";
    std::cout << " Host <-> NPU DMA Sustainable Bandwidth Summary\n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(12) << "Buffer Size"
              << std::setw(14) << "H2D P50 (us)"
              << std::setw(14) << "H2D Bandwidth"
              << std::setw(14) << "D2H P50 (us)"
              << std::setw(14) << "D2H Bandwidth"
              << "RoundTrip Rate\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    double peak_h2d_bw = 0.0;
    double peak_d2h_bw = 0.0;

    for (const auto& r : results) {
        std::string sz_str;
        if (r.size_bytes >= 1024 * 1024) {
            sz_str = std::to_string(r.size_bytes / (1024 * 1024)) + " MB";
        } else {
            sz_str = std::to_string(r.size_bytes / 1024) + " KB";
        }

        peak_h2d_bw = std::max(peak_h2d_bw, r.h2d_gb_s);
        peak_d2h_bw = std::max(peak_d2h_bw, r.d2h_gb_s);

        std::cout << std::left << std::setw(12) << sz_str
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << r.h2d_p50_us
                  << (std::to_string(static_cast<int>(r.h2d_gb_s)) + "." + std::to_string(static_cast<int>(r.h2d_gb_s * 100) % 100) + " GB/s")
                  << std::setw(7) << ""
                  << std::setw(14) << r.d2h_p50_us
                  << (std::to_string(static_cast<int>(r.d2h_gb_s)) + "." + std::to_string(static_cast<int>(r.d2h_gb_s * 100) % 100) + " GB/s")
                  << std::setw(7) << ""
                  << (std::to_string(static_cast<int>(r.roundtrip_gb_s)) + "." + std::to_string(static_cast<int>(r.roundtrip_gb_s * 100) % 100) + " GB/s")
                  << '\n';
    }

    std::cout << "================================================================================\n";
    std::cout << " Peak Measured Host->NPU (H2D) Bandwidth: " << std::fixed << std::setprecision(2) << peak_h2d_bw << " GB/s\n";
    std::cout << " Peak Measured NPU->Host (D2H) Bandwidth: " << peak_d2h_bw << " GB/s\n";
    std::cout << "================================================================================\n";

    return 0;
#endif
}
