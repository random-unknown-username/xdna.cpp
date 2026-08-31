#include "xdna/xdna_gemv_engine.h"
#include "xdna/q4_prepack.h"
#include "xdna/cpu_q4_gemv_engine.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifdef XDNA_WITH_XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#endif

int main() {
    std::cout << "================================================================================================\n";
    std::cout << " REAL 27B TENSOR ISOLATION BENCHMARK (blk.0.ffn_gate.weight: 17408 x 5120, 47.81 MB)\n";
    std::cout << "================================================================================================\n";

    const size_t N = 17408;
    const size_t K = 5120;
    const size_t bytes_per_row = (K / 32) * 18;
    const size_t total_packed_bytes = N * bytes_per_row; // 50,135,040 bytes (~47.81 MB)

    std::vector<uint8_t> raw_q4(total_packed_bytes);
    for (size_t i = 0; i < total_packed_bytes; ++i) {
        raw_q4[i] = static_cast<uint8_t>((i * 137 + 73) & 0xFF);
    }

    std::vector<float> act(K, 0.01f);
    std::vector<float> out(N, 0.0f);

    xdna::Q4GemvEngine cpu_engine(4, 4);

    // -------------------------------------------------------------
    // Mode A: Anonymous Prepacked RAM
    // -------------------------------------------------------------
    std::vector<uint8_t> prepacked_anon(total_packed_bytes);
    xdna::Q4Prepacker::pack(raw_q4.data(), N, K, prepacked_anon.data(), 4, 32);

    // Warmup
    cpu_engine.run(raw_q4.data(), total_packed_bytes, N, K, act.data(), out.data());

    int iters = 50;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cpu_engine.run(raw_q4.data(), total_packed_bytes, N, K, act.data(), out.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_a = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gb_a = (static_cast<double>(total_packed_bytes) / (ms_a * 1e-3)) / 1e9;

    // -------------------------------------------------------------
    // Mode B: File-backed XQ4 mmap
    // -------------------------------------------------------------
    const char* tmp_file = "/tmp/test_tensor_blk0_ffngate.xq4";
    int fd = open(tmp_file, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (ftruncate(fd, total_packed_bytes) != 0) {
        std::cerr << "ftruncate failed\n";
    }
    write(fd, prepacked_anon.data(), total_packed_bytes);
    close(fd);

    fd = open(tmp_file, O_RDONLY);
    void* mmap_ptr = mmap(nullptr, total_packed_bytes, PROT_READ, MAP_SHARED, fd, 0);

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        cpu_engine.run(reinterpret_cast<const uint8_t*>(mmap_ptr), total_packed_bytes, N, K, act.data(), out.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double ms_b = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gb_b = (static_cast<double>(total_packed_bytes) / (ms_b * 1e-3)) / 1e9;

    // -------------------------------------------------------------
    // Mode C: Bounded Staging BO Copied from File-backed mmap
    // -------------------------------------------------------------
#ifdef XDNA_WITH_XRT
    xrt::device dev(0);
    xrt::bo staging_bo(dev, total_packed_bytes, xrt::bo::flags::host_only, 0);
    uint8_t* bo_ptr = staging_bo.map<uint8_t*>();

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        std::memcpy(bo_ptr, mmap_ptr, total_packed_bytes);
        staging_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, total_packed_bytes, 0);
        cpu_engine.run(bo_ptr, total_packed_bytes, N, K, act.data(), out.data());
    }
    t1 = std::chrono::high_resolution_clock::now();
    double ms_c = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gb_c = (static_cast<double>(total_packed_bytes) / (ms_c * 1e-3)) / 1e9;
#else
    double ms_c = 0.0, gb_c = 0.0;
#endif

    // -------------------------------------------------------------
    // Mode D: Direct Userptr/Shared BO over mmap
    // -------------------------------------------------------------
#ifdef XDNA_WITH_XRT
    double ms_d = 0.0, gb_d = 0.0;
    try {
        xrt::bo userptr_bo(dev, mmap_ptr, total_packed_bytes, 0);
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            userptr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, total_packed_bytes, 0);
            cpu_engine.run(reinterpret_cast<const uint8_t*>(mmap_ptr), total_packed_bytes, N, K, act.data(), out.data());
        }
        t1 = std::chrono::high_resolution_clock::now();
        ms_d = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        gb_d = (static_cast<double>(total_packed_bytes) / (ms_d * 1e-3)) / 1e9;
    } catch (const std::exception& e) {
        ms_d = -1.0;
        gb_d = 0.0;
    }
#else
    double ms_d = 0.0, gb_d = 0.0;
#endif

    munmap(mmap_ptr, total_packed_bytes);
    close(fd);
    unlink(tmp_file);

    std::cout << std::left << std::setw(36) << "Memory Mode"
              << std::setw(16) << "Avg Time (ms)"
              << std::setw(18) << "Bandwidth (GB/s)"
              << "Relative Overhead\n";
    std::cout << "------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(36) << "Mode A: Anonymous Prepacked RAM"
              << std::fixed << std::setprecision(2)
              << std::setw(16) << ms_a
              << std::setw(18) << gb_a
              << "1.00x (Baseline)\n";
    std::cout << std::left << std::setw(36) << "Mode B: File-backed XQ4 mmap"
              << std::setw(16) << ms_b
              << std::setw(18) << gb_b
              << (std::to_string(ms_b / ms_a).substr(0, 4) + "x") << "\n";
    std::cout << std::left << std::setw(36) << "Mode C: Staging BO Copy + Sync"
              << std::setw(16) << ms_c
              << std::setw(18) << gb_c
              << (std::to_string(ms_c / ms_a).substr(0, 4) + "x") << "\n";
    std::cout << std::left << std::setw(36) << "Mode D: Direct Userptr BO over mmap"
              << std::setw(16) << ms_d
              << std::setw(18) << gb_d
              << (std::to_string(ms_d / ms_a).substr(0, 4) + "x") << "\n";
    std::cout << "================================================================================================\n";

    return 0;
}
