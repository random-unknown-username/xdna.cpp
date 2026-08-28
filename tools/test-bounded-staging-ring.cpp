#include "xdna/cpu_q4_gemv_engine.h"
#include "xdna/q4_prepack.h"
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

void test_ring(size_t ring_buffer_bytes, size_t total_stream_bytes) {
    const size_t K = 5120;
    const size_t bytes_per_row = (K / 32) * 18;

    // Simulate streaming through many tensors (e.g. 1 GB total)
    size_t num_tensors = total_stream_bytes / (48 * 1024 * 1024);
    size_t tensor_bytes = 48 * 1024 * 1024;
    size_t N = tensor_bytes / bytes_per_row;
    N = ((N + 3) / 4) * 4;
    size_t actual_tensor_bytes = N * bytes_per_row;

    std::vector<uint8_t> synthetic_xq4(actual_tensor_bytes, 0x55);
    std::vector<float> act(K, 0.01f);
    std::vector<float> out(N, 0.0f);

    xdna::Q4GemvEngine cpu_engine(4, 4);

#ifdef XDNA_WITH_XRT
    xrt::device dev(0);
    // Double-buffered ring: 2 BOs of ring_buffer_bytes each
    xrt::bo ring_bo[2] = {
        xrt::bo(dev, ring_buffer_bytes, xrt::bo::flags::host_only, 0),
        xrt::bo(dev, ring_buffer_bytes, xrt::bo::flags::host_only, 0)
    };

    uint8_t* map_bo[2] = {
        ring_bo[0].map<uint8_t*>(),
        ring_bo[1].map<uint8_t*>()
    };

    auto t_start = std::chrono::high_resolution_clock::now();

    size_t ring_idx = 0;
    for (size_t t = 0; t < num_tensors; ++t) {
        size_t offset_in_tensor = 0;
        while (offset_in_tensor < actual_tensor_bytes) {
            size_t chunk = std::min(ring_buffer_bytes, actual_tensor_bytes - offset_in_tensor);
            size_t rows_in_chunk = chunk / bytes_per_row;

            std::memcpy(map_bo[ring_idx], synthetic_xq4.data() + offset_in_tensor, chunk);
            ring_bo[ring_idx].sync(XCL_BO_SYNC_BO_TO_DEVICE, chunk, 0);

            cpu_engine.run(map_bo[ring_idx], chunk, rows_in_chunk, K, act.data(), out.data());

            offset_in_tensor += chunk;
            ring_idx = 1 - ring_idx;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double total_streamed_gb = static_cast<double>(num_tensors * actual_tensor_bytes) / 1e9;
    double sustained_gb_s = total_streamed_gb / (total_ms * 1e-3);

    std::cout << "Ring Buffer Size: " << std::setw(6) << (ring_buffer_bytes / (1024 * 1024)) << " MB (x2 Ping-Pong)"
              << " | Streamed: " << std::fixed << std::setprecision(2) << total_streamed_gb << " GB"
              << " | Time: " << std::setw(8) << total_ms << " ms"
              << " | Sustained Bandwidth: " << std::setw(6) << sustained_gb_s << " GB/s\n";
#endif
}

int main() {
    std::cout << "========================================================================================================\n";
    std::cout << " BOUNDED STAGING BO RING EXPERIMENT (Streaming 1 GB across different ring buffer sizes)\n";
    std::cout << "========================================================================================================\n";

    const std::vector<size_t> ring_sizes_mb = {8, 16, 32, 64, 128, 256};
    for (size_t mb : ring_sizes_mb) {
        test_ring(mb * 1024 * 1024, 1024 * 1024 * 1024);
    }

    std::cout << "========================================================================================================\n";
    return 0;
}
