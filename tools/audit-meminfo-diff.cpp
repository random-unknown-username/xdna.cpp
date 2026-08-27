#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

#ifdef XDNA_WITH_XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#endif

std::unordered_map<std::string, uint64_t> get_meminfo() {
    std::unordered_map<std::string, uint64_t> info;
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        char key[64];
        uint64_t val = 0;
        if (sscanf(line.c_str(), "%63[^:]: %lu", key, &val) == 2) {
            info[key] = val;
        }
    }
    return info;
}

int main() {
    auto m0 = get_meminfo();
#ifdef XDNA_WITH_XRT
    xrt::device dev(0);
    std::vector<std::unique_ptr<xrt::bo>> test_bos;
    for (int i = 0; i < 40; ++i) { // 2000 MB
        auto bo = std::make_unique<xrt::bo>(dev, 50 * 1024 * 1024, xrt::bo::flags::host_only, 0);
        uint8_t* p = bo->map<uint8_t*>();
        for (size_t k = 0; k < 50 * 1024 * 1024; k += 4096) p[k] = 1;
        bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, 50 * 1024 * 1024, 0);
        test_bos.push_back(std::move(bo));
    }
#endif
    auto m1 = get_meminfo();

    std::cout << "--- MEMINFO DELTA AFTER 2000 MB BO ALLOCATION ---\n";
    auto print_diff = [&](const std::string& k) {
        int64_t d = static_cast<int64_t>(m1[k]) - static_cast<int64_t>(m0[k]);
        std::cout << "  " << k << ": " << (d / 1024) << " MB (Before: " << (m0[k] / 1024) << " MB -> After: " << (m1[k] / 1024) << " MB)\n";
    };

    print_diff("MemAvailable");
    print_diff("MemFree");
    print_diff("Shmem");
    print_diff("Unevictable");
    print_diff("Mlocked");
    print_diff("Active(anon)");
    print_diff("Inactive(anon)");
    print_diff("Active(file)");
    print_diff("Inactive(file)");
    return 0;
}
