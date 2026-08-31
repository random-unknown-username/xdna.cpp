#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#ifdef XDNA_WITH_XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#endif

void print_fdinfo() {
    pid_t pid = getpid();
    std::string fdinfo_dir = "/proc/" + std::to_string(pid) + "/fdinfo/";
    std::cout << "--- DRM & AMDXDNA FDINFO (PID " << pid << ") ---\n";
    for (const auto& entry : std::filesystem::directory_iterator(fdinfo_dir)) {
        std::ifstream f(entry.path());
        std::string line;
        bool printed_header = false;
        while (std::getline(f, line)) {
            if (line.find("drm-") != std::string::npos || line.find("amdxdna") != std::string::npos || line.find("memory") != std::string::npos) {
                if (!printed_header) {
                    std::cout << "File: " << entry.path().filename() << "\n";
                    printed_header = true;
                }
                std::cout << "  " << line << "\n";
            }
        }
    }
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << " DRM FDINFO & DRIVER MEMORY ACCOUNTING PROBE\n";
    std::cout << "================================================================================\n";

#ifdef XDNA_WITH_XRT
    xrt::device dev(0);
    std::vector<std::unique_ptr<xrt::bo>> test_bos;
    // Allocate 1 GB of BOs
    for (int i = 0; i < 20; ++i) {
        auto bo = std::make_unique<xrt::bo>(dev, 50 * 1024 * 1024, xrt::bo::flags::host_only, 0);
        uint8_t* p = bo->map<uint8_t*>();
        p[0] = 1;
        bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, 50 * 1024 * 1024, 0);
        test_bos.push_back(std::move(bo));
    }
    std::cout << "Allocated 1000 MB across 20 BOs\n";
    print_fdinfo();
#endif

    std::cout << "================================================================================\n";
    return 0;
}
