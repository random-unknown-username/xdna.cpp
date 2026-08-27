#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef XDNA_WITH_XRT
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#endif

struct ProcStatus {
    uint64_t vm_rss_kb = 0;
    uint64_t vm_hwm_kb = 0;
    uint64_t vm_lck_kb = 0;
    uint64_t vm_swap_kb = 0;
    uint64_t rss_anon_kb = 0;
    uint64_t rss_file_kb = 0;
};

ProcStatus read_proc_status() {
    ProcStatus status;
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) sscanf(line.c_str(), "VmRSS: %lu", &status.vm_rss_kb);
        else if (line.rfind("VmHWM:", 0) == 0) sscanf(line.c_str(), "VmHWM: %lu", &status.vm_hwm_kb);
        else if (line.rfind("VmLck:", 0) == 0) sscanf(line.c_str(), "VmLck: %lu", &status.vm_lck_kb);
        else if (line.rfind("VmSwap:", 0) == 0) sscanf(line.c_str(), "VmSwap: %lu", &status.vm_swap_kb);
        else if (line.rfind("RssAnon:", 0) == 0) sscanf(line.c_str(), "RssAnon: %lu", &status.rss_anon_kb);
        else if (line.rfind("RssFile:", 0) == 0) sscanf(line.c_str(), "RssFile: %lu", &status.rss_file_kb);
    }
    return status;
}

std::unordered_map<std::string, uint64_t> read_vmstat() {
    std::unordered_map<std::string, uint64_t> stats;
    std::ifstream f("/proc/vmstat");
    std::string key;
    uint64_t val;
    while (f >> key >> val) {
        stats[key] = val;
    }
    return stats;
}

std::unordered_map<std::string, uint64_t> read_meminfo() {
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

void print_stage(const std::string& stage_name) {
    auto status = read_proc_status();
    auto mem = read_meminfo();
    std::cout << std::left << std::setw(30) << stage_name
              << " | VmRSS: " << std::setw(8) << (status.vm_rss_kb / 1024) << " MB"
              << " | RssAnon: " << std::setw(8) << (status.rss_anon_kb / 1024) << " MB"
              << " | RssFile: " << std::setw(8) << (status.rss_file_kb / 1024) << " MB"
              << " | VmLck: " << std::setw(8) << (status.vm_lck_kb / 1024) << " MB"
              << " | VmSwap: " << std::setw(8) << (status.vm_swap_kb / 1024) << " MB"
              << " | MemAvail: " << std::setw(8) << (mem["MemAvailable"] / 1024) << " MB\n";
}

int main() {
    std::cout << "========================================================================================================\n";
    std::cout << " XRT BO PINNING & SYSTEM MEMORY RESIDENCY AUDIT\n";
    std::cout << "========================================================================================================\n";

    print_stage("1. Process Start (Baseline)");

#ifdef XDNA_WITH_XRT
    xrt::device dev(0);
    print_stage("2. XRT Device Opened");

    std::vector<std::unique_ptr<xrt::bo>> test_bos;
    const size_t bo_chunk_bytes = 48 * 1024 * 1024; // ~48 MB per tensor

    for (int step = 1; step <= 10; ++step) {
        // Allocate 10 chunks of 48 MB each (~480 MB per step)
        for (int c = 0; c < 10; ++c) {
            auto bo = std::make_unique<xrt::bo>(dev, bo_chunk_bytes, xrt::bo::flags::host_only, 0);
            uint8_t* ptr = bo->map<uint8_t*>();
            // Touch all pages
            for (size_t p = 0; p < bo_chunk_bytes; p += 4096) {
                ptr[p] = static_cast<uint8_t>(step);
            }
            bo->sync(XCL_BO_SYNC_BO_TO_DEVICE, bo_chunk_bytes, 0);
            test_bos.push_back(std::move(bo));
        }

        std::string label = "3. BO Alloc (" + std::to_string(step * 480) + " MB Total BOs)";
        print_stage(label);
    }
#endif

    std::cout << "========================================================================================================\n";
    return 0;
}
