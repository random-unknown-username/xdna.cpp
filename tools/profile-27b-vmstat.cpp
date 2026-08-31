#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>

std::unordered_map<std::string, uint64_t> get_vmstat() {
    std::unordered_map<std::string, uint64_t> stats;
    std::ifstream f("/proc/vmstat");
    std::string k;
    uint64_t v;
    while (f >> k >> v) {
        stats[k] = v;
    }
    return stats;
}

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

int main(int argc, char** argv) {
    std::string model = (argc > 1) ? argv[1] : "/home/satvik/FORK/Qwen3.8-27B-Q4_0.gguf";
    std::string dev = (argc > 2) ? argv[2] : "XDNA0";
    int n_tokens = (argc > 3) ? std::stoi(argv[3]) : 4;

    std::cout << "========================================================================================================\n";
    std::cout << " LINUX VMSTAT & MEMORY PRESSURE AUDIT: " << model << " on " << dev << " (n=" << n_tokens << ")\n";
    std::cout << "========================================================================================================\n";

    auto vm_before = get_vmstat();
    auto mem_before = get_meminfo();

    std::string cmd = "/home/satvik/FORK/llama.cpp/build/bin/llama-completion -m " + model +
                      " --device " + dev + " -p 'Hi' -n " + std::to_string(n_tokens) + " -no-cnv > /dev/null 2>&1";

    auto t_start = std::chrono::high_resolution_clock::now();
    int ret = std::system(cmd.c_str());
    auto t_end = std::chrono::high_resolution_clock::now();

    auto vm_after = get_vmstat();
    auto mem_after = get_meminfo();

    double wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    auto diff = [&](const std::string& k) -> int64_t {
        uint64_t b = vm_before.count(k) ? vm_before[k] : 0;
        uint64_t a = vm_after.count(k) ? vm_after[k] : 0;
        return static_cast<int64_t>(a - b);
    };

    std::cout << " Exit Code:                     " << ret << "\n";
    std::cout << " Total Run Wall Time:           " << std::fixed << std::setprecision(2) << wall_ms << " ms\n";
    std::cout << " MemAvailable (Before -> After):" << (mem_before["MemAvailable"] / 1024) << " MB -> "
              << (mem_after["MemAvailable"] / 1024) << " MB\n";
    std::cout << " MemFree (Before -> After):     " << (mem_before["MemFree"] / 1024) << " MB -> "
              << (mem_after["MemFree"] / 1024) << " MB\n";
    std::cout << " SwapUsed (Before -> After):    " << ((mem_before["SwapTotal"] - mem_before["SwapFree"]) / 1024) << " MB -> "
              << ((mem_after["SwapTotal"] - mem_after["SwapFree"]) / 1024) << " MB\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";
    std::cout << " Kernel VM Counters (Delta during run):\n";
    std::cout << "   - Minor Page Faults (pgfault):            " << diff("pgfault") << "\n";
    std::cout << "   - Major Page Faults (pgmajfault):          " << diff("pgmajfault") << "\n";
    std::cout << "   - kswapd Page Scans (pgscan_kswapd):       " << diff("pgscan_kswapd") << "\n";
    std::cout << "   - Direct Page Scans (pgscan_direct):       " << diff("pgscan_direct") << "\n";
    std::cout << "   - kswapd Page Steals (pgsteal_kswapd):     " << diff("pgsteal_kswapd") << "\n";
    std::cout << "   - Direct Page Steals (pgsteal_direct):     " << diff("pgsteal_direct") << "\n";
    std::cout << "   - Swap Pages In (pswpin):                  " << diff("pswpin") << "\n";
    std::cout << "   - Swap Pages Out (pswpout):                " << diff("pswpout") << "\n";
    std::cout << "   - Working Set Refaults (workingset_refault):" << diff("workingset_refault") << "\n";
    std::cout << "   - Working Set Activates (workingset_activate):" << diff("workingset_activate") << "\n";
    std::cout << "   - Memory Compaction Stalls (compact_stall): " << diff("compact_stall") << "\n";
    std::cout << "========================================================================================================\n";

    return 0;
}
