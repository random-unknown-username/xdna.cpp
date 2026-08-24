#include "xdna/device.h"
#include "xdna/kernel_probe.h"
#include "xdna/xrt_probe.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--strict] [--accel-node PATH] [--sysfs-device PATH]\n";
}

std::string cache_directory() {
    if (const char* configured = std::getenv("XDNA_CACHE_DIR");
        configured != nullptr && configured[0] != '\0') {
        return configured;
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
        return std::string(home) + "/.cache/xdna.cpp/kernels";
    }
    return "not configured (set XDNA_CACHE_DIR)";
}

}  // namespace

int main(int argc, char** argv) {
    xdna::DiscoveryPaths paths;
    bool strict = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (argument == "--strict") {
            strict = true;
            continue;
        }
        if (argument == "--accel-node" || argument == "--sysfs-device") {
            if (index + 1 >= argc) {
                std::cerr << argument << " requires a path\n";
                print_usage(argv[0]);
                return 2;
            }
            const std::filesystem::path path = argv[++index];
            if (argument == "--accel-node") {
                paths.accel_node = path;
            } else {
                paths.sysfs_device = path;
            }
            continue;
        }
        std::cerr << "unknown argument: " << argument << '\n';
        print_usage(argv[0]);
        return 2;
    }

    const auto sysfs = xdna::discover_sysfs(paths);
    const auto kernel = xdna::probe_kernel(paths.accel_node);
    const auto xrt = xdna::probe_xrt();
    std::cout << xdna::format_device_report(sysfs, xrt, kernel,
                                             cache_directory());
    if (strict && !xrt.strict_succeeded()) {
        return 1;
    }
    return 0;
}
