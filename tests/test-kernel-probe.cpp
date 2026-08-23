#include "xdna/kernel_probe.h"

#include <cstdlib>
#include <iostream>

namespace {

void expect(bool condition, const char* expression) {
    if (!condition) {
        std::cerr << "FAIL: " << expression << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    xdna::KernelProbe fixture;
    fixture.accel_node = "/fixture/accel0";
    fixture.device_opened = true;
    fixture.metadata_query_attempted = true;
    fixture.firmware_query_attempted = true;
    fixture.aie = xdna::KernelAieSnapshot{
        0x100000,
        xdna::Topology{6, 5},
        1,
        1,
        xdna::KernelTileSnapshot{4, 2, 2, 16, 16},
        xdna::KernelTileSnapshot{1, 1, 2, 16, 16},
        xdna::KernelTileSnapshot{1, 0, 8, 16, 16},
    };
    fixture.firmware = xdna::KernelFirmwareSnapshot{1, 5, 5, 391};

    const std::string formatted = xdna::format_kernel_probe(fixture);
    expect(formatted.find("Device open: success") != std::string::npos,
           "format successful open");
    expect(formatted.find("AIE version: 1.1") != std::string::npos,
           "format AIE version");
    expect(formatted.find("Physical AIE topology: 6 rows x 5 columns (30 tile positions)") !=
               std::string::npos,
           "format physical topology");
    expect(formatted.find("Compute row range: 2..5 (4 rows)") !=
               std::string::npos,
           "format compute row range");
    expect(formatted.find("Firmware version: 1.5.5.391") != std::string::npos,
           "format firmware version");
    expect(formatted.find("Workload execution proof: no (read-only GET_INFO only)") !=
               std::string::npos,
           "do not claim execution");

    xdna::KernelProbe error_fixture;
    error_fixture.accel_node = "/fixture/accel0";
    error_fixture.device_opened = true;
    error_fixture.metadata_query_attempted = true;
    error_fixture.errors.push_back({
        "ioctl",
        "DRM_IOCTL_AMDXDNA_GET_INFO param=DRM_AMDXDNA_QUERY_AIE_METADATA",
        11,
        "Resource temporarily unavailable",
    });
    const std::string error_text = xdna::format_kernel_probe(error_fixture);
    expect(error_text.find(
               "Error: ioctl(DRM_IOCTL_AMDXDNA_GET_INFO param=DRM_AMDXDNA_QUERY_AIE_METADATA): errno=11 (Resource temporarily unavailable)") !=
               std::string::npos,
           "format exact ioctl error");
    expect(error_text.find("GET_INFO AIE metadata: failure") != std::string::npos,
           "format failed query status");

    std::cout << "xdna kernel probe formatting tests: PASS\n";
    return 0;
}
