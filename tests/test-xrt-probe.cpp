#include "xdna/device.h"
#include "xdna/kernel_probe.h"
#include "xdna/xrt_probe.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void expect(bool condition, const char* expression) {
    if (!condition) {
        std::cerr << "FAIL: " << expression << '\n';
        std::exit(1);
    }
}

xdna::SysfsSnapshot fixture_sysfs() {
    xdna::SysfsSnapshot sysfs;
    sysfs.paths.accel_node = "/fixture/accel0";
    sysfs.paths.sysfs_device = "/fixture/device";
    sysfs.accel_node_present = true;
    sysfs.sysfs_device_present = true;
    sysfs.pci_bdf = "0000:06:00.1";
    sysfs.driver = "amdxdna";
    sysfs.vbnv = "RyzenAI-npu1";
    sysfs.architecture = xdna::Architecture::xdna1_aie2;
    return sysfs;
}

}  // namespace

int main() {
    xdna::KernelProbe kernel;
    kernel.accel_node = "/fixture/accel0";

    xdna::XrtProbe opened;
    opened.compiled_with_xrt = true;
    opened.library_version = "2.26.0";
    opened.device_count = 1;
    opened.any_device_opened = true;
    xdna::XrtDeviceSnapshot opened_device;
    opened_device.index = 0;
    opened_device.opened = true;
    opened_device.bdf = "0000:06:00.1";
    opened_device.name = "AMD Ryzen AI NPU";
    opened.devices.push_back(opened_device);

    expect(opened.api_probe_succeeded(), "enumeration success fixture");
    expect(opened.strict_succeeded(), "opened device passes strict status");
    const std::string opened_text = xdna::format_device_report(
        fixture_sysfs(), opened, kernel, "/fixture/cache");
    expect(opened_text.find("At least one device opened: yes") !=
               std::string::npos,
           "format at least one opened");
    expect(opened_text.find("Strict status: pass") != std::string::npos,
           "format strict pass");
    expect(opened_text.find("Device 0: opened=yes") != std::string::npos,
           "format per-device opened result");

    xdna::XrtProbe zero_devices;
    zero_devices.compiled_with_xrt = true;
    zero_devices.library_version = "2.26.0";
    zero_devices.device_count = 0;
    expect(!zero_devices.strict_succeeded(),
           "zero devices fail strict status");
    const std::string zero_text = xdna::format_device_report(
        fixture_sysfs(), zero_devices, kernel, "/fixture/cache");
    expect(zero_text.find("API probe: success") != std::string::npos,
           "zero devices retain enumeration API success");
    expect(zero_text.find("At least one device opened: no") !=
               std::string::npos,
           "zero devices format no opened device");
    expect(zero_text.find(
               "Strict status: failure (XRT enumeration returned zero devices)") !=
               std::string::npos,
           "zero devices format strict failure");
    expect(zero_text.find(
               "Device opening: not attempted (XRT enumeration returned zero)") !=
               std::string::npos,
           "zero devices format opening not attempted");

    const std::string exact_open_error =
        "mmap(addr=0x7f5800000000, len=67108864, prot=3, flags=8209, "
        "offset=4294967296) failed (err=-11): Resource temporarily unavailable";
    xdna::XrtProbe open_failed;
    open_failed.compiled_with_xrt = true;
    open_failed.library_version = "2.26.0";
    open_failed.device_count = 1;
    xdna::XrtDeviceSnapshot failed_device;
    failed_device.index = 0;
    failed_device.error = exact_open_error;
    open_failed.devices.push_back(failed_device);
    expect(!open_failed.strict_succeeded(), "open failure fails strict status");
    const std::string failed_text = xdna::format_device_report(
        fixture_sysfs(), open_failed, kernel, "/fixture/cache");
    expect(failed_text.find("Strict status: failure (no enumerated XRT device was opened)") !=
               std::string::npos,
           "open failure format strict failure");
    expect(failed_text.find("Device 0: opened=no") != std::string::npos,
           "format failed per-device open");
    expect(failed_text.find("detail=" + exact_open_error) != std::string::npos,
           "preserve exact XRT error text");

    xdna::XrtProbe enumeration_failed;
    enumeration_failed.compiled_with_xrt = true;
    enumeration_failed.error = exact_open_error;
    expect(!enumeration_failed.strict_succeeded(),
           "enumeration failure fails strict status");
    const std::string enumeration_text = xdna::format_device_report(
        fixture_sysfs(), enumeration_failed, kernel, "/fixture/cache");
    expect(enumeration_text.find("API probe: failure") != std::string::npos,
           "enumeration failure format");
    expect(enumeration_text.find("Probe error: " + exact_open_error) !=
               std::string::npos,
           "preserve exact enumeration error text");

    xdna::XrtProbe unavailable;
    unavailable.error = "xdna.cpp was built without XRT support";
    const std::string unavailable_text = xdna::format_device_report(
        fixture_sysfs(), unavailable, kernel, "/fixture/cache");
    expect(unavailable_text.find("API probe: unavailable") !=
               std::string::npos,
           "format unavailable XRT");
    expect(unavailable_text.find(
               "Strict status: failure (XRT support was not compiled)") !=
               std::string::npos,
           "unavailable XRT fails strict status");

    std::cout << "xdna XRT probe formatting tests: PASS\n";
    return 0;
}
