#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace xdna {

enum class Architecture {
    unknown,
    xdna1_aie2,
};

std::string_view architecture_name(Architecture architecture);

struct Topology {
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint64_t compute_tiles() const noexcept;
};

// Topology text is interpreted as rows x columns. Named forms such as
// "rows=4 columns=5" and "4 rows x 5 columns" are accepted too.
std::optional<Topology> parse_topology(std::string_view text);
std::string format_topology(const Topology& topology);

// Parse the line-oriented KEY=VALUE format used by PCI/sysfs uevent files.
std::map<std::string, std::string> parse_key_value_lines(std::string_view text);

Architecture classify_architecture(std::string_view vbnv,
                                   std::string_view pci_device_id = {});

struct DiscoveryPaths {
    std::filesystem::path accel_node = "/dev/accel/accel0";
    std::filesystem::path sysfs_device = "/sys/class/accel/accel0/device";
};

struct SysfsSnapshot {
    DiscoveryPaths paths;
    bool accel_node_present = false;
    bool sysfs_device_present = false;

    std::string pci_bdf;
    std::string vendor_id;
    std::string device_id;
    std::string subsystem_vendor_id;
    std::string subsystem_device_id;
    std::string driver;
    std::string vbnv;
    std::string firmware_version;
    std::map<std::string, std::string> uevent;

    Architecture architecture = Architecture::unknown;
    // This is the physical array shape documented by the inspected driver,
    // not the accessible shape of a workload partition.
    std::optional<Topology> physical_topology;
    std::string physical_topology_basis;
};

SysfsSnapshot discover_sysfs(const DiscoveryPaths& paths = {});

struct XrtProbe;
struct KernelProbe;
std::string format_device_report(const SysfsSnapshot& sysfs,
                                 const XrtProbe& xrt,
                                 const KernelProbe& kernel,
                                 std::string_view cache_directory);

// Compatibility overload for callers that only have the original discovery
// and XRT snapshots.  It does not run a kernel probe.
std::string format_device_report(const SysfsSnapshot& sysfs,
                                 const XrtProbe& xrt,
                                 std::string_view cache_directory);

}  // namespace xdna
