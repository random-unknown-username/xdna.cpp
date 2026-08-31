#include "xdna/kernel_probe.h"

#include "xdna/amdxdna_uapi.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/ioctl.h>
#include <unistd.h>

namespace xdna {
namespace {

class FileDescriptor {
  public:
    explicit FileDescriptor(int value) : value_(value) {}
    ~FileDescriptor() {
        if (value_ >= 0) {
            (void)::close(value_);
        }
    }

    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

  private:
    int value_;
};

KernelProbeError error_for(const char* operation, const char* parameter,
                           int error_number) {
    KernelProbeError error;
    error.operation = operation;
    error.parameter = parameter;
    error.error_number = error_number;
    error.error_message = std::strerror(error_number);
    return error;
}

KernelTileSnapshot tile_snapshot(
    const amdxdna_uapi::AieTileMetadata& metadata) {
    return KernelTileSnapshot{
        metadata.row_count,
        metadata.row_start,
        metadata.dma_channel_count,
        metadata.lock_count,
        metadata.event_reg_count,
    };
}

std::string format_errno(const KernelProbeError& error) {
    std::ostringstream output;
    if (error.operation == "open") {
        output << "open: ";
    } else {
        output << error.operation << "(" << error.parameter << "): ";
    }
    output << "errno=" << error.error_number << " ("
           << error.error_message << ')';
    return output.str();
}

std::string query_status(bool attempted, bool succeeded) {
    if (!attempted) {
        return "not attempted";
    }
    return succeeded ? "success" : "failure";
}

std::string format_row_range(const KernelTileSnapshot& tile) {
    if (tile.row_count == 0) {
        return "unknown";
    }
    const auto last = static_cast<std::uint32_t>(tile.row_start) +
                      static_cast<std::uint32_t>(tile.row_count) - 1U;
    return std::to_string(tile.row_start) + ".." + std::to_string(last) +
           " (" + std::to_string(tile.row_count) + " rows)";
}

std::string format_firmware(const KernelFirmwareSnapshot& firmware) {
    return std::to_string(firmware.major) + "." +
           std::to_string(firmware.minor) + "." +
           std::to_string(firmware.patch) + "." +
           std::to_string(firmware.build);
}

std::string format_physical_topology(const Topology& topology) {
    if (!topology.valid()) {
        return "unknown";
    }
    return std::to_string(topology.rows) + " rows x " +
           std::to_string(topology.columns) + " columns (" +
           std::to_string(static_cast<std::uint64_t>(topology.rows) *
                          topology.columns) +
           " tile positions)";
}

}  // namespace

KernelProbe probe_kernel(const std::filesystem::path& accel_node) {
    KernelProbe result;
    result.accel_node = accel_node;

#if defined(__linux__)
    const int fd = ::open(accel_node.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        result.errors.push_back(error_for("open", "", errno));
        return result;
    }
    const FileDescriptor descriptor(fd);

    amdxdna_uapi::AieMetadata metadata{};
    amdxdna_uapi::GetInfo metadata_query{
        amdxdna_uapi::kQueryAieMetadata,
        static_cast<std::uint32_t>(sizeof(metadata)),
        reinterpret_cast<std::uint64_t>(&metadata),
    };
    result.metadata_query_attempted = true;
    if (::ioctl(fd, amdxdna_uapi::kIoctlGetInfo, &metadata_query) < 0) {
        result.errors.push_back(error_for(
            "ioctl", "DRM_IOCTL_AMDXDNA_GET_INFO "
                     "param=DRM_AMDXDNA_QUERY_AIE_METADATA",
            errno));
    } else {
        result.aie = KernelAieSnapshot{
            metadata.column_size,
            Topology{metadata.rows, metadata.columns},
            metadata.version.major,
            metadata.version.minor,
            tile_snapshot(metadata.core),
            tile_snapshot(metadata.memory),
            tile_snapshot(metadata.shim),
        };
    }

    amdxdna_uapi::FirmwareVersion firmware{};
    amdxdna_uapi::GetInfo firmware_query{
        amdxdna_uapi::kQueryFirmwareVersion,
        static_cast<std::uint32_t>(sizeof(firmware)),
        reinterpret_cast<std::uint64_t>(&firmware),
    };
    result.firmware_query_attempted = true;
    if (::ioctl(fd, amdxdna_uapi::kIoctlGetInfo, &firmware_query) < 0) {
        result.errors.push_back(error_for(
            "ioctl", "DRM_IOCTL_AMDXDNA_GET_INFO "
                     "param=DRM_AMDXDNA_QUERY_FIRMWARE_VERSION",
            errno));
    } else {
        result.firmware = KernelFirmwareSnapshot{
            firmware.major,
            firmware.minor,
            firmware.patch,
            firmware.build,
        };
    }
    result.device_opened = true;
#else
    result.errors.push_back(
        error_for("kernel probe", "", ENOTSUP));
#endif

    return result;
}

std::string format_kernel_probe(const KernelProbe& probe) {
    std::ostringstream output;
    output << "Kernel amdxdna probe:\n";
    output << "  Accelerator node: " << probe.accel_node.string() << '\n';
    output << "  Device open: " << (probe.device_opened ? "success" : "failure")
           << '\n';
    output << "  GET_INFO AIE metadata: "
           << query_status(probe.metadata_query_attempted,
                           probe.metadata_query_succeeded())
           << '\n';
    if (probe.aie) {
        output << "  AIE version: " << probe.aie->aie_major << '.'
               << probe.aie->aie_minor << '\n';
        output << "  Physical AIE topology: "
               << format_physical_topology(probe.aie->physical_topology)
               << '\n';
        output << "  Compute row range: "
               << format_row_range(probe.aie->compute) << '\n';
        output << "  Memory row range: "
               << format_row_range(probe.aie->memory) << '\n';
        output << "  Shim row range: " << format_row_range(probe.aie->shim)
               << '\n';
        output << "  Column size: " << probe.aie->column_size << " bytes\n";
    }
    output << "  GET_INFO firmware version: "
           << query_status(probe.firmware_query_attempted,
                           probe.firmware_query_succeeded())
           << '\n';
    if (probe.firmware) {
        output << "  Firmware version: " << format_firmware(*probe.firmware)
               << '\n';
    }
    for (const auto& error : probe.errors) {
        output << "  Error: " << format_errno(error) << '\n';
    }
    output << "  Workload execution proof: no (read-only GET_INFO only)\n";
    output << "  Accessible workload topology: unknown (not proven by GET_INFO)\n";
    return output.str();
}

}  // namespace xdna
