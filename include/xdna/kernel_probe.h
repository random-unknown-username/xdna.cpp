#pragma once

#include "xdna/device.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace xdna {

struct KernelTileSnapshot {
    std::uint16_t row_count = 0;
    std::uint16_t row_start = 0;
    std::uint16_t dma_channel_count = 0;
    std::uint16_t lock_count = 0;
    std::uint16_t event_reg_count = 0;
};

struct KernelAieSnapshot {
    std::uint32_t column_size = 0;
    Topology physical_topology;
    std::uint32_t aie_major = 0;
    std::uint32_t aie_minor = 0;
    KernelTileSnapshot compute;
    KernelTileSnapshot memory;
    KernelTileSnapshot shim;
};

struct KernelFirmwareSnapshot {
    std::uint32_t major = 0;
    std::uint32_t minor = 0;
    std::uint32_t patch = 0;
    std::uint32_t build = 0;
};

struct KernelProbeError {
    std::string operation;
    std::string parameter;
    int error_number = 0;
    std::string error_message;
};

struct KernelProbe {
    std::filesystem::path accel_node = "/dev/accel/accel0";
    bool device_opened = false;
    bool metadata_query_attempted = false;
    bool firmware_query_attempted = false;
    std::optional<KernelAieSnapshot> aie;
    std::optional<KernelFirmwareSnapshot> firmware;
    std::vector<KernelProbeError> errors;

    [[nodiscard]] bool metadata_query_succeeded() const noexcept {
        return aie.has_value();
    }

    [[nodiscard]] bool firmware_query_succeeded() const noexcept {
        return firmware.has_value();
    }

    [[nodiscard]] bool all_queries_succeeded() const noexcept {
        return metadata_query_succeeded() && firmware_query_succeeded();
    }
};

// Opens only the explicitly supplied accelerator node and issues only
// DRM_IOCTL_AMDXDNA_GET_INFO queries for read-only metadata.
KernelProbe probe_kernel(const std::filesystem::path& accel_node =
                             "/dev/accel/accel0");

std::string format_kernel_probe(const KernelProbe& probe);

}  // namespace xdna
