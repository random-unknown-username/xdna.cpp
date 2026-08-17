#pragma once

// This is the narrow, read-only part of the amdxdna UAPI used by the kernel
// discovery probe.  The fields and ioctl number are shared by the installed
// libdrm header and the running NPU1 driver.  Keeping this small ABI mirror
// lets the probe build without XRT or libdrm development packages.

#include <cstddef>
#include <cstdint>

#if defined(__linux__)
#include <sys/ioctl.h>
#endif

// Optional installed reference header.  It is included outside our
// namespace so its C UAPI declarations retain their expected names.
#if defined(__linux__) && defined(__has_include)
#if __has_include(<drm/amdxdna_accel.h>)
#include <drm/amdxdna_accel.h>
#define XDNA_HAS_INSTALLED_AMDXDNA_UAPI 1
#endif
#endif

namespace xdna::amdxdna_uapi {

struct AieVersion {
    std::uint32_t major;
    std::uint32_t minor;
};

struct AieTileMetadata {
    std::uint16_t row_count;
    std::uint16_t row_start;
    std::uint16_t dma_channel_count;
    std::uint16_t lock_count;
    std::uint16_t event_reg_count;
    std::uint16_t pad[3];
};

struct AieMetadata {
    std::uint32_t column_size;
    std::uint16_t columns;
    std::uint16_t rows;
    AieVersion version;
    AieTileMetadata core;
    AieTileMetadata memory;
    AieTileMetadata shim;
};

struct FirmwareVersion {
    std::uint32_t major;
    std::uint32_t minor;
    std::uint32_t patch;
    std::uint32_t build;
};

struct GetInfo {
    std::uint32_t param;
    std::uint32_t buffer_size;
    std::uint64_t buffer;
};

constexpr std::uint32_t kQueryAieMetadata = 1;
constexpr std::uint32_t kQueryFirmwareVersion = 8;

#if defined(__linux__)
// DRM_COMMAND_BASE is 0x40.  The amdxdna UAPI assigns GET_INFO command 7.
constexpr unsigned long kIoctlGetInfo = _IOWR('d', 0x40U + 7U, GetInfo);
#else
constexpr unsigned long kIoctlGetInfo = 0;
#endif

static_assert(sizeof(AieVersion) == 8, "amdxdna AIE version ABI changed");
static_assert(sizeof(AieTileMetadata) == 16,
              "amdxdna AIE tile metadata ABI changed");
static_assert(sizeof(AieMetadata) == 64, "amdxdna AIE metadata ABI changed");
static_assert(sizeof(FirmwareVersion) == 16,
              "amdxdna firmware version ABI changed");
static_assert(sizeof(GetInfo) == 16, "amdxdna GET_INFO ABI changed");
static_assert(offsetof(AieMetadata, version) == 8,
              "amdxdna AIE metadata version offset changed");
static_assert(offsetof(AieMetadata, core) == 16,
              "amdxdna AIE metadata core offset changed");
static_assert(offsetof(AieMetadata, memory) == 32,
              "amdxdna AIE metadata memory offset changed");
static_assert(offsetof(AieMetadata, shim) == 48,
              "amdxdna AIE metadata shim offset changed");
static_assert(offsetof(GetInfo, buffer) == 8,
              "amdxdna GET_INFO buffer offset changed");

// When the distro header is present, verify this mirror against it.  The
// source checkout may contain newer unrelated UAPI additions; only these
// stable common definitions are intentionally consumed here.
#if defined(XDNA_HAS_INSTALLED_AMDXDNA_UAPI)
static_assert(sizeof(AieMetadata) == sizeof(::amdxdna_drm_query_aie_metadata),
              "amdxdna AIE metadata does not match installed UAPI");
static_assert(sizeof(FirmwareVersion) ==
                  sizeof(::amdxdna_drm_query_firmware_version),
              "amdxdna firmware version does not match installed UAPI");
static_assert(sizeof(GetInfo) == sizeof(::amdxdna_drm_get_info),
              "amdxdna GET_INFO does not match installed UAPI");
static_assert(kIoctlGetInfo == DRM_IOCTL_AMDXDNA_GET_INFO,
              "amdxdna GET_INFO ioctl number does not match installed UAPI");
static_assert(kQueryAieMetadata == DRM_AMDXDNA_QUERY_AIE_METADATA,
              "amdxdna AIE metadata parameter does not match installed UAPI");
static_assert(kQueryFirmwareVersion == DRM_AMDXDNA_QUERY_FIRMWARE_VERSION,
              "amdxdna firmware parameter does not match installed UAPI");
#endif

}  // namespace xdna::amdxdna_uapi
