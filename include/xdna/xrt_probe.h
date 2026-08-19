#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xdna {

struct XrtDeviceSnapshot {
    std::uint32_t index = 0;
    bool opened = false;
    std::string bdf;
    std::string name;
    std::string error;
};

struct XrtProbe {
    bool compiled_with_xrt = false;
    std::string library_version;
    std::optional<std::uint32_t> device_count;
    bool any_device_opened = false;
    std::string error;
    std::vector<XrtDeviceSnapshot> devices;

    [[nodiscard]] bool api_probe_succeeded() const noexcept {
        return device_count.has_value();
    }

    [[nodiscard]] bool strict_succeeded() const noexcept {
        // Strict mode requires enumeration to complete and at least one
        // enumerated device constructor to succeed.  Per-device metadata
        // errors remain visible in the snapshot and are not rewritten here.
        return api_probe_succeeded() && any_device_opened;
    }
};

XrtProbe probe_xrt();

}  // namespace xdna
