#include "xdna/xrt_probe.h"

#include <xrt/experimental/xrt_system.h>
#include <xrt/experimental/xrt_version.h>
#include <xrt/xrt_device.h>

#include <exception>
#include <sstream>
#include <utility>

namespace xdna {

XrtProbe probe_xrt() {
    XrtProbe result;
    result.compiled_with_xrt = true;

    std::ostringstream version;
    version << xrt::version::major() << '.' << xrt::version::minor() << '.'
            << xrt::version::patch();
    result.library_version = version.str();

    try {
        result.device_count = xrt::system::enumerate_devices();
    } catch (const std::exception& exception) {
        result.error = exception.what();
        return result;
    } catch (...) {
        result.error = "unknown exception from XRT device enumeration";
        return result;
    }

    for (std::uint32_t index = 0; index < *result.device_count; ++index) {
        XrtDeviceSnapshot device;
        device.index = index;
        try {
            const xrt::device xrt_device(index);
            device.opened = true;
            result.any_device_opened = true;
            try {
                device.bdf = xrt_device.get_info<xrt::info::device::bdf>();
                device.name = xrt_device.get_info<xrt::info::device::name>();
            } catch (const std::exception& exception) {
                // Keep the library's exact diagnostic text.  Opening the
                // device has already succeeded, even if metadata did not.
                device.error = exception.what();
            } catch (...) {
                device.error = "unknown exception while opening XRT device";
            }
        } catch (const std::exception& exception) {
            device.error = exception.what();
        } catch (...) {
            device.error = "unknown exception while opening XRT device";
        }
        result.devices.push_back(std::move(device));
    }
    return result;
}

}  // namespace xdna
