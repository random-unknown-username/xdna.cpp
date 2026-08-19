#include "xdna/gate1.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifdef XDNA_WITH_XRT
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#endif

namespace {

using xdna::Gate1State;
using xdna::Gate1StateResult;
using xdna::Gate1Status;

std::vector<Gate1Status> initial_statuses() {
    return {
        {Gate1State::image_found, Gate1StateResult::not_applicable,
         "not attempted"},
        {Gate1State::target_validated, Gate1StateResult::not_applicable,
         "not attempted"},
        {Gate1State::device_opened, Gate1StateResult::not_applicable,
         "not attempted"},
        {Gate1State::image_loaded, Gate1StateResult::not_applicable,
         "not attempted"},
        {Gate1State::buffers_created, Gate1StateResult::not_applicable,
         "not attempted"},
        {Gate1State::dispatch_submitted, Gate1StateResult::not_applicable,
         "not attempted"},
        {Gate1State::completion_observed, Gate1StateResult::not_applicable,
         "not attempted"},
        {Gate1State::output_reference_verified, Gate1StateResult::not_applicable,
         "not attempted"},
    };
}

Gate1Status& status_for(std::vector<Gate1Status>& statuses, Gate1State state) {
    for (auto& status : statuses) {
        if (status.state == state) {
            return status;
        }
    }
    throw std::logic_error("missing Gate-1 state");
}

void set_status(std::vector<Gate1Status>& statuses, Gate1State state,
                Gate1StateResult result, std::string reason) {
    auto& status = status_for(statuses, state);
    status.result = result;
    status.reason = std::move(reason);
}

bool selected_file_exists(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error;
}

bool has_failed_state(const std::vector<Gate1Status>& statuses) {
    for (const auto& status : statuses) {
        if (status.result == Gate1StateResult::fail) {
            return true;
        }
    }
    return false;
}

void mark_unreached(std::vector<Gate1Status>& statuses, Gate1State first,
                    std::string_view reason) {
    bool after = false;
    for (auto& status : statuses) {
        if (status.state == first) {
            after = true;
            continue;
        }
        if (after && status.result == Gate1StateResult::not_applicable &&
            status.state != Gate1State::output_reference_verified) {
            status.reason = std::string("not attempted: ") + std::string(reason);
        }
    }
}

int run_gate1(const xdna::Gate1Options& options) {
    auto statuses = initial_statuses();
    std::cout << "xdna.cpp Gate-1 real-device harness\n"
              << "Selected xclbin: " << options.xclbin.string() << '\n'
              << "Selected ELF: "
              << (options.elf.empty() ? "not supplied" : options.elf.string()) << '\n'
              << "Device index: " << options.device_index << '\n'
              << "Kernel name: " << options.kernel << "\n\n";

    if (!selected_file_exists(options.xclbin)) {
        set_status(statuses, Gate1State::image_found, Gate1StateResult::fail,
                   "selected xclbin is missing or is not a regular file: " +
                       options.xclbin.string());
        mark_unreached(statuses, Gate1State::image_found,
                       "image_found did not pass");
        set_status(statuses, Gate1State::output_reference_verified,
                   Gate1StateResult::not_applicable,
                   "no documented input/output contract supplied; no TCT/NOP "
                   "vector-add correctness invoked");
        std::cout << xdna::format_gate1_status(statuses);
        return 1;
    }
    set_status(statuses, Gate1State::image_found, Gate1StateResult::pass,
               "selected xclbin exists: " + options.xclbin.string());

    const auto metadata = xdna::read_gate1_metadata(options.xclbin);
    std::cout << xdna::format_gate1_metadata(metadata) << '\n';
    if (!metadata.target_validated) {
        set_status(statuses, Gate1State::target_validated,
                   Gate1StateResult::fail,
                   metadata.error.empty()
                       ? "xclbin metadata target validation failed"
                       : metadata.error);
        mark_unreached(statuses, Gate1State::target_validated,
                       "target_validated did not pass");
        set_status(statuses, Gate1State::output_reference_verified,
                   Gate1StateResult::not_applicable,
                   "no documented input/output contract supplied; no TCT/NOP "
                   "vector-add correctness invoked");
        std::cout << xdna::format_gate1_status(statuses);
        return 1;
    }
    set_status(
        statuses, Gate1State::target_validated, Gate1StateResult::pass,
        "actual xclbin metadata matches the Phoenix/NPU1/AIE2 profile; "
        "filename was not used for validation");

#ifndef XDNA_WITH_XRT
    set_status(statuses, Gate1State::device_opened,
               Gate1StateResult::not_applicable,
               "XRT support was disabled at build time; no real-device API "
               "was called");
    mark_unreached(statuses, Gate1State::device_opened,
                   "device_opened is not applicable without XRT");
#else
    std::unique_ptr<xrt::device> device;
    try {
        // This constructor is the real XRT device-open operation.  Preserve
        // exception text unchanged because memlock failures are actionable
        // evidence for this gate.
        device = std::make_unique<xrt::device>(options.device_index);
        set_status(statuses, Gate1State::device_opened, Gate1StateResult::pass,
                   "xrt::device(index) completed");
    } catch (const std::exception& exception) {
        set_status(statuses, Gate1State::device_opened, Gate1StateResult::fail,
                   exception.what());
        mark_unreached(statuses, Gate1State::device_opened,
                       "device_opened did not pass");
    } catch (...) {
        set_status(statuses, Gate1State::device_opened, Gate1StateResult::fail,
                   "unknown exception from xrt::device(index)");
        mark_unreached(statuses, Gate1State::device_opened,
                       "device_opened did not pass");
    }

    if (device) {
        if (!metadata.real_hardware_image) {
            set_status(
                statuses, Gate1State::image_loaded,
                Gate1StateResult::not_applicable,
                "selected xclbin metadata mode is " + metadata.xclbin_mode_name +
                    "; real-device Gate-1 refuses emulation images and will not "
                    "call xrt::device::load_xclbin");
            mark_unreached(statuses, Gate1State::image_loaded,
                           "real-device image load was refused");
        } else {
            try {
                xrt::xclbin image(options.xclbin.string());
                // XRT 2.26 documents this API as deprecated in favor of a
                // context flow, but it is the installed public API that
                // actually loads the complete xclbin image.  The result is
                // only marked after this call returns.
                device->load_xclbin(image);
                set_status(statuses, Gate1State::image_loaded,
                           Gate1StateResult::pass,
                           "xrt::device::load_xclbin completed");

                try {
                    xrt::bo scratch{*device, 4096, xrt::bo::flags::host_only, 0};
                    set_status(statuses, Gate1State::buffers_created,
                               Gate1StateResult::pass,
                               "created one 4096-byte XRT host-only scratch BO; "
                               "it is not passed as a kernel argument");

                    if (options.elf.empty()) {
                        set_status(statuses, Gate1State::dispatch_submitted,
                                   Gate1StateResult::not_applicable,
                                   "no ELF supplied; no dispatch was attempted");
                        mark_unreached(statuses, Gate1State::dispatch_submitted,
                                       "no ELF was supplied");
                    } else {
                        try {
                            xrt::elf elf(options.elf.string());
                            xrt::hw_context hwctx{*device, elf};
                            xrt::kernel kernel =
                                xrt::ext::kernel{hwctx, options.kernel};
                            xrt::run run{kernel};
                            run.start();
                            set_status(statuses, Gate1State::dispatch_submitted,
                                       Gate1StateResult::pass,
                                       "xrt::run::start submitted one ELF-backed "
                                       "kernel run");
                            const auto state = run.wait(options.timeout_ms);
                            if (state == ERT_CMD_STATE_COMPLETED) {
                                set_status(
                                    statuses, Gate1State::completion_observed,
                                    Gate1StateResult::pass,
                                    "xrt::run::wait returned "
                                    "ERT_CMD_STATE_COMPLETED");
                            } else {
                                set_status(
                                    statuses, Gate1State::completion_observed,
                                    Gate1StateResult::fail,
                                    "xrt::run::wait returned command state " +
                                        std::to_string(static_cast<int>(state)));
                            }
                        } catch (const std::exception& exception) {
                            set_status(statuses, Gate1State::dispatch_submitted,
                                       Gate1StateResult::fail, exception.what());
                            mark_unreached(statuses,
                                           Gate1State::dispatch_submitted,
                                           "dispatch setup/submission failed");
                        } catch (...) {
                            set_status(
                                statuses, Gate1State::dispatch_submitted,
                                Gate1StateResult::fail,
                                "unknown exception during ELF context/kernel/run setup");
                            mark_unreached(statuses,
                                           Gate1State::dispatch_submitted,
                                           "dispatch setup/submission failed");
                        }
                    }
                } catch (const std::exception& exception) {
                    set_status(statuses, Gate1State::buffers_created,
                               Gate1StateResult::fail, exception.what());
                    mark_unreached(statuses, Gate1State::buffers_created,
                                   "buffers_created did not pass");
                } catch (...) {
                    set_status(statuses, Gate1State::buffers_created,
                               Gate1StateResult::fail,
                               "unknown exception while creating XRT BO");
                    mark_unreached(statuses, Gate1State::buffers_created,
                                   "buffers_created did not pass");
                }
            } catch (const std::exception& exception) {
                set_status(statuses, Gate1State::image_loaded,
                           Gate1StateResult::fail, exception.what());
                mark_unreached(statuses, Gate1State::image_loaded,
                               "image_loaded did not pass");
            } catch (...) {
                set_status(statuses, Gate1State::image_loaded,
                           Gate1StateResult::fail,
                           "unknown exception from xrt::device::load_xclbin");
                mark_unreached(statuses, Gate1State::image_loaded,
                               "image_loaded did not pass");
            }
        }
    }
#endif

    set_status(statuses, Gate1State::output_reference_verified,
               Gate1StateResult::not_applicable,
               "no documented input/output contract supplied; no TCT/NOP "
               "vector-add correctness invoked");
    std::cout << xdna::format_gate1_status(statuses);

    if (has_failed_state(statuses)) {
        return 1;
    }
    for (const auto& status : statuses) {
        if (status.state == Gate1State::completion_observed &&
            status.result == Gate1StateResult::pass) {
            return 0;
        }
    }
    return 3;
}

}  // namespace

int main(int argc, char** argv) {
    xdna::Gate1Options defaults;
    if (const char* xclbin = std::getenv("XDNA_GATE1_XCLBIN");
        xclbin != nullptr && xclbin[0] != '\0') {
        defaults.xclbin = xclbin;
    }
    if (const char* elf = std::getenv("XDNA_GATE1_ELF");
        elf != nullptr && elf[0] != '\0') {
        defaults.elf = elf;
    }

    const auto parsed = xdna::parse_gate1_arguments(argc, argv, defaults);
    if (parsed.help) {
        std::cout << xdna::gate1_usage(argv[0]);
        return 0;
    }
    if (!parsed.ok) {
        std::cerr << "argument error: " << parsed.error << '\n'
                  << xdna::gate1_usage(argv[0]);
        return 2;
    }
    return run_gate1(parsed.options);
}
