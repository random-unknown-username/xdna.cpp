#include "xdna/dma_streamer.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>

namespace xdna {

namespace {

// Assets are looked up relative to the working directory first, then in
// $XDNA_ASSET_DIR. An explicit path always wins and is passed through so the
// caller can report a useful error.
std::string find_file(const std::string& user_path, const std::string& default_rel) {
    if (!user_path.empty()) {
        return user_path;
    }
    if (std::filesystem::exists(default_rel)) {
        return default_rel;
    }
    if (const char* dir = std::getenv("XDNA_ASSET_DIR"); dir != nullptr && dir[0] != '\0') {
        const auto candidate = std::filesystem::path(dir) /
                               std::filesystem::path(default_rel).filename();
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }
    return default_rel;
}

inline bool operator!=(const xrt::bo& lhs, const xrt::bo& rhs) noexcept {
    return !(lhs == rhs);
}

} // namespace

xrt::bo pin_user_memory(xrt::device& dev, const void* ptr, size_t size) {
    if (!ptr) {
        throw std::invalid_argument("pin_user_memory: user pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("pin_user_memory: size must be greater than zero");
    }
    auto addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr & 4095) != 0) {
        throw std::invalid_argument("pin_user_memory: pointer address (" + std::to_string(addr) +
                                    ") must be 4096-byte page aligned");
    }
    try {
        return xrt::bo(dev, const_cast<void*>(ptr), size, xrt::bo::flags::host_only, 0);
    } catch (const std::exception& e) {
        throw std::runtime_error("pin_user_memory: failed to pin user memory: " + std::string(e.what()));
    }
}

PinnedBuffer pin_unaligned_user_memory(xrt::device& dev, const void* ptr, size_t size) {
    if (!ptr) {
        throw std::invalid_argument("pin_unaligned_user_memory: user pointer cannot be null");
    }
    if (size == 0) {
        throw std::invalid_argument("pin_unaligned_user_memory: size must be greater than zero");
    }
    auto addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t page_base = addr & ~4095UL;
    size_t page_offset = addr & 4095UL;
    size_t total_pinned = (page_offset + size + 4095UL) & ~4095UL;

    try {
        xrt::bo parent_bo(dev, reinterpret_cast<void*>(page_base), total_pinned, xrt::bo::flags::host_only, 0);
        xrt::bo sub_bo(parent_bo, size, page_offset);
        return PinnedBuffer{std::move(parent_bo), std::move(sub_bo), page_offset};
    } catch (const std::exception& e) {
        throw std::runtime_error("pin_unaligned_user_memory: failed to pin unaligned memory: " + std::string(e.what()));
    }
}

VmStat read_vmstat() {
    VmStat stat;
    std::ifstream file("/proc/vmstat");
    if (!file.is_open()) {
        return stat;
    }
    std::string key;
    uint64_t val = 0;
    while (file >> key >> val) {
        if (key == "pgmajfault") {
            stat.pgmajfault = val;
        } else if (key == "pswpout") {
            stat.pswpout = val;
        }
    }
    return stat;
}

bool verify_zero_swap_and_faults(const VmStat& before, const VmStat& after) {
    return (after.pgmajfault == before.pgmajfault) && (after.pswpout == before.pswpout);
}

struct DmaStreamer::Impl {
    xrt::device& dev;
    StreamConfig config;
    std::vector<xrt::bo> staging_ring;
    xrt::bo sink_bo[2]; // Per-slot destination instances bounded to staging size (<= 256 MB)

    // Bandwidth kernel (df_bw for >= 1024 MB transfers)
    xrt::uuid uuid_bw;
    xrt::hw_context hwctx_bw;
    xrt::module mod_bw;
    xrt::kernel kernel_bw;

    // Multi-column tile control kernel (tct_4col for <= 128 MB and general sizes)
    xrt::uuid uuid_tct;
    xrt::hw_context hwctx_tct;
    xrt::module mod_tct;
    xrt::kernel kernel_tct;

    bool has_bw_kernel = false;
    bool has_tct_kernel = false;
    bool hardware_ready = false;

    struct SlotState {
        xrt::run run_bw;
        xrt::run run_tct;
        bool busy = false;
        bool using_bw = false;
        bool using_sync = false;
        std::future<void> sync_future;
        size_t bytes = 0;
        uint64_t last_in_addr_bw = 0;
        uint64_t last_sink_addr_bw = 0;
        size_t last_in_bytes_bw = 0;
        uint64_t last_in_addr_tct = 0;
        uint64_t last_sink_addr_tct = 0;
        size_t last_in_bytes_tct = 0;
        std::chrono::high_resolution_clock::time_point start_time;
        StreamMetrics metrics;
    };
    SlotState slots[2];

    explicit Impl(xrt::device& d, const StreamConfig& cfg)
        : dev(d), config(cfg) {
        if (config.ring_buffer_bytes == 0) {
            throw std::invalid_argument("DmaStreamer: ring_buffer_bytes must be greater than zero");
        }
        if (config.num_columns <= 0) {
            throw std::invalid_argument("DmaStreamer: num_columns must be positive");
        }
        if (config.dma_channels_per_col <= 0) {
            throw std::invalid_argument("DmaStreamer: dma_channels_per_col must be positive");
        }

        // Allocate bounded double-buffered staging ring (2 slots, 128 MB each = 256 MB total)
        staging_ring.reserve(2);
        staging_ring.emplace_back(dev, config.ring_buffer_bytes, xrt::bo::flags::host_only, 0);
        staging_ring.emplace_back(dev, config.ring_buffer_bytes, xrt::bo::flags::host_only, 0);

        // Separate per-slot sink BO instances backed by staging ring memory to guarantee <= 256 MB pinned footprint
        sink_bo[0] = staging_ring[0];
        sink_bo[1] = staging_ring[1];

        // Resolve paths for df_bw
        std::string xclbin_file_bw = find_file(config.xclbin_path, "assets/phx_bins/validate_df_bandwidth.xclbin");
        std::string elf_file_bw = find_file(config.elf_path, "assets/phx_bins/df_bw.elf");

        // Resolve paths for tct_4col
        std::string xclbin_file_tct = find_file("", "assets/phx_bins/tct_all_col.xclbin");
        std::string elf_file_tct = find_file("", "assets/phx_bins/tct_4col.elf");

        bool bw_ok = false;
        if (std::filesystem::exists(xclbin_file_bw) && std::filesystem::exists(elf_file_bw)) {
            try {
                xrt::xclbin img(xclbin_file_bw);
                uuid_bw = dev.register_xclbin(img);
                hwctx_bw = xrt::hw_context(dev, uuid_bw);
                xrt::elf elf(elf_file_bw);
                mod_bw = xrt::module(elf);
                kernel_bw = xrt::ext::kernel(hwctx_bw, mod_bw, "DPU");
                slots[0].run_bw = xrt::run(kernel_bw);
                slots[1].run_bw = xrt::run(kernel_bw);
                bw_ok = true;
            } catch (...) {
                bw_ok = false;
            }
        }

        bool tct_ok = false;
        if (std::filesystem::exists(xclbin_file_tct) && std::filesystem::exists(elf_file_tct)) {
            try {
                xrt::xclbin img(xclbin_file_tct);
                uuid_tct = dev.register_xclbin(img);
                hwctx_tct = xrt::hw_context(dev, uuid_tct);
                xrt::elf elf(elf_file_tct);
                mod_tct = xrt::module(elf);
                kernel_tct = xrt::ext::kernel(hwctx_tct, mod_tct, "DPU_ELF");
                slots[0].run_tct = xrt::run(kernel_tct);
                slots[1].run_tct = xrt::run(kernel_tct);
                tct_ok = true;
            } catch (...) {
                tct_ok = false;
            }
        }

        has_bw_kernel = bw_ok;
        has_tct_kernel = tct_ok;
        hardware_ready = true;
    }

    ~Impl() {
        // Robust cleanup: drain or abort any active xrt::run handles or futures to prevent SIGABRT
        for (int i = 0; i < 2; ++i) {
            if (slots[i].busy) {
                if (slots[i].using_sync) {
                    if (slots[i].sync_future.valid()) {
                        slots[i].sync_future.wait();
                    }
                } else {
                    try {
                        auto& r = slots[i].using_bw ? slots[i].run_bw : slots[i].run_tct;
                        if (r) {
                            auto state = r.wait(500);
                            if (state != ERT_CMD_STATE_COMPLETED) {
                                r.abort();
                            }
                        }
                    } catch (...) {
                        try {
                            auto& r = slots[i].using_bw ? slots[i].run_bw : slots[i].run_tct;
                            if (r) {
                                r.abort();
                            }
                        } catch (...) {}
                    }
                }
                slots[i].busy = false;
            }
        }
    }

    xrt::bo pin_user_memory_impl(const void* page_aligned_ptr, size_t size) {
        return xdna::pin_user_memory(dev, page_aligned_ptr, size);
    }

    PinnedBuffer pin_unaligned_user_memory_impl(const void* ptr, size_t size) {
        return xdna::pin_unaligned_user_memory(dev, ptr, size);
    }

    void stream_async_impl(xrt::bo& src_bo, size_t offset, size_t bytes, int buffer_slot) {
        if (buffer_slot < 0 || buffer_slot >= 2) {
            throw std::out_of_range("stream_async: buffer_slot must be 0 or 1");
        }
        if (!hardware_ready) {
            throw std::runtime_error("stream_async: XRT hardware kernel is not initialized or ready");
        }
        auto& slot = slots[buffer_slot];
        if (slot.busy) {
            throw std::runtime_error("stream_async: slot " + std::to_string(buffer_slot) + " is already active");
        }
        if (bytes == 0) {
            throw std::invalid_argument("stream_async: transfer bytes must be positive");
        }
        if (bytes > config.ring_buffer_bytes) {
            throw std::invalid_argument("stream_async: transfer bytes (" + std::to_string(bytes) +
                                        ") exceeds ring buffer slot capacity (" +
                                        std::to_string(config.ring_buffer_bytes) + ")");
        }
        if ((offset & 4095) != 0) {
            throw std::invalid_argument("stream_async: offset (" + std::to_string(offset) +
                                        ") must be 4096-byte page aligned");
        }
        if (offset + bytes > src_bo.size()) {
            throw std::out_of_range("stream_async: offset + bytes (" + std::to_string(offset + bytes) +
                                    ") exceeds buffer size (" + std::to_string(src_bo.size()) + ")");
        }

        // Enforce 16-byte AXI burst alignment padding for Shim DMA engine transfers
        const size_t burst_bytes = (bytes + 15UL) & ~15UL;
        const size_t transfer_bytes = (offset + burst_bytes <= src_bo.size() && burst_bytes <= config.ring_buffer_bytes)
                                          ? burst_bytes
                                          : bytes;

        xrt::bo in_bo = (offset == 0 && transfer_bytes == src_bo.size())
                            ? src_bo
                            : xrt::bo(src_bo, transfer_bytes, offset);

        if (sink_bo[buffer_slot].size() < transfer_bytes) {
            sink_bo[buffer_slot] = xrt::bo(dev, transfer_bytes, xrt::bo::flags::host_only, 0);
        }
        xrt::bo& out_bo = sink_bo[buffer_slot];

        if (bytes >= 1024UL * 1024UL * 1024UL && has_bw_kernel) {
            slot.using_bw = true;
            slot.using_sync = false;
            auto& r = slot.run_bw;
            uint64_t in_addr = in_bo.address();
            uint64_t sink_addr = out_bo.address();
            if (slot.last_in_addr_bw != in_addr || slot.last_sink_addr_bw != sink_addr || slot.last_in_bytes_bw != bytes) {
                r.set_arg(0, static_cast<uint64_t>(3));
                r.set_arg(1, static_cast<uint64_t>(0));
                r.set_arg(2, static_cast<uint32_t>(0));
                r.set_arg(3, in_bo);
                r.set_arg(4, static_cast<uint64_t>(0));
                r.set_arg(5, out_bo);
                r.set_arg(6, static_cast<uint64_t>(0));
                r.set_arg(7, static_cast<uint64_t>(0));
                slot.last_in_addr_bw = in_addr;
                slot.last_sink_addr_bw = sink_addr;
                slot.last_in_bytes_bw = bytes;
            }
            slot.bytes = bytes;
            slot.start_time = std::chrono::high_resolution_clock::now();
            slot.busy = true;
            r.start();
        } else if (has_tct_kernel) {
            slot.using_bw = false;
            slot.using_sync = false;
            auto& r = slot.run_tct;
            uint64_t in_addr = in_bo.address();
            uint64_t sink_addr = out_bo.address();
            if (slot.last_in_addr_tct != in_addr || slot.last_sink_addr_tct != sink_addr || slot.last_in_bytes_tct != bytes) {
                r.set_arg(0, static_cast<uint64_t>(3));
                r.set_arg(1, static_cast<uint64_t>(0));
                r.set_arg(2, static_cast<uint32_t>(0));
                r.set_arg(3, in_bo);
                r.set_arg(4, static_cast<uint64_t>(0));
                r.set_arg(5, out_bo);
                r.set_arg(6, static_cast<uint64_t>(0));
                r.set_arg(7, static_cast<uint64_t>(0));
                slot.last_in_addr_tct = in_addr;
                slot.last_sink_addr_tct = sink_addr;
                slot.last_in_bytes_tct = bytes;
            }
            slot.bytes = bytes;
            slot.start_time = std::chrono::high_resolution_clock::now();
            slot.busy = true;
            r.start();
        } else {
            // Hardware fallback: Direct hardware DMA sync on /dev/accel/accel0
            slot.using_bw = false;
            slot.using_sync = true;
            slot.bytes = bytes;
            slot.start_time = std::chrono::high_resolution_clock::now();
            slot.busy = true;
            slot.sync_future = std::async(std::launch::async, [in_bo, bytes]() mutable {
                in_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
            });
        }
    }

    void wait_slot_impl(int buffer_slot) {
        if (buffer_slot < 0 || buffer_slot >= 2) {
            throw std::out_of_range("wait_slot: buffer_slot must be 0 or 1");
        }
        auto& slot = slots[buffer_slot];
        if (!slot.busy) {
            return;
        }

        // RAII guard to guarantee slot.busy is always reset to false, even on exception or timeout
        struct SlotGuard {
            bool& busy_ref;
            ~SlotGuard() { busy_ref = false; }
        } guard{slot.busy};

        if (slot.using_sync) {
            if (slot.sync_future.valid()) {
                slot.sync_future.wait();
            }
        } else {
            auto& active_run = slot.using_bw ? slot.run_bw : slot.run_tct;
            auto state = active_run.wait(10000);
            if (state != ERT_CMD_STATE_COMPLETED) {
                throw std::runtime_error("wait_slot: DMA transfer failed on slot " +
                                         std::to_string(buffer_slot) + " with state: " +
                                         std::to_string(static_cast<int>(state)));
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - slot.start_time).count();
        double duplex_bytes = 2.0 * static_cast<double>(slot.bytes);
        double throughput_gb_s = (elapsed_ms > 0.0) ? ((duplex_bytes / 1e9) / (elapsed_ms / 1000.0)) : 0.0;

        slot.metrics.elapsed_ms = elapsed_ms;
        slot.metrics.bytes_transferred = duplex_bytes;
        slot.metrics.physical_throughput_gb_s = throughput_gb_s;
        slot.metrics.active_columns = static_cast<uint32_t>(config.num_columns);
        slot.metrics.active_channels = static_cast<uint32_t>(config.num_columns * config.dma_channels_per_col);
    }

    size_t get_total_pinned_bytes_impl() const noexcept {
        size_t total = 0;
        for (const auto& bo : staging_ring) {
            if (bo) total += bo.size();
        }
        for (const auto& bo : sink_bo) {
            if (staging_ring.size() >= 2) {
                if (bo && bo != staging_ring[0] && bo != staging_ring[1]) {
                    total += bo.size();
                }
            } else if (bo) {
                total += bo.size();
            }
        }
        return total;
    }
};

DmaStreamer::DmaStreamer(xrt::device& dev, const StreamConfig& config)
    : m_impl(std::make_unique<Impl>(dev, config)) {}

DmaStreamer::~DmaStreamer() = default;

DmaStreamer::DmaStreamer(DmaStreamer&&) noexcept = default;
DmaStreamer& DmaStreamer::operator=(DmaStreamer&&) noexcept = default;

xrt::bo DmaStreamer::pin_user_memory(const void* page_aligned_ptr, size_t size) {
    return m_impl->pin_user_memory_impl(page_aligned_ptr, size);
}

PinnedBuffer DmaStreamer::pin_unaligned_user_memory(const void* ptr, size_t size) {
    return m_impl->pin_unaligned_user_memory_impl(ptr, size);
}

xrt::bo& DmaStreamer::get_staging_buffer(int slot) {
    if (slot < 0 || slot >= 2) {
        throw std::out_of_range("get_staging_buffer: slot must be 0 or 1");
    }
    return m_impl->staging_ring.at(slot);
}

const xrt::bo& DmaStreamer::get_staging_buffer(int slot) const {
    if (slot < 0 || slot >= 2) {
        throw std::out_of_range("get_staging_buffer: slot must be 0 or 1");
    }
    return m_impl->staging_ring.at(slot);
}

int DmaStreamer::num_slots() const noexcept {
    return 2;
}

size_t DmaStreamer::get_total_pinned_bytes() const noexcept {
    return m_impl->get_total_pinned_bytes_impl();
}

void DmaStreamer::stream_async(xrt::bo& src_bo, size_t offset, size_t bytes, int buffer_slot) {
    m_impl->stream_async_impl(src_bo, offset, bytes, buffer_slot);
}

void DmaStreamer::wait_slot(int buffer_slot) {
    m_impl->wait_slot_impl(buffer_slot);
}

bool DmaStreamer::is_hardware_ready() const noexcept {
    return m_impl->hardware_ready;
}

bool DmaStreamer::is_slot_busy(int slot) const {
    if (slot < 0 || slot >= 2) {
        throw std::out_of_range("is_slot_busy: slot must be 0 or 1");
    }
    return m_impl->slots[slot].busy;
}

StreamMetrics DmaStreamer::get_last_metrics(int slot) const {
    if (slot < 0 || slot >= 2) {
        throw std::out_of_range("get_last_metrics: slot must be 0 or 1");
    }
    return m_impl->slots[slot].metrics;
}

const StreamConfig& DmaStreamer::get_config() const noexcept {
    return m_impl->config;
}

} // namespace xdna
