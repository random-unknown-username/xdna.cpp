#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>

namespace xdna {

/// Configuration for parallel Shim DMA streaming and bounded staging ring.
struct StreamConfig {
    /// Bounded staging ring buffer size per slot (default 128 MB per slot, 2 slots = 256 MB total).
    size_t ring_buffer_bytes = 128 * 1024 * 1024;
    /// Number of active columns in AIE partition (Phoenix/Hawk Point has 4 active compute columns).
    int num_columns = 4;
    /// Number of Shim MM2S DMA channels per active column (2 channels per column = 8 channels total).
    int dma_channels_per_col = 2;
    /// Optional path to XCLBIN container. If empty, standard candidate paths are searched.
    std::string xclbin_path;
    /// Optional path to control code ELF. If empty, standard candidate paths are searched.
    std::string elf_path;
};

/// Hardware execution and throughput metrics for streaming transfers.
struct StreamMetrics {
    double elapsed_ms = 0.0;
    double bytes_transferred = 0.0;
    double physical_throughput_gb_s = 0.0;
    uint32_t active_columns = 4;
    uint32_t active_channels = 8;
};

/// Linux VM statistics snapshot for verifying bounded memory footprint.
struct VmStat {
    uint64_t pgmajfault = 0;
    uint64_t pswpout = 0;
};

/// Structure representing pinned virtual memory with sub-buffer offset support.
struct PinnedBuffer {
    xrt::bo parent_bo;
    xrt::bo sub_bo;
    size_t offset = 0;
};

/// Pin user virtual memory into NPU IOMMU zero-copy without memcpy using xrt::bo::flags::host_only.
/// The memory must be 4096-byte page aligned and have write permissions in VMA (e.g. MAP_PRIVATE | PROT_READ | PROT_WRITE).
xrt::bo pin_user_memory(xrt::device& dev, const void* ptr, size_t size);

/// Pin arbitrary user virtual memory into NPU IOMMU zero-copy (e.g. 32-byte aligned GGUF tensors).
PinnedBuffer pin_unaligned_user_memory(xrt::device& dev, const void* ptr, size_t size);

/// Read system memory activity counters (/proc/vmstat).
VmStat read_vmstat();

/// Verify zero major page faults and zero swap activity between two vmstat snapshots.
bool verify_zero_swap_and_faults(const VmStat& before, const VmStat& after);

/// High-performance zero-copy DMA streaming engine and bounded double-buffered staging ring.
class DmaStreamer {
public:
    explicit DmaStreamer(xrt::device& dev, const StreamConfig& config = {});
    ~DmaStreamer();

    DmaStreamer(const DmaStreamer&) = delete;
    DmaStreamer& operator=(const DmaStreamer&) = delete;
    DmaStreamer(DmaStreamer&&) noexcept;
    DmaStreamer& operator=(DmaStreamer&&) noexcept;

    /// Pin user memory zero-copy without memcpy using this streamer's device.
    xrt::bo pin_user_memory(const void* page_aligned_ptr, size_t size);

    /// Pin arbitrary unaligned user memory zero-copy (e.g. 32-byte aligned GGUF weights).
    PinnedBuffer pin_unaligned_user_memory(const void* ptr, size_t size);

    /// Get staging buffer for slot 0 or slot 1.
    xrt::bo& get_staging_buffer(int slot);
    const xrt::bo& get_staging_buffer(int slot) const;

    /// Number of slots in the staging ring (2 slots).
    int num_slots() const noexcept;

    /// Total pinned memory bytes currently managed by DmaStreamer.
    size_t get_total_pinned_bytes() const noexcept;

    /// Submit asynchronous 4-column parallel streaming transfer on buffer_slot (0 or 1).
    void stream_async(xrt::bo& src_bo, size_t offset, size_t bytes, int buffer_slot);

    /// Wait for streaming transfer completion on buffer_slot (0 or 1).
    void wait_slot(int buffer_slot);

    /// Check if hardware execution kernel and context are ready.
    bool is_hardware_ready() const noexcept;

    /// Check if a slot currently has an in-flight transfer.
    bool is_slot_busy(int slot) const;

    /// Get metrics from the last completed transfer on a given slot.
    StreamMetrics get_last_metrics(int slot) const;

    /// Get active streaming configuration.
    const StreamConfig& get_config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace xdna
