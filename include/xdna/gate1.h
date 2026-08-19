#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xdna {

// These values are read from the xclbin container and its AIE_METADATA and
// AIE_PARTITION sections.  They are deliberately kept independent of XRT so
// metadata validation and the no-XRT build exercise the same parser.
struct Gate1Metadata {
    bool container_valid = false;
    bool has_build_metadata = false;
    bool has_aie_metadata = false;
    bool has_aie_partition = false;
    bool target_validated = false;
    bool real_hardware_image = false;

    std::uint16_t xclbin_mode = 0;
    std::uint16_t action_mask = 0;
    std::string xclbin_mode_name;
    std::string platform_vbnv;
    std::string dsa_vendor;
    std::string dsa_board_id;
    std::string dsa_name;

    std::uint32_t hw_gen = 0;
    std::uint32_t num_rows = 0;
    std::uint32_t num_columns = 0;
    std::uint32_t compute_row_start = 0;
    std::uint32_t compute_row_count = 0;
    std::uint32_t partition_num_columns = 0;
    std::uint32_t partition_start_column = 0;
    std::uint32_t partition_section_width = 0;
    std::uint32_t partition_section_start_column = 0;

    std::string error;
};

Gate1Metadata parse_gate1_metadata(const std::vector<std::uint8_t>& image);
Gate1Metadata read_gate1_metadata(const std::filesystem::path& path);

std::string format_gate1_metadata(const Gate1Metadata& metadata);

struct Gate1Options {
    std::filesystem::path xclbin;
    std::filesystem::path elf;
    std::uint32_t device_index = 0;
    std::uint32_t timeout_ms = 10000;
    std::string kernel = "DPU:dpu";
};

struct Gate1ArgumentParse {
    bool ok = false;
    bool help = false;
    Gate1Options options;
    std::string error;
};

Gate1ArgumentParse parse_gate1_arguments(int argc, const char* const argv[],
                                         Gate1Options defaults = {});
std::string gate1_usage(std::string_view program);

enum class Gate1State {
    image_found,
    target_validated,
    device_opened,
    image_loaded,
    buffers_created,
    dispatch_submitted,
    completion_observed,
    output_reference_verified,
};

enum class Gate1StateResult {
    pass,
    fail,
    not_applicable,
};

struct Gate1Status {
    Gate1State state;
    Gate1StateResult result;
    std::string reason;
};

std::string_view gate1_state_name(Gate1State state);
std::string_view gate1_result_name(Gate1StateResult result);
std::string format_gate1_status(const std::vector<Gate1Status>& statuses);

}  // namespace xdna
