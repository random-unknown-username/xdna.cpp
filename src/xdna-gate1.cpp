#include "xdna/gate1.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace xdna {
namespace {

// Layout constants mirror the public axlf layout in the installed
// xrt/detail/xclbin.h.  Byte parsing keeps this validator available when XRT
// is deliberately disabled and avoids treating a filename as metadata.
constexpr std::size_t kAxlfHeaderOffset = 304;
constexpr std::size_t kAxlfHeaderLengthOffset = kAxlfHeaderOffset + 0;
constexpr std::size_t kAxlfHeaderModeOffset = kAxlfHeaderOffset + 28;
constexpr std::size_t kAxlfHeaderActionMaskOffset = kAxlfHeaderOffset + 30;
constexpr std::size_t kAxlfHeaderPlatformOffset = kAxlfHeaderOffset + 48;
constexpr std::size_t kAxlfHeaderSectionCountOffset = kAxlfHeaderOffset + 144;
constexpr std::size_t kAxlfSectionsOffset = 456;
constexpr std::size_t kAxlfSectionSize = 40;

constexpr std::uint32_t kBuildMetadataSection = 14;
constexpr std::uint32_t kAieMetadataSection = 25;
constexpr std::uint32_t kAiePartitionSection = 32;
constexpr std::uint16_t kLoadAieAction = 0x1;

// aie_partition_info::column_width and ::start_columns in the installed
// xrt/detail/xclbin.h-backed AIE_PARTITION section.
constexpr std::size_t kPartitionColumnWidthOffset = 32;
constexpr std::size_t kPartitionStartColumnsCountOffset = 40;
constexpr std::size_t kPartitionStartColumnsOffsetOffset = 44;

std::uint16_t read_u16(const std::vector<std::uint8_t>& data,
                       std::size_t offset) {
    return static_cast<std::uint16_t>(data[offset]) |
           (static_cast<std::uint16_t>(data[offset + 1]) << 8);
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& data,
                       std::size_t offset) {
    return static_cast<std::uint32_t>(data[offset]) |
           (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(data[offset + 3]) << 24);
}

std::uint64_t read_u64(const std::vector<std::uint8_t>& data,
                       std::size_t offset) {
    std::uint64_t value = 0;
    for (unsigned byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(data[offset + byte]) << (byte * 8);
    }
    return value;
}

std::string fixed_string(const std::vector<std::uint8_t>& data,
                         std::size_t offset, std::size_t size) {
    const auto end = std::find(data.begin() + static_cast<std::ptrdiff_t>(offset),
                               data.begin() + static_cast<std::ptrdiff_t>(offset + size),
                               static_cast<std::uint8_t>(0));
    return std::string(data.begin() + static_cast<std::ptrdiff_t>(offset), end);
}

std::string trim(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() &&
           (text[begin] == ' ' || text[begin] == '\n' || text[begin] == '\r' ||
            text[begin] == '\t')) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin &&
           (text[end - 1] == ' ' || text[end - 1] == '\n' ||
            text[end - 1] == '\r' || text[end - 1] == '\t')) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

std::optional<std::size_t> json_value_start(std::string_view object,
                                             std::string_view key) {
    const std::string needle = "\"" + std::string(key) + "\"";
    const auto key_pos = object.find(needle);
    if (key_pos == std::string_view::npos) {
        return std::nullopt;
    }
    auto pos = key_pos + needle.size();
    while (pos < object.size() &&
           (object[pos] == ' ' || object[pos] == '\n' || object[pos] == '\r' ||
            object[pos] == '\t')) {
        ++pos;
    }
    if (pos >= object.size() || object[pos] != ':') {
        return std::nullopt;
    }
    ++pos;
    while (pos < object.size() &&
           (object[pos] == ' ' || object[pos] == '\n' || object[pos] == '\r' ||
            object[pos] == '\t')) {
        ++pos;
    }
    return pos;
}

std::optional<std::string> json_string(std::string_view object,
                                       std::string_view key) {
    const auto start = json_value_start(object, key);
    if (!start || *start >= object.size() || object[*start] != '"') {
        return std::nullopt;
    }
    std::string value;
    bool escaped = false;
    for (std::size_t pos = *start + 1; pos < object.size(); ++pos) {
        const char character = object[pos];
        if (escaped) {
            switch (character) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                default: value.push_back(character); break;
            }
            escaped = false;
        } else if (character == '\\') {
            escaped = true;
        } else if (character == '"') {
            return value;
        } else {
            value.push_back(character);
        }
    }
    return std::nullopt;
}

std::optional<std::string> json_object(std::string_view object,
                                       std::string_view key) {
    const auto start = json_value_start(object, key);
    if (!start || *start >= object.size() || object[*start] != '{') {
        return std::nullopt;
    }
    std::size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t pos = *start; pos < object.size(); ++pos) {
        const char character = object[pos];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (character == '\\') {
                escaped = true;
            } else if (character == '"') {
                in_string = false;
            }
            continue;
        }
        if (character == '"') {
            in_string = true;
        } else if (character == '{') {
            ++depth;
        } else if (character == '}') {
            --depth;
            if (depth == 0) {
                return std::string(object.substr(*start, pos - *start + 1));
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> json_first_array_string(std::string_view object,
                                                   std::string_view key) {
    const auto start = json_value_start(object, key);
    if (!start || *start >= object.size() || object[*start] != '[') {
        return std::nullopt;
    }
    const auto quote = object.find('"', *start + 1);
    if (quote == std::string_view::npos) {
        return std::nullopt;
    }
    std::string value;
    bool escaped = false;
    for (std::size_t pos = quote + 1; pos < object.size(); ++pos) {
        const char character = object[pos];
        if (escaped) {
            value.push_back(character);
            escaped = false;
        } else if (character == '\\') {
            escaped = true;
        } else if (character == '"') {
            return value;
        } else {
            value.push_back(character);
        }
    }
    return std::nullopt;
}

std::optional<std::uint32_t> decimal_string(std::string_view text) {
    const std::string clean = trim(text);
    if (clean.empty()) {
        return std::nullopt;
    }
    std::uint32_t value = 0;
    const auto result = std::from_chars(clean.data(), clean.data() + clean.size(),
                                        value, 10);
    if (result.ec != std::errc{} || result.ptr != clean.data() + clean.size()) {
        return std::nullopt;
    }
    return value;
}

std::optional<std::uint32_t> json_decimal(std::string_view object,
                                          std::string_view key) {
    const auto value = json_string(object, key);
    return value ? decimal_string(*value) : std::nullopt;
}

std::string mode_name(std::uint16_t mode) {
    switch (mode) {
        case 0: return "hw";
        case 4: return "hw_emu";
        case 5: return "sw_emu";
        case 6: return "hw_emu_pr";
        default: return "unknown";
    }
}

void set_error(Gate1Metadata& metadata, std::string message) {
    if (metadata.error.empty()) {
        metadata.error = std::move(message);
    }
}

struct Section {
    std::uint32_t kind = 0;
    std::size_t offset = 0;
    std::size_t size = 0;
};

std::optional<Section> find_section(const std::vector<std::uint8_t>& image,
                                    std::uint32_t kind,
                                    std::uint32_t count) {
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto offset = kAxlfSectionsOffset +
                            static_cast<std::size_t>(index) * kAxlfSectionSize;
        if (read_u32(image, offset) != kind) {
            continue;
        }
        // axlf_section_header has four bytes of padding after the 16-byte
        // name so the two uint64_t fields are aligned at offsets 24 and 32.
        // Reading at 16/24 accidentally interpreted the tail of the name as
        // a file offset and rejected every real xclbin section table.
        const auto section_offset = read_u64(image, offset + 24);
        const auto section_size = read_u64(image, offset + 32);
        if (section_offset > image.size() || section_size > image.size() - section_offset ||
            section_offset > std::numeric_limits<std::size_t>::max() ||
            section_size > std::numeric_limits<std::size_t>::max()) {
            return std::nullopt;
        }
        return Section{kind, static_cast<std::size_t>(section_offset),
                       static_cast<std::size_t>(section_size)};
    }
    return std::nullopt;
}

std::string section_text(const std::vector<std::uint8_t>& image,
                         const Section& section) {
    return std::string(
        image.begin() + static_cast<std::ptrdiff_t>(section.offset),
        image.begin() + static_cast<std::ptrdiff_t>(section.offset + section.size));
}

}  // namespace

Gate1Metadata parse_gate1_metadata(const std::vector<std::uint8_t>& image) {
    Gate1Metadata metadata;
    if (image.size() < kAxlfSectionsOffset) {
        set_error(metadata, "xclbin is shorter than its axlf header");
        return metadata;
    }
    if (image[0] != 'x' || image[1] != 'c' || image[2] != 'l' ||
        image[3] != 'b' || image[4] != 'i' || image[5] != 'n' ||
        image[6] != '2' || image[7] != 0) {
        set_error(metadata, "xclbin magic is not xclbin2");
        return metadata;
    }

    const auto declared_length = read_u64(image, kAxlfHeaderLengthOffset);
    if (declared_length != image.size()) {
        set_error(metadata, "xclbin length does not match the selected file");
        return metadata;
    }
    const auto section_count = read_u32(image, kAxlfHeaderSectionCountOffset);
    if (section_count == 0 || section_count > 0x10000 ||
        section_count > (image.size() - kAxlfSectionsOffset) / kAxlfSectionSize) {
        set_error(metadata, "xclbin section table is outside the selected file");
        return metadata;
    }
    for (std::uint32_t index = 0; index < section_count; ++index) {
        const auto offset = kAxlfSectionsOffset +
                            static_cast<std::size_t>(index) * kAxlfSectionSize;
        const auto section_offset = read_u64(image, offset + 24);
        const auto section_size = read_u64(image, offset + 32);
        if (section_offset > image.size() || section_size > image.size() - section_offset) {
            set_error(metadata, "xclbin section extends beyond the selected file");
            return metadata;
        }
    }

    metadata.container_valid = true;
    metadata.xclbin_mode = read_u16(image, kAxlfHeaderModeOffset);
    metadata.action_mask = read_u16(image, kAxlfHeaderActionMaskOffset);
    metadata.xclbin_mode_name = mode_name(metadata.xclbin_mode);
    metadata.platform_vbnv = fixed_string(image, kAxlfHeaderPlatformOffset, 64);

    const auto build = find_section(image, kBuildMetadataSection, section_count);
    const auto aie = find_section(image, kAieMetadataSection, section_count);
    const auto partition = find_section(image, kAiePartitionSection, section_count);
    metadata.has_build_metadata = build.has_value();
    metadata.has_aie_metadata = aie.has_value();
    metadata.has_aie_partition = partition.has_value();
    if (!build || !aie || !partition) {
        set_error(metadata, "xclbin is missing BUILD_METADATA, AIE_METADATA, or AIE_PARTITION");
        return metadata;
    }

    const std::string build_text = section_text(image, *build);
    const std::string aie_text = section_text(image, *aie);
    const auto build_root = json_object(build_text, "build_metadata");
    const auto dsa = build_root ? json_object(*build_root, "dsa") : std::nullopt;
    const auto aie_root = json_object(aie_text, "aie_metadata");
    const auto driver_config = aie_root ? json_object(*aie_root, "driver_config") : std::nullopt;
    if (!dsa || !driver_config) {
        set_error(metadata, "BUILD_METADATA or AIE_METADATA is not valid JSON metadata");
        return metadata;
    }

    metadata.dsa_vendor = json_string(*dsa, "vendor").value_or("");
    metadata.dsa_board_id = json_string(*dsa, "board_id").value_or("");
    metadata.dsa_name = json_string(*dsa, "name").value_or("");
    const auto hw_gen = json_decimal(*driver_config, "hw_gen");
    const auto num_rows = json_decimal(*driver_config, "num_rows");
    const auto num_columns = json_decimal(*driver_config, "num_columns");
    const auto row_start = json_decimal(*driver_config, "aie_tile_row_start");
    const auto row_count = json_decimal(*driver_config, "aie_tile_num_rows");
    const auto partition_columns = json_decimal(*driver_config, "partition_num_cols");
    const auto overlay_start =
        json_first_array_string(*driver_config, "partition_overlay_start_cols");
    if (!hw_gen || !num_rows || !num_columns || !row_start || !row_count ||
        !partition_columns || !overlay_start || !decimal_string(*overlay_start)) {
        set_error(metadata, "AIE_METADATA driver_config lacks required target fields");
        return metadata;
    }
    metadata.hw_gen = *hw_gen;
    metadata.num_rows = *num_rows;
    metadata.num_columns = *num_columns;
    metadata.compute_row_start = *row_start;
    metadata.compute_row_count = *row_count;
    metadata.partition_num_columns = *partition_columns;
    metadata.partition_start_column = *decimal_string(*overlay_start);

    const auto partition_offset = partition->offset;
    if (partition->size < kPartitionStartColumnsOffsetOffset + sizeof(std::uint32_t) ||
        partition->size < kPartitionColumnWidthOffset + sizeof(std::uint16_t)) {
        set_error(metadata, "AIE_PARTITION section is shorter than its public header");
        return metadata;
    }
    metadata.partition_section_width = read_u16(image, partition_offset +
                                                 kPartitionColumnWidthOffset);
    const auto start_count = read_u32(image, partition_offset +
                                      kPartitionStartColumnsCountOffset);
    const auto start_offset = read_u32(image, partition_offset +
                                       kPartitionStartColumnsOffsetOffset);
    if (start_count == 0 || start_count > partition->size / sizeof(std::uint16_t) ||
        start_offset > partition->size ||
        static_cast<std::uint64_t>(start_count) * sizeof(std::uint16_t) >
            partition->size - start_offset) {
        set_error(metadata, "AIE_PARTITION start_columns array is outside its section");
        return metadata;
    }
    metadata.partition_section_start_column =
        read_u16(image, partition_offset + start_offset);

    const bool profile_matches =
        metadata.platform_vbnv == "xilinx_v1_ipu_0_0" &&
        metadata.dsa_vendor == "xilinx" && metadata.dsa_board_id == "v1" &&
        metadata.dsa_name == "ipu" && metadata.hw_gen == 3 &&
        metadata.num_rows == 6 && metadata.num_columns == 5 &&
        metadata.compute_row_start == 2 && metadata.compute_row_count == 4 &&
        (metadata.action_mask & kLoadAieAction) != 0 &&
        metadata.partition_num_columns > 0 &&
        metadata.partition_start_column + metadata.partition_num_columns <=
            metadata.num_columns &&
        metadata.partition_section_width == metadata.partition_num_columns &&
        metadata.partition_section_start_column == metadata.partition_start_column;
    metadata.target_validated = profile_matches;
    metadata.real_hardware_image = metadata.target_validated && metadata.xclbin_mode == 0;
    if (!profile_matches) {
        set_error(metadata,
                  "xclbin metadata does not match the Phoenix/NPU1/AIE2 profile");
    }
    return metadata;
}

Gate1Metadata read_gate1_metadata(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        Gate1Metadata metadata;
        metadata.error = "cannot open selected xclbin: " + path.string();
        return metadata;
    }
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end <= 0) {
        Gate1Metadata metadata;
        metadata.error = "selected xclbin is empty: " + path.string();
        return metadata;
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> image(static_cast<std::size_t>(end));
    input.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(image.size()));
    if (!input) {
        Gate1Metadata metadata;
        metadata.error = "cannot read selected xclbin: " + path.string();
        return metadata;
    }
    return parse_gate1_metadata(image);
}

std::string format_gate1_metadata(const Gate1Metadata& metadata) {
    std::ostringstream output;
    output << "xclbin metadata:\n";
    output << "  container_valid: " << (metadata.container_valid ? "yes" : "no") << '\n';
    output << "  platform_vbnv: " << (metadata.platform_vbnv.empty() ? "unknown" : metadata.platform_vbnv) << '\n';
    output << "  dsa: vendor=" << (metadata.dsa_vendor.empty() ? "unknown" : metadata.dsa_vendor)
           << ", board_id=" << (metadata.dsa_board_id.empty() ? "unknown" : metadata.dsa_board_id)
           << ", name=" << (metadata.dsa_name.empty() ? "unknown" : metadata.dsa_name) << '\n';
    output << "  mode: " << metadata.xclbin_mode << " ("
           << (metadata.xclbin_mode_name.empty() ? "unknown" : metadata.xclbin_mode_name)
           << ")\n";
    output << "  action_mask: 0x" << std::hex << metadata.action_mask << std::dec << '\n';
    output << "  AIE_METADATA: hw_gen=" << metadata.hw_gen
           << ", rows=" << metadata.num_rows << ", columns=" << metadata.num_columns;
    if (metadata.compute_row_count != 0) {
        output << ", compute_rows=" << metadata.compute_row_start << ".."
               << metadata.compute_row_start + metadata.compute_row_count - 1;
    } else {
        output << ", compute_rows=unknown";
    }
    output << '\n';
    output << "  AIE partition: width=" << metadata.partition_num_columns
           << ", start_column=" << metadata.partition_start_column
           << " (section width=" << metadata.partition_section_width
           << ", section start=" << metadata.partition_section_start_column << ")\n";
    output << "  Phoenix/NPU1/AIE2 target metadata: "
           << (metadata.target_validated ? "validated" : "rejected") << '\n';
    output << "  real-hardware image mode: "
           << (metadata.real_hardware_image ? "yes" : "no");
    if (!metadata.error.empty()) {
        output << "\n  metadata error: " << metadata.error;
    }
    output << '\n';
    return output.str();
}

Gate1ArgumentParse parse_gate1_arguments(int argc, const char* const argv[],
                                         Gate1Options defaults) {
    Gate1ArgumentParse result;
    result.options = std::move(defaults);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index] == nullptr ? "" : argv[index];
        if (argument == "--help" || argument == "-h") {
            result.ok = true;
            result.help = true;
            return result;
        }
        auto take_value = [&](const char* option) -> std::optional<std::string> {
            if (index + 1 >= argc || argv[index + 1] == nullptr ||
                argv[index + 1][0] == '\0') {
                result.error = std::string(option) + " requires a value";
                return std::nullopt;
            }
            ++index;
            return std::string(argv[index]);
        };
        if (argument == "--xclbin") {
            const auto value = take_value("--xclbin");
            if (!value) return result;
            result.options.xclbin = *value;
        } else if (argument == "--elf") {
            const auto value = take_value("--elf");
            if (!value) return result;
            result.options.elf = *value;
        } else if (argument == "--device-index") {
            const auto value = take_value("--device-index");
            if (!value) return result;
            const auto parsed = decimal_string(*value);
            if (!parsed) {
                result.error = "--device-index requires a non-negative decimal integer";
                return result;
            }
            result.options.device_index = *parsed;
        } else if (argument == "--timeout-ms") {
            const auto value = take_value("--timeout-ms");
            if (!value) return result;
            const auto parsed = decimal_string(*value);
            if (!parsed || *parsed == 0) {
                result.error = "--timeout-ms requires a positive decimal integer";
                return result;
            }
            result.options.timeout_ms = *parsed;
        } else if (argument == "--kernel") {
            const auto value = take_value("--kernel");
            if (!value) return result;
            result.options.kernel = *value;
        } else {
            result.error = "unknown argument: " + argument;
            return result;
        }
    }
    if (result.options.xclbin.empty()) {
        result.error = "--xclbin is required (or set XDNA_GATE1_XCLBIN)";
        return result;
    }
    if (result.options.kernel.empty()) {
        result.error = "--kernel requires a non-empty value";
        return result;
    }
    result.ok = true;
    return result;
}

std::string gate1_usage(std::string_view program) {
    std::ostringstream output;
    output << "Usage: " << program
           << " --xclbin PATH [--elf PATH] [options]\n"
           << "\n"
           << "Real-device Gate-1 harness. The xclbin is metadata-validated;"
           << " no vector-add/reference check is performed.\n"
           << "Options:\n"
           << "  --xclbin PATH       selected xclbin (or XDNA_GATE1_XCLBIN)\n"
           << "  --elf PATH          XRT ELF for the demonstrated NPU dispatch path\n"
           << "  --device-index N    XRT device index (default 0)\n"
           << "  --timeout-ms N      completion wait timeout (default 10000)\n"
           << "  --kernel NAME       XRT ELF kernel name (default DPU:dpu)\n"
           << "  --help              show this text\n";
    return output.str();
}

std::string_view gate1_state_name(Gate1State state) {
    switch (state) {
        case Gate1State::image_found: return "image_found";
        case Gate1State::target_validated: return "target_validated";
        case Gate1State::device_opened: return "device_opened";
        case Gate1State::image_loaded: return "image_loaded";
        case Gate1State::buffers_created: return "buffers_created";
        case Gate1State::dispatch_submitted: return "dispatch_submitted";
        case Gate1State::completion_observed: return "completion_observed";
        case Gate1State::output_reference_verified: return "output_reference_verified";
    }
    return "unknown";
}

std::string_view gate1_result_name(Gate1StateResult result) {
    switch (result) {
        case Gate1StateResult::pass: return "pass";
        case Gate1StateResult::fail: return "fail";
        case Gate1StateResult::not_applicable: return "not_applicable";
    }
    return "unknown";
}

std::string format_gate1_status(const std::vector<Gate1Status>& statuses) {
    std::ostringstream output;
    output << "Gate-1 states:\n";
    for (const auto& status : statuses) {
        output << "  " << gate1_state_name(status.state) << ": "
               << gate1_result_name(status.result) << " (" << status.reason << ")\n";
    }
    return output.str();
}

}  // namespace xdna
