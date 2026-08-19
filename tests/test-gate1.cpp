#include "xdna/gate1.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace {

void expect(bool condition, const char* expression) {
    if (!condition) {
        std::cerr << "FAIL: " << expression << '\n';
        std::exit(1);
    }
}

void put_u16(std::vector<std::uint8_t>& image, std::size_t offset,
             std::uint16_t value) {
    image[offset] = static_cast<std::uint8_t>(value);
    image[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

void put_u32(std::vector<std::uint8_t>& image, std::size_t offset,
             std::uint32_t value) {
    for (unsigned byte = 0; byte < 4; ++byte) {
        image[offset + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
    }
}

void put_u64(std::vector<std::uint8_t>& image, std::size_t offset,
             std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte) {
        image[offset + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
    }
}

std::vector<std::uint8_t> fixture_xclbin(std::uint16_t mode = 0) {
    constexpr std::size_t header_size = 456;
    constexpr std::size_t section_header_size = 40;
    constexpr std::size_t section_count = 3;
    const std::string build =
        R"({"build_metadata":{"dsa":{"vendor":"xilinx","board_id":"v1","name":"ipu"}}})";
    const std::string aie = R"({"aie_metadata":{"driver_config":{"hw_gen":"3","num_rows":"6","num_columns":"5","aie_tile_row_start":"2","aie_tile_num_rows":"4","partition_num_cols":"4","partition_overlay_start_cols":["1"]}}})";
    std::vector<std::uint8_t> partition(208, 0);
    put_u16(partition, 32, 4);
    put_u32(partition, 40, 1);
    put_u32(partition, 44, 184);
    put_u16(partition, 184, 1);

    std::vector<std::uint8_t> image(header_size + section_count * section_header_size,
                                    0);
    const char magic[] = "xclbin2";
    for (std::size_t index = 0; index < sizeof(magic) - 1; ++index) {
        image[index] = static_cast<std::uint8_t>(magic[index]);
    }
    image[7] = 0;
    put_u16(image, 304 + 28, mode);
    put_u16(image, 304 + 30, 1);
    const std::string platform = "xilinx_v1_ipu_0_0";
    for (std::size_t index = 0; index < platform.size(); ++index) {
        image[304 + 48 + index] = static_cast<std::uint8_t>(platform[index]);
    }
    put_u32(image, 304 + 144, section_count);

    const std::vector<std::pair<std::uint32_t, std::vector<std::uint8_t>>> sections = {
        {14, std::vector<std::uint8_t>(build.begin(), build.end())},
        {25, std::vector<std::uint8_t>(aie.begin(), aie.end())},
        {32, partition},
    };
    for (std::size_t index = 0; index < sections.size(); ++index) {
        const auto& section = sections[index];
        const std::size_t section_header = header_size + index * section_header_size;
        const std::size_t section_offset = image.size();
        put_u32(image, section_header, section.first);
        // axlf_section_header is 4-byte kind + 16-byte name + 4-byte
        // alignment padding + uint64 offset + uint64 size.
        put_u64(image, section_header + 24, section_offset);
        put_u64(image, section_header + 32, section.second.size());
        image.insert(image.end(), section.second.begin(), section.second.end());
    }
    put_u64(image, 304, image.size());
    return image;
}

}  // namespace

int main() {
    const auto valid = xdna::parse_gate1_metadata(fixture_xclbin());
    expect(valid.container_valid, "valid fixture container");
    expect(valid.has_build_metadata && valid.has_aie_metadata &&
               valid.has_aie_partition,
           "valid fixture required sections");
    expect(valid.target_validated, "metadata target validates");
    expect(valid.real_hardware_image, "mode zero is real-hardware eligible");
    expect(valid.platform_vbnv == "xilinx_v1_ipu_0_0", "platform from header");
    expect(valid.hw_gen == 3 && valid.num_rows == 6 && valid.num_columns == 5,
           "AIE driver metadata dimensions");
    expect(valid.compute_row_start == 2 && valid.compute_row_count == 4,
           "AIE compute row metadata");
    expect(valid.partition_num_columns == 4 && valid.partition_start_column == 1,
           "JSON partition metadata");
    expect(valid.partition_section_width == 4 &&
               valid.partition_section_start_column == 1,
           "binary partition metadata cross-check");

    const auto emulation = xdna::parse_gate1_metadata(fixture_xclbin(4));
    expect(emulation.target_validated, "emulation image still has target metadata");
    expect(!emulation.real_hardware_image, "emulation image is refused for hardware");
    expect(emulation.xclbin_mode_name == "hw_emu", "mode name is decoded");

    auto wrong_platform = fixture_xclbin();
    wrong_platform[304 + 48] = 'z';
    const auto rejected = xdna::parse_gate1_metadata(wrong_platform);
    expect(!rejected.target_validated, "wrong platform is rejected");
    expect(!rejected.error.empty(), "rejected target has a reason");

    auto broken_section = fixture_xclbin();
    put_u64(broken_section, 456 + 24, broken_section.size() + 1);
    const auto malformed = xdna::parse_gate1_metadata(broken_section);
    expect(!malformed.container_valid, "section bounds invalidate container");

    const char* valid_argv[] = {"xdna-gate1", "--xclbin", "image.xclbin",
                                "--elf", "kernel.elf", "--device-index", "2",
                                "--timeout-ms", "42", "--kernel", "DPU:test"};
    const auto parsed = xdna::parse_gate1_arguments(
        static_cast<int>(std::size(valid_argv)), valid_argv);
    expect(parsed.ok, "valid argument set");
    expect(parsed.options.xclbin == "image.xclbin" &&
               parsed.options.elf == "kernel.elf" &&
               parsed.options.device_index == 2 && parsed.options.timeout_ms == 42 &&
               parsed.options.kernel == "DPU:test",
           "argument values are decoded");

    const char* missing_value[] = {"xdna-gate1", "--xclbin"};
    expect(!xdna::parse_gate1_arguments(2, missing_value).ok,
           "missing option value rejected");
    const char* unknown[] = {"xdna-gate1", "--xclbin", "a", "--bogus"};
    expect(!xdna::parse_gate1_arguments(4, unknown).ok,
           "unknown option rejected");
    const char* zero_timeout[] = {"xdna-gate1", "--xclbin", "a", "--timeout-ms", "0"};
    expect(!xdna::parse_gate1_arguments(5, zero_timeout).ok,
           "zero timeout rejected");
    const char* no_image[] = {"xdna-gate1"};
    expect(!xdna::parse_gate1_arguments(1, no_image).ok,
           "missing xclbin rejected");

    const std::vector<xdna::Gate1Status> statuses = {
        {xdna::Gate1State::image_found, xdna::Gate1StateResult::pass, "exists"},
        {xdna::Gate1State::target_validated, xdna::Gate1StateResult::pass, "metadata"},
        {xdna::Gate1State::device_opened, xdna::Gate1StateResult::fail, "exact error"},
        {xdna::Gate1State::image_loaded, xdna::Gate1StateResult::not_applicable, "blocked"},
        {xdna::Gate1State::buffers_created, xdna::Gate1StateResult::not_applicable, "blocked"},
        {xdna::Gate1State::dispatch_submitted, xdna::Gate1StateResult::not_applicable, "blocked"},
        {xdna::Gate1State::completion_observed, xdna::Gate1StateResult::not_applicable, "blocked"},
        {xdna::Gate1State::output_reference_verified, xdna::Gate1StateResult::not_applicable, "no contract"},
    };
    const auto status_text = xdna::format_gate1_status(statuses);
    expect(status_text.find("image_found: pass") != std::string::npos,
           "status pass formatting");
    expect(status_text.find("device_opened: fail (exact error)") != std::string::npos,
           "status fail formatting");
    expect(status_text.find("output_reference_verified: not_applicable") !=
               std::string::npos,
           "status not-applicable formatting");

    std::cout << "xdna Gate-1 metadata/status/argument tests: PASS\n";
    return 0;
}
