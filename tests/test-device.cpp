#include "xdna/device.h"

#include <cstdlib>
#include <iostream>

namespace {

void expect(bool condition, const char* expression) {
    if (!condition) {
        std::cerr << "FAIL: " << expression << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    const auto compact = xdna::parse_topology("4x5");
    expect(compact.has_value(), "parse_topology(4x5)");
    expect(compact->rows == 4 && compact->columns == 5,
           "compact topology has rows=4 columns=5");
    expect(compact->compute_tiles() == 20, "4x5 has 20 compute tiles");
    expect(xdna::format_topology(*compact) ==
               "4 rows x 5 columns (20 compute tiles)",
           "format_topology(4x5)");

    const auto named = xdna::parse_topology("rows=4 columns=5");
    expect(named.has_value() && named->rows == 4 && named->columns == 5,
           "parse named topology");

    const auto reversed_words = xdna::parse_topology("5 columns x 4 rows");
    expect(reversed_words.has_value() && reversed_words->rows == 4 &&
               reversed_words->columns == 5,
           "parse column-first topology");

    const auto unicode = xdna::parse_topology(u8"4×5");
    expect(unicode.has_value() && unicode->rows == 4 && unicode->columns == 5,
           "parse unicode multiplication sign");

    expect(!xdna::parse_topology("not a topology").has_value(),
           "reject invalid topology");
    expect(!xdna::parse_topology("0x5").has_value(),
           "reject zero topology dimension");
    expect(xdna::format_topology(xdna::Topology{}) == "unknown",
           "format invalid topology");

    const auto uevent = xdna::parse_key_value_lines(
        "DRIVER=amdxdna\nPCI_SLOT_NAME=0000:06:00.1\n"
        "PCI_ID=1022:1502\nignored line\n");
    expect(uevent.at("DRIVER") == "amdxdna", "parse DRIVER uevent");
    expect(uevent.at("PCI_SLOT_NAME") == "0000:06:00.1",
           "parse PCI slot uevent");
    expect(uevent.at("PCI_ID") == "1022:1502", "parse PCI ID uevent");

    expect(xdna::classify_architecture("RyzenAI-npu1") ==
               xdna::Architecture::xdna1_aie2,
           "classify NPU1 VBNV");
    expect(xdna::classify_architecture("unknown", "0x1502") ==
               xdna::Architecture::xdna1_aie2,
           "classify NPU1 PCI ID");
    expect(xdna::architecture_name(xdna::Architecture::xdna1_aie2) ==
               "XDNA1 / AIE2",
           "format XDNA1 architecture");

    std::cout << "xdna device parsing tests: PASS\n";
    return 0;
}

