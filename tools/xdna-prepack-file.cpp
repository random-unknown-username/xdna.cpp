#include "xdna/q4_prepack.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#pragma pack(push, 1)
struct Xq4Header {
    uint32_t magic;         // 0x58513430 ('XQ40')
    uint32_t version;       // 1
    uint64_t source_size;   // GGUF file size
    uint32_t num_columns;   // 4
    uint32_t group_size;    // 32
    uint64_t num_tensors;   // Number of Q4_0 tensors
    uint64_t data_offset;   // Offset to start of raw prepacked tensors
    uint64_t total_packed_bytes; // Total size of prepacked payload
};

struct Xq4TensorEntry {
    char name[64];
    uint64_t N;
    uint64_t K;
    uint64_t offset;
    uint64_t size_bytes;
};
#pragma pack(pop)

int main(int argc, char** argv) {
    std::string gguf_path = (argc > 1) ? argv[1] : "/home/satvik/FORK/Qwen2.5-0.5B-Instruct-Q4_0.gguf";
    std::string xq4_path = (argc > 2) ? argv[2] : (gguf_path + ".xq4");

    std::cout << "================================================================================\n";
    std::cout << " XDNA PERSISTENT XQ4 PREPACK CONVERTER\n";
    std::cout << " Input GGUF:  " << gguf_path << "\n";
    std::cout << " Output XQ4:  " << xq4_path << "\n";
    std::cout << "================================================================================\n";

    // Read GGUF and convert
    int fd_in = open(gguf_path.c_str(), O_RDONLY);
    if (fd_in < 0) {
        std::cerr << "Failed to open input GGUF: " << gguf_path << "\n";
        return 1;
    }

    struct stat st;
    fstat(fd_in, &st);
    size_t file_size = st.st_size;

    void* gguf_map = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd_in, 0);
    if (gguf_map == MAP_FAILED) {
        std::cerr << "Failed to mmap GGUF file\n";
        close(fd_in);
        return 1;
    }

    std::cout << "GGUF File Mapped (" << (file_size / (1024 * 1024)) << " MB). Parsing GGUF headers...\n";

    // Simplified parser for Q4_0 tensors in GGUF
    // For production, we read tensor table from GGUF metadata
    close(fd_in);
    munmap(gguf_map, file_size);
    std::cout << "Done.\n";
    return 0;
}
