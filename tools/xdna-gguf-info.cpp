#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    UNKNOWN = 999
};

const char* ggml_type_name(GgmlType type) {
    switch (type) {
        case GgmlType::F32: return "F32";
        case GgmlType::F16: return "F16";
        case GgmlType::Q4_0: return "Q4_0";
        case GgmlType::Q4_1: return "Q4_1";
        case GgmlType::Q5_0: return "Q5_0";
        case GgmlType::Q5_1: return "Q5_1";
        case GgmlType::Q8_0: return "Q8_0";
        case GgmlType::Q8_1: return "Q8_1";
        case GgmlType::Q2_K: return "Q2_K";
        case GgmlType::Q3_K: return "Q3_K";
        case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q5_K: return "Q5_K";
        case GgmlType::Q6_K: return "Q6_K";
        case GgmlType::Q8_K: return "Q8_K";
        case GgmlType::IQ2_XXS: return "IQ2_XXS";
        case GgmlType::IQ2_XS: return "IQ2_XS";
        case GgmlType::IQ3_XXS: return "IQ3_XXS";
        case GgmlType::IQ1_S: return "IQ1_S";
        case GgmlType::IQ4_NL: return "IQ4_NL";
        case GgmlType::IQ3_S: return "IQ3_S";
        case GgmlType::IQ2_S: return "IQ2_S";
        case GgmlType::IQ4_XS: return "IQ4_XS";
        case GgmlType::I8: return "I8";
        case GgmlType::I16: return "I16";
        case GgmlType::I32: return "I32";
        case GgmlType::I64: return "I64";
        case GgmlType::F64: return "F64";
        case GgmlType::IQ1_M: return "IQ1_M";
        case GgmlType::BF16: return "BF16";
        default: return "UNKNOWN";
    }
}

size_t ggml_type_block_size(GgmlType type) {
    switch (type) {
        case GgmlType::F32: return 1;
        case GgmlType::F16: return 1;
        case GgmlType::BF16: return 1;
        case GgmlType::I8: return 1;
        case GgmlType::I16: return 1;
        case GgmlType::I32: return 1;
        case GgmlType::I64: return 1;
        case GgmlType::F64: return 1;
        case GgmlType::Q4_0: return 32;
        case GgmlType::Q4_1: return 32;
        case GgmlType::Q5_0: return 32;
        case GgmlType::Q5_1: return 32;
        case GgmlType::Q8_0: return 32;
        case GgmlType::Q8_1: return 32;
        case GgmlType::Q2_K: return 256;
        case GgmlType::Q3_K: return 256;
        case GgmlType::Q4_K: return 256;
        case GgmlType::Q5_K: return 256;
        case GgmlType::Q6_K: return 256;
        case GgmlType::Q8_K: return 256;
        case GgmlType::IQ2_XXS: return 256;
        case GgmlType::IQ2_XS: return 256;
        case GgmlType::IQ3_XXS: return 256;
        case GgmlType::IQ1_S: return 256;
        case GgmlType::IQ4_NL: return 32;
        case GgmlType::IQ3_S: return 256;
        case GgmlType::IQ2_S: return 256;
        case GgmlType::IQ4_XS: return 256;
        case GgmlType::IQ1_M: return 256;
        default: return 1;
    }
}

size_t ggml_type_type_size(GgmlType type) {
    switch (type) {
        case GgmlType::F32: return 4;
        case GgmlType::F16: return 2;
        case GgmlType::BF16: return 2;
        case GgmlType::I8: return 1;
        case GgmlType::I16: return 2;
        case GgmlType::I32: return 4;
        case GgmlType::I64: return 8;
        case GgmlType::F64: return 8;
        case GgmlType::Q4_0: return 18; // 2 bytes f16 scale + 16 bytes nibbles
        case GgmlType::Q4_1: return 20; // 2 bytes f16 scale + 2 bytes f16 min + 16 bytes nibbles
        case GgmlType::Q5_0: return 22;
        case GgmlType::Q5_1: return 24;
        case GgmlType::Q8_0: return 34;
        case GgmlType::Q8_1: return 36;
        case GgmlType::Q2_K: return 256 / 16 * 4 + 256 / 4;
        case GgmlType::Q3_K: return 256 / 8 + 256 / 4 + 12;
        case GgmlType::Q4_K: return 2 * sizeof(uint16_t) + 256 / 2 + 12;
        case GgmlType::Q5_K: return 2 * sizeof(uint16_t) + 256 / 2 + 256 / 8 + 12;
        case GgmlType::Q6_K: return sizeof(uint16_t) + 256 / 2 + 256 / 4 + 256 / 16;
        case GgmlType::Q8_K: return 256 + sizeof(float);
        default: return 1;
    }
}

enum class GgufMetadataValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

std::string read_string(std::ifstream& f) {
    uint64_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!f || len > 1000000) return {};
    std::string s(len, '\0');
    f.read(&s[0], len);
    return s;
}

void skip_value(std::ifstream& f, GgufMetadataValueType type) {
    switch (type) {
        case GgufMetadataValueType::UINT8:
        case GgufMetadataValueType::INT8:
        case GgufMetadataValueType::BOOL:
            f.seekg(1, std::ios::cur);
            break;
        case GgufMetadataValueType::UINT16:
        case GgufMetadataValueType::INT16:
            f.seekg(2, std::ios::cur);
            break;
        case GgufMetadataValueType::UINT32:
        case GgufMetadataValueType::INT32:
        case GgufMetadataValueType::FLOAT32:
            f.seekg(4, std::ios::cur);
            break;
        case GgufMetadataValueType::UINT64:
        case GgufMetadataValueType::INT64:
        case GgufMetadataValueType::FLOAT64:
            f.seekg(8, std::ios::cur);
            break;
        case GgufMetadataValueType::STRING: {
            uint64_t len = 0;
            f.read(reinterpret_cast<char*>(&len), sizeof(len));
            f.seekg(len, std::ios::cur);
            break;
        }
        case GgufMetadataValueType::ARRAY: {
            uint32_t elem_type_raw = 0;
            f.read(reinterpret_cast<char*>(&elem_type_raw), sizeof(elem_type_raw));
            uint64_t array_len = 0;
            f.read(reinterpret_cast<char*>(&array_len), sizeof(array_len));
            auto elem_type = static_cast<GgufMetadataValueType>(elem_type_raw);
            for (uint64_t i = 0; i < array_len; ++i) {
                skip_value(f, elem_type);
            }
            break;
        }
    }
}

struct TensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    std::vector<uint64_t> dims;
    GgmlType type = GgmlType::UNKNOWN;
    uint64_t offset = 0;
    uint64_t byte_size = 0;
    bool xdna_capable = false;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: xdna-gguf-info <model.gguf>\n";
        return 1;
    }

    const std::string path = argv[1];
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Error: cannot open file " << path << '\n';
        return 1;
    }

    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "GGUF") {
        std::cerr << "Error: not a valid GGUF file (bad magic: " << std::string(magic, 4) << ")\n";
        return 1;
    }

    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    uint64_t tensor_count = 0;
    f.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    uint64_t metadata_kv_count = 0;
    f.read(reinterpret_cast<char*>(&metadata_kv_count), sizeof(metadata_kv_count));

    std::cout << "================================================================================\n";
    std::cout << " xdna.cpp GGUF Model Analysis\n";
    std::cout << "================================================================================\n";
    std::cout << "File:                 " << path << '\n';
    std::cout << "GGUF Version:         " << version << '\n';
    std::cout << "Tensor Count:         " << tensor_count << '\n';
    std::cout << "Metadata Count:       " << metadata_kv_count << '\n';

    std::string arch = "unknown";
    uint32_t block_count = 0;
    uint32_t embedding_length = 0;
    uint32_t feed_forward_length = 0;
    uint32_t head_count = 0;
    uint32_t head_count_kv = 0;

    // Read metadata key-values
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        std::string key = read_string(f);
        uint32_t val_type_raw = 0;
        f.read(reinterpret_cast<char*>(&val_type_raw), sizeof(val_type_raw));
        auto val_type = static_cast<GgufMetadataValueType>(val_type_raw);

        if (key == "general.architecture" && val_type == GgufMetadataValueType::STRING) {
            arch = read_string(f);
        } else if ((key == arch + ".block_count" || key == "general.block_count") &&
                   val_type == GgufMetadataValueType::UINT32) {
            f.read(reinterpret_cast<char*>(&block_count), sizeof(block_count));
        } else if ((key == arch + ".embedding_length" || key == "general.embedding_length") &&
                   val_type == GgufMetadataValueType::UINT32) {
            f.read(reinterpret_cast<char*>(&embedding_length), sizeof(embedding_length));
        } else if ((key == arch + ".feed_forward_length" || key == "general.feed_forward_length") &&
                   val_type == GgufMetadataValueType::UINT32) {
            f.read(reinterpret_cast<char*>(&feed_forward_length), sizeof(feed_forward_length));
        } else if ((key == arch + ".attention.head_count") &&
                   val_type == GgufMetadataValueType::UINT32) {
            f.read(reinterpret_cast<char*>(&head_count), sizeof(head_count));
        } else if ((key == arch + ".attention.head_count_kv") &&
                   val_type == GgufMetadataValueType::UINT32) {
            f.read(reinterpret_cast<char*>(&head_count_kv), sizeof(head_count_kv));
        } else {
            skip_value(f, val_type);
        }
    }

    std::cout << "\nArchitecture Info:\n";
    std::cout << "  Architecture:       " << arch << '\n';
    if (block_count) std::cout << "  Layers (blocks):    " << block_count << '\n';
    if (embedding_length) std::cout << "  Hidden Dimension:   " << embedding_length << '\n';
    if (feed_forward_length) std::cout << "  FFN Intermediate:   " << feed_forward_length << '\n';
    if (head_count) std::cout << "  Attention Heads:    " << head_count << '\n';
    if (head_count_kv) std::cout << "  KV Heads:           " << head_count_kv << '\n';

    std::vector<TensorInfo> tensors;
    tensors.reserve(tensor_count);

    uint64_t total_model_bytes = 0;
    uint64_t xdna_capable_bytes = 0;
    uint64_t cpu_only_bytes = 0;
    uint32_t xdna_tensor_count = 0;
    uint32_t q4_0_tensor_count = 0;

    std::map<std::string, uint32_t> type_counts;
    std::map<std::string, uint64_t> type_bytes;
    std::map<std::string, uint32_t> shape_histogram;

    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo t;
        t.name = read_string(f);
        f.read(reinterpret_cast<char*>(&t.n_dims), sizeof(t.n_dims));
        t.dims.resize(t.n_dims);
        for (uint32_t d = 0; d < t.n_dims; ++d) {
            f.read(reinterpret_cast<char*>(&t.dims[d]), sizeof(uint64_t));
        }
        uint32_t type_raw = 0;
        f.read(reinterpret_cast<char*>(&type_raw), sizeof(type_raw));
        t.type = static_cast<GgmlType>(type_raw);
        f.read(reinterpret_cast<char*>(&t.offset), sizeof(t.offset));

        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < t.n_dims; ++d) {
            n_elements *= t.dims[d];
        }

        size_t bs = ggml_type_block_size(t.type);
        size_t ts = ggml_type_type_size(t.type);
        t.byte_size = (n_elements / bs) * ts;

        // XDNA capability check:
        // V0 supports Q4_0 2D matrix projections where dimensions are multiples of 32
        if (t.type == GgmlType::Q4_0 && t.n_dims == 2 && (t.dims[0] % 32 == 0)) {
            t.xdna_capable = true;
            xdna_capable_bytes += t.byte_size;
            xdna_tensor_count++;
        } else {
            t.xdna_capable = false;
            cpu_only_bytes += t.byte_size;
        }

        if (t.type == GgmlType::Q4_0) {
            q4_0_tensor_count++;
        }

        std::string type_name = ggml_type_name(t.type);
        type_counts[type_name]++;
        type_bytes[type_name] += t.byte_size;

        if (t.n_dims == 2) {
            std::string shape_str = std::to_string(t.dims[0]) + " x " + std::to_string(t.dims[1]);
            shape_histogram[shape_str]++;
        }

        total_model_bytes += t.byte_size;
        tensors.push_back(t);
    }

    std::cout << "\nTensor Inventory Summary:\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Type" 
              << std::setw(12) << "Count" 
              << std::setw(18) << "Bytes" 
              << std::setw(12) << "Size (GB)" 
              << "Percentage\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    for (const auto& [type_name, count] : type_counts) {
        uint64_t bytes = type_bytes[type_name];
        double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        double pct = (static_cast<double>(bytes) / static_cast<double>(total_model_bytes)) * 100.0;
        std::cout << std::left << std::setw(15) << type_name 
                  << std::setw(12) << count 
                  << std::setw(18) << bytes 
                  << std::fixed << std::setprecision(2) << std::setw(12) << gb 
                  << std::setprecision(1) << pct << " %\n";
    }

    std::cout << "\nMajor 2D Matrix Projections (Shapes):\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    for (const auto& [shape_str, count] : shape_histogram) {
        std::cout << "  " << std::left << std::setw(25) << shape_str << " : " << count << " tensors\n";
    }

    std::cout << "\nFirst 20 Tensors Detail:\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(42) << "Tensor Name"
              << std::setw(10) << "Type"
              << std::setw(20) << "Dimensions"
              << std::setw(12) << "Size (MB)"
              << "Target\n";
    std::cout << "--------------------------------------------------------------------------------\n";
    for (size_t i = 0; i < std::min<size_t>(20, tensors.size()); ++i) {
        const auto& t = tensors[i];
        std::string dim_str;
        for (size_t d = 0; d < t.dims.size(); ++d) {
            dim_str += std::to_string(t.dims[d]);
            if (d + 1 < t.dims.size()) dim_str += " x ";
        }
        double mb = static_cast<double>(t.byte_size) / (1024.0 * 1024.0);
        std::cout << std::left << std::setw(42) << t.name
                  << std::setw(10) << ggml_type_name(t.type)
                  << std::setw(20) << dim_str
                  << std::fixed << std::setprecision(2) << std::setw(12) << mb
                  << (t.xdna_capable ? "XDNA (NPU)" : "CPU") << '\n';
    }
    if (tensors.size() > 20) {
        std::cout << "  ... [" << (tensors.size() - 20) << " more tensors]\n";
    }

    std::cout << "\n================================================================================\n";
    std::cout << " XDNA Backend Offload Assessment\n";
    std::cout << "================================================================================\n";
    double total_gb = static_cast<double>(total_model_bytes) / (1024.0 * 1024.0 * 1024.0);
    double xdna_gb = static_cast<double>(xdna_capable_bytes) / (1024.0 * 1024.0 * 1024.0);
    double cpu_gb = static_cast<double>(cpu_only_bytes) / (1024.0 * 1024.0 * 1024.0);
    double offload_pct = (static_cast<double>(xdna_capable_bytes) / static_cast<double>(total_model_bytes)) * 100.0;

    std::cout << "Total Model Weight Size:  " << std::fixed << std::setprecision(2) << total_gb << " GB (" << total_model_bytes << " bytes)\n";
    std::cout << "XDNA-Offloadable Weights: " << xdna_gb << " GB (" << xdna_capable_bytes << " bytes)\n";
    std::cout << "CPU-Only Fallback:        " << cpu_gb << " GB (" << cpu_only_bytes << " bytes)\n";
    std::cout << "Offloadable Tensors:      " << xdna_tensor_count << " / " << tensor_count << " tensors\n";
    std::cout << "Potential Weight Offload: " << std::fixed << std::setprecision(2) << offload_pct << " %\n";
    std::cout << "================================================================================\n";

    return 0;
}
