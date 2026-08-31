#include "llama.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct ShapeRecord {
    std::string tensor_name;
    int64_t N;
    int64_t K;
    size_t weight_bytes;
    int invocations = 0;
};

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : "/home/satvik/FORK/Qwen2.5-0.5B-Instruct-Q4_0.gguf";

    std::cout << "========================================================================================================\n";
    std::cout << " QWEN2.5-0.5B SHAPE INVENTORY DUMP\n";
    std::cout << " Model: " << model_path << "\n";
    std::cout << "========================================================================================================\n";

    llama_backend_init();
    ggml_backend_load_all_from_path("/home/satvik/FORK/llama.cpp/build/bin");

    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 512;
    cparams.n_threads = 4;

    llama_context* ctx = llama_init_from_model(model, cparams);

    // Warmup 1 token
    llama_token tok = 100;
    llama_batch batch = llama_batch_get_one(&tok, 1);
    llama_decode(ctx, batch);

    // Inventory of all 24 layers of Qwen2.5-0.5B
    // In Qwen2.5-0.5B (24 layers):
    // blk.i.attn_q.weight: 896 x 896
    // blk.i.attn_k.weight: 128 x 896
    // blk.i.attn_v.weight: 128 x 896
    // blk.i.attn_output.weight: 896 x 896
    // blk.i.ffn_gate.weight: 4864 x 896
    // blk.i.ffn_up.weight: 4864 x 896
    // blk.i.ffn_down.weight: 896 x 4864
    // token_embd.weight: 151936 x 896
    // output.weight: 151936 x 896

    std::map<std::pair<int64_t, int64_t>, std::vector<std::string>> shape_map;

    shape_map[{896, 896}] = {"blk.*.attn_q.weight (24x)", "blk.*.attn_output.weight (24x)"};
    shape_map[{128, 896}] = {"blk.*.attn_k.weight (24x)", "blk.*.attn_v.weight (24x)"};
    shape_map[{4864, 896}] = {"blk.*.ffn_gate.weight (24x)", "blk.*.ffn_up.weight (24x)"};
    shape_map[{896, 4864}] = {"blk.*.ffn_down.weight (24x)"};
    shape_map[{151936, 896}] = {"output.weight (1x)"};

    std::cout << std::left << std::setw(18) << "Shape (NxK)"
              << std::setw(10) << "K-Dim"
              << std::setw(12) << "Size (KB)"
              << std::setw(10) << "Count"
              << std::setw(14) << "Total MB"
              << "Associated Projections\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    size_t total_ops = 0;
    size_t total_bytes = 0;

    for (const auto& kv : shape_map) {
        int64_t N = kv.first.first;
        int64_t K = kv.first.second;
        size_t bpr = (K / 32) * 18;
        size_t tensor_bytes = N * bpr;
        size_t count = (N == 151936) ? 1 : ((N == 4864 || N == 896 && K == 896 || N == 128) ? 48 : 24);
        if (N == 896 && K == 4864) count = 24;

        total_ops += count;
        total_bytes += count * tensor_bytes;

        std::string desc = "";
        for (const auto& s : kv.second) desc += s + " ";

        std::cout << std::left << std::setw(18) << (std::to_string(N) + "x" + std::to_string(K))
                  << std::setw(10) << K
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << (tensor_bytes / 1024.0)
                  << std::setw(10) << count
                  << std::setw(14) << ((count * tensor_bytes) / (1024.0 * 1024.0))
                  << desc << "\n";
    }

    std::cout << "--------------------------------------------------------------------------------------------------------\n";
    std::cout << " Total Decode Ops:   " << total_ops << " matrices\n";
    std::cout << " Total Weight Footprint: " << (total_bytes / (1024.0 * 1024.0)) << " MiB\n";
    std::cout << "========================================================================================================\n";

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
