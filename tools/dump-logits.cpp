#include "llama.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <output_bin> <n_tokens>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string output_bin = argv[2];
    int n_tokens = std::stoi(argv[3]);

    llama_backend_init();
    ggml_backend_load_all_from_path("/home/satvik/FORK/llama.cpp/build/bin");

    llama_model_params mparams = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_tokens + 16;
    cparams.n_batch = 512;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to initialize llama context\n";
        return 1;
    }

    std::ofstream out(output_bin, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file: " << output_bin << "\n";
        return 1;
    }

    // Write file header: [n_tokens: int32, n_vocab: int32]
    int32_t header[2] = {n_tokens, n_vocab};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));

    // Initial prompt token
    llama_token current_token = 100; // 'Hi'
    llama_batch batch = llama_batch_get_one(&current_token, 1);
    llama_decode(ctx, batch);

    for (int t = 0; t < n_tokens; ++t) {
        float* logits = llama_get_logits_ith(ctx, 0);
        out.write(reinterpret_cast<const char*>(logits), n_vocab * sizeof(float));

        // Use deterministic fixed input tokens to keep context sequence identical
        llama_token next_token = 1000 + t;
        batch = llama_batch_get_one(&next_token, 1);
        llama_decode(ctx, batch);
    }

    out.close();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    std::cout << "Successfully dumped " << n_tokens << " tokens (" << n_vocab << " vocab) to " << output_bin << "\n";
    return 0;
}
