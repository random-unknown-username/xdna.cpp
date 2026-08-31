#include "llama.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct HighResTimer {
    std::chrono::high_resolution_clock::time_point start_t;
    void start() { start_t = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        auto end_t = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_t - start_t).count();
    }
};

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : "/home/satvik/FORK/Qwen3.8-27B-Q4_0.gguf";

    std::cout << "========================================================================================================\n";
    std::cout << " COMPREHENSIVE 595 ms TOKEN BREAKDOWN INSTRUMENTATION\n";
    std::cout << " Model: " << model_path << "\n";
    std::cout << "========================================================================================================\n";

    llama_backend_init();
    ggml_backend_load_all_from_path("/home/satvik/FORK/llama.cpp/build/bin");

    llama_model_params mparams = llama_model_default_params();

    HighResTimer timer;
    timer.start();
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }
    double load_time_ms = timer.elapsed_ms();
    std::cout << "Model Loaded in " << std::fixed << std::setprecision(2) << load_time_ms << " ms\n";

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to create llama context\n";
        return 1;
    }

    // Warmup 1 token
    llama_token token = 100; // 'Hi'
    llama_batch batch = llama_batch_get_one(&token, 1);
    llama_decode(ctx, batch);

    std::cout << "Warmup complete. Profiling warm decode tokens...\n\n";

    // Profile 4 warm decode tokens
    int n_warm_tokens = 4;
    double total_decode_wall_ms = 0.0;

    for (int t = 0; t < n_warm_tokens; ++t) {
        token = 100 + t;
        batch = llama_batch_get_one(&token, 1);

        timer.start();
        llama_decode(ctx, batch);
        double decode_ms = timer.elapsed_ms();
        total_decode_wall_ms += decode_ms;
    }

    double avg_token_ms = total_decode_wall_ms / n_warm_tokens;
    double tok_per_sec = 1000.0 / avg_token_ms;

    std::cout << "========================================================================================================\n";
    std::cout << " WARM DECODE TOKEN LATENCY: " << std::fixed << std::setprecision(2) << avg_token_ms << " ms ("
              << std::setprecision(2) << tok_per_sec << " tok/s)\n";
    std::cout << "========================================================================================================\n";

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
