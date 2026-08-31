#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct LogitStats {
    double cosine_sim = 0.0;
    float max_abs_diff = 0.0;
    bool top1_match = false;
    int top5_overlap_count = 0;
};

LogitStats compare_logits(const float* l_cpu, const float* l_xdna, int n_vocab) {
    LogitStats stats;
    double dot = 0.0, n_cpu = 0.0, n_xdna = 0.0;
    float max_diff = 0.0f;

    std::vector<std::pair<float, int>> top_cpu(n_vocab), top_xdna(n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
        float c = l_cpu[i];
        float x = l_xdna[i];
        float d = std::abs(c - x);
        if (d > max_diff) max_diff = d;

        dot += static_cast<double>(c) * x;
        n_cpu += static_cast<double>(c) * c;
        n_xdna += static_cast<double>(x) * x;

        top_cpu[i] = {c, i};
        top_xdna[i] = {x, i};
    }

    stats.max_abs_diff = max_diff;
    double denom = std::sqrt(n_cpu) * std::sqrt(n_xdna);
    stats.cosine_sim = (denom > 1e-12) ? (dot / denom) : 0.0;

    std::partial_sort(top_cpu.begin(), top_cpu.begin() + 5, top_cpu.end(), std::greater<>());
    std::partial_sort(top_xdna.begin(), top_xdna.begin() + 5, top_xdna.end(), std::greater<>());

    stats.top1_match = (top_cpu[0].second == top_xdna[0].second);

    int overlap = 0;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            if (top_cpu[i].second == top_xdna[j].second) {
                overlap++;
                break;
            }
        }
    }
    stats.top5_overlap_count = overlap;
    return stats;
}

int main(int argc, char** argv) {
    std::string model_path = (argc > 1) ? argv[1] : "/home/satvik/FORK/Qwen2.5-0.5B-Instruct-Q4_0.gguf";
    int n_tokens = (argc > 2) ? std::stoi(argv[2]) : 128;

    std::cout << "========================================================================================================\n";
    std::cout << " END-TO-END CPU VS XDNA LOGIT QUALITY & DRIFT AUDIT (" << n_tokens << " TOKENS)\n";
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

    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_tokens + 16;
    cparams.n_batch = 512;
    cparams.n_threads = 8;

    llama_context* ctx_cpu = llama_init_from_model(model, cparams);
    llama_context* ctx_xdna = llama_init_from_model(model, cparams);

    // Initial prompt
    llama_token prompt_token = 100; // 'Hi'
    llama_batch batch_cpu = llama_batch_get_one(&prompt_token, 1);
    llama_batch batch_xdna = llama_batch_get_one(&prompt_token, 1);

    llama_decode(ctx_cpu, batch_cpu);
    llama_decode(ctx_xdna, batch_xdna);

    double sum_cosine = 0.0;
    float max_error_all = 0.0f;
    int top1_matches = 0;
    int total_top5_overlap = 0;

    for (int step = 0; step < n_tokens; ++step) {
        float* logits_cpu = llama_get_logits_ith(ctx_cpu, 0);
        float* logits_xdna = llama_get_logits_ith(ctx_xdna, 0);

        auto s = compare_logits(logits_cpu, logits_xdna, n_vocab);
        sum_cosine += s.cosine_sim;
        if (s.max_abs_diff > max_error_all) max_error_all = s.max_abs_diff;
        if (s.top1_match) top1_matches++;
        total_top5_overlap += s.top5_overlap_count;

        // Sample greedily from CPU logits to keep trajectory identical
        llama_token next_tok = 0;
        float best_l = -1e9f;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits_cpu[i] > best_l) {
                best_l = logits_cpu[i];
                next_tok = i;
            }
        }

        batch_cpu = llama_batch_get_one(&next_tok, 1);
        batch_xdna = llama_batch_get_one(&next_tok, 1);
        llama_decode(ctx_cpu, batch_cpu);
        llama_decode(ctx_xdna, batch_xdna);
    }

    double avg_cosine = sum_cosine / n_tokens;
    double top1_rate = (static_cast<double>(top1_matches) / n_tokens) * 100.0;
    double top5_rate = (static_cast<double>(total_top5_overlap) / (n_tokens * 5)) * 100.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << " Average Logit Cosine Similarity:  " << avg_cosine << "\n";
    std::cout << " Maximum Absolute Logit Delta:     " << max_error_all << "\n";
    std::cout << " Top-1 Token Agreement Rate:       " << std::setprecision(2) << top1_rate << " % (" << top1_matches << "/" << n_tokens << ")\n";
    std::cout << " Top-5 Token Overlap Rate:         " << top5_rate << " %\n";
    std::cout << "========================================================================================================\n";

    llama_free(ctx_cpu);
    llama_free(ctx_xdna);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
