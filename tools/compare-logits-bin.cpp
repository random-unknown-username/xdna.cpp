#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <cpu_logits.bin> <xdna_logits.bin>\n";
        return 1;
    }

    std::string cpu_path = argv[1];
    std::string xdna_path = argv[2];

    std::ifstream cpu_f(cpu_path, std::ios::binary);
    std::ifstream xdna_f(xdna_path, std::ios::binary);

    if (!cpu_f || !xdna_f) {
        std::cerr << "Failed to open input binary files\n";
        return 1;
    }

    int32_t cpu_header[2], xdna_header[2];
    cpu_f.read(reinterpret_cast<char*>(cpu_header), sizeof(cpu_header));
    xdna_f.read(reinterpret_cast<char*>(xdna_header), sizeof(xdna_header));

    int n_tokens = cpu_header[0];
    int n_vocab = cpu_header[1];

    if (xdna_header[0] != n_tokens || xdna_header[1] != n_vocab) {
        std::cerr << "Header mismatch: CPU (" << cpu_header[0] << "x" << cpu_header[1]
                  << ") vs XDNA (" << xdna_header[0] << "x" << xdna_header[1] << ")\n";
        return 1;
    }

    std::cout << "========================================================================================================\n";
    std::cout << " STANDALONE BINARY LOGIT COMPARISON (CPU vs XDNA)\n";
    std::cout << " CPU File:  " << cpu_path << "\n";
    std::cout << " XDNA File: " << xdna_path << "\n";
    std::cout << " Shape:     " << n_tokens << " tokens x " << n_vocab << " vocab\n";
    std::cout << "========================================================================================================\n";

    std::vector<float> cpu_logits(n_vocab), xdna_logits(n_vocab);
    double sum_cosine = 0.0;
    double sum_rel_l2 = 0.0;
    float max_abs_diff_all = 0.0f;
    int top1_matches = 0;
    int top5_overlap_total = 0;

    std::cout << std::left << std::setw(8) << "Token"
              << std::setw(18) << "Cosine Sim"
              << std::setw(18) << "Relative L2"
              << std::setw(18) << "Max Abs Diff"
              << std::setw(14) << "Top-1 Match"
              << "Top-5 Overlap\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    for (int t = 0; t < n_tokens; ++t) {
        cpu_f.read(reinterpret_cast<char*>(cpu_logits.data()), n_vocab * sizeof(float));
        xdna_f.read(reinterpret_cast<char*>(xdna_logits.data()), n_vocab * sizeof(float));

        double dot = 0.0, n_cpu = 0.0, n_xdna = 0.0, l2_diff = 0.0;
        float max_d = 0.0f;
        std::vector<std::pair<float, int>> top_c(n_vocab), top_x(n_vocab);

        for (int i = 0; i < n_vocab; ++i) {
            float c = cpu_logits[i];
            float x = xdna_logits[i];
            float d = std::abs(c - x);
            if (d > max_d) max_d = d;
            l2_diff += (c - x) * (c - x);
            dot += static_cast<double>(c) * x;
            n_cpu += static_cast<double>(c) * c;
            n_xdna += static_cast<double>(x) * x;

            top_c[i] = {c, i};
            top_x[i] = {x, i};
        }

        if (max_d > max_abs_diff_all) max_abs_diff_all = max_d;
        double rel_l2 = (n_cpu > 1e-12) ? std::sqrt(l2_diff / n_cpu) : 0.0;
        double denom = std::sqrt(n_cpu) * std::sqrt(n_xdna);
        double cosine = (denom > 1e-12) ? (dot / denom) : 0.0;

        sum_cosine += cosine;
        sum_rel_l2 += rel_l2;

        std::partial_sort(top_c.begin(), top_c.begin() + 5, top_c.end(), std::greater<>());
        std::partial_sort(top_x.begin(), top_x.begin() + 5, top_x.end(), std::greater<>());

        bool t1 = (top_c[0].second == top_x[0].second);
        if (t1) top1_matches++;

        int overlap = 0;
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                if (top_c[i].second == top_x[j].second) {
                    overlap++;
                    break;
                }
            }
        }
        top5_overlap_total += overlap;

        if (t < 8 || t == n_tokens - 1) {
            std::cout << std::left << std::setw(8) << t
                      << std::fixed << std::setprecision(8)
                      << std::setw(18) << cosine
                      << std::setw(18) << rel_l2
                      << std::setprecision(6)
                      << std::setw(18) << max_d
                      << std::setw(14) << (t1 ? "YES" : "NO")
                      << (std::to_string(overlap) + " / 5") << "\n";
            if (t == 7 && n_tokens > 9) {
                std::cout << "  ... [" << (n_tokens - 9) << " intermediate tokens omitted] ...\n";
            }
        }
    }

    double avg_cosine = sum_cosine / n_tokens;
    double avg_rel_l2 = sum_rel_l2 / n_tokens;
    double top1_pct = (static_cast<double>(top1_matches) / n_tokens) * 100.0;
    double top5_pct = (static_cast<double>(top5_overlap_total) / (n_tokens * 5)) * 100.0;

    std::cout << "========================================================================================================\n";
    std::cout << " SUMMARY ACROSS ALL " << n_tokens << " TOKENS:\n";
    std::cout << "   - Average Cosine Similarity:  " << std::fixed << std::setprecision(8) << avg_cosine << "\n";
    std::cout << "   - Average Relative L2 Error:  " << std::setprecision(8) << avg_rel_l2 << "\n";
    std::cout << "   - Maximum Absolute Delta:     " << std::setprecision(6) << max_abs_diff_all << "\n";
    std::cout << "   - Top-1 Token Agreement:      " << std::setprecision(2) << top1_pct << " % (" << top1_matches << "/" << n_tokens << ")\n";
    std::cout << "   - Top-5 Token Overlap Rate:   " << top5_pct << " %\n";
    std::cout << "========================================================================================================\n";

    return 0;
}
