// M5b 集成测试:完整 BERT 推理,和 Python transformers 对拍
// 基准来源:.venv/bin/python3 scripts/ref_embed.py
#include "io/bert_model.h"
#include "io/tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int feq(float a, float b, float eps) {
    float d = a - b; if (d < 0) d = -d;
    return d < eps;
}

int main(int argc, char** argv) {
    const char* dir = "models/bge-small-zh-v1.5";
    const char* text = "北京的天气不错";
    if (argc > 1) text = argv[1];

    // 加载
    BertModel m;
    if (bert_load(&m, dir) != 0) {
        fprintf(stderr, "加载模型失败\n");
        return 1;
    }
    printf("模型加载 OK: %d 层, hidden=%d, heads=%d\n",
           m.cfg.num_layers, m.cfg.hidden_dim, m.cfg.num_heads);

    // 分词
    WordPiece wp;
    if (wp_load(&wp, "models/bge-small-zh-v1.5/vocab.txt") != 0) return 1;
    int ids[512];
    size_t n_ids = 0;
    wp_encode(&wp, text, ids, &n_ids, 512);
    printf("tokens(%zu):", n_ids);
    for (size_t i = 0; i < n_ids; i++) printf(" %d", ids[i]);
    printf("\n");

    // 推理
    float* vec = malloc(m.cfg.hidden_dim * sizeof(float));
    if (bert_encode(&m, ids, (int)n_ids, vec) != 0) {
        fprintf(stderr, "推理失败\n");
        return 1;
    }

    // 输出前 8 个分量 + L2 长度(应为 1)
    printf("前 8 个分量:");
    for (int i = 0; i < 8; i++) printf(" %.8f", vec[i]);
    printf("\n");
    float sum2 = 0;
    for (int d = 0; d < m.cfg.hidden_dim; d++) sum2 += vec[d]*vec[d];
    printf("L2 长度 = %.8f (应≈1.0)\n", sqrtf(sum2));

    // 若提供了基准向量文件,逐分量对拍
    if (argc > 2) {
        FILE* f = fopen(argv[2], "r");
        if (!f) { fprintf(stderr, "打不开基准 %s\n", argv[2]); return 1; }
        int bad = 0;
        float max_err = 0;
        for (int d = 0; d < m.cfg.hidden_dim; d++) {
            float ref;
            if (fscanf(f, "%f", &ref) != 1) { fprintf(stderr, "基准读取失败\n"); return 1; }
            float e = fabsf(vec[d] - ref);
            if (e > max_err) max_err = e;
            if (!feq(vec[d], ref, 1e-5f)) bad++;
        }
        fclose(f);
        printf("对拍: %d/%d 分量超出 1e-5 误差, 最大误差 = %.8f\n",
               bad, m.cfg.hidden_dim, max_err);
        if (bad == 0) printf("[PASS] 和 Python 基准一致\n");
        else         printf("[FAIL] 有分量不一致\n");
    }

    free(vec);
    wp_free(&wp);
    bert_free(&m);
    return 0;
}
