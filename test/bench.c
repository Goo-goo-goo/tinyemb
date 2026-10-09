// 基准测试:测单条编码延迟 + 吞吐
// 用法:./bench [次数] [句子]
#include "io/tinyemb.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// 简单冒泡排序求中位数(基准数据量小)
static int cmp(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 100;
    const char* text = (argc > 2) ? argv[2] : "北京的天气不错";
    int n_threads = (argc > 3) ? atoi(argv[3]) : 0;   // 0=单线程

    double t0 = now_ms();
    TinyEmb* h = tinyemb_load("models/bge-small-zh-v1.5");
    if (!h) { fprintf(stderr, "加载失败\n"); return 1; }
    double t_load = now_ms() - t0;
    tinyemb_set_threads(h, n_threads);

    int dim = tinyemb_dim(h);
    float* vec = malloc(dim * sizeof(float));

    // 预热(排除首次冷启动影响)
    for (int i = 0; i < 5; i++) tinyemb_encode_text(h, text, vec, dim);

    // 测量
    double* lat = malloc(N * sizeof(double));
    double t_start = now_ms();
    for (int i = 0; i < N; i++) {
        double s = now_ms();
        tinyemb_encode_text(h, text, vec, dim);
        lat[i] = now_ms() - s;
    }
    double t_total = now_ms() - t_start;

    qsort(lat, N, sizeof(double), cmp);
    double p50 = lat[N / 2];
    double p99 = lat[(int)(N * 0.99) < N ? (int)(N * 0.99) : N - 1];
    double mean = t_total / N;

    printf("=== tinyemb 基准 ===\n");
    printf("CPU: Apple M3 (4P+4E) | 文本: \"%s\" (%d token) | 线程: %d\n",
           text, tinyemb_token_count(h, text), n_threads);
    printf("模型加载: %.1f ms\n", t_load);
    printf("隐藏维度: %d\n", dim);
    printf("样本数: %d\n", N);
    printf("延迟 mean = %.3f ms | p50 = %.3f ms | p99 = %.3f ms\n", mean, p50, p99);
    printf("单线程吞吐: %.1f sent/s\n", 1000.0 / mean);

    free(lat); free(vec);
    tinyemb_free(h);
    return 0;
}
