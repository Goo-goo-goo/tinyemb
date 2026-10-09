// tinyemb 共享库导出实现:包装 bert_model + tokenizer
#include "io/tinyemb.h"
#include "io/bert_model.h"
#include "io/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct TinyEmb {
    BertModel model;
    WordPiece  wp;
};

TinyEmb* tinyemb_load(const char* model_dir) {
    TinyEmb* h = calloc(1, sizeof(TinyEmb));
    if (!h) return NULL;

    if (bert_load(&h->model, model_dir) != 0) {
        free(h);
        return NULL;
    }

    // 词表路径 = 模型目录/vocab.txt
    char vocab[1024];
    snprintf(vocab, sizeof(vocab), "%s/vocab.txt", model_dir);
    if (wp_load(&h->wp, vocab) != 0) {
        bert_free(&h->model);
        free(h);
        return NULL;
    }
    return h;
}

void tinyemb_free(TinyEmb* h) {
    if (!h) return;
    bert_free(&h->model);
    wp_free(&h->wp);
    free(h);
}

int tinyemb_dim(TinyEmb* h) {
    return h ? h->model.cfg.hidden_dim : 0;
}

int tinyemb_set_threads(TinyEmb* h, int n_threads) {
    if (!h) return 0;
    h->model.cfg.n_threads = n_threads;
    return n_threads;
}

// 内部:文本 → id 序列(带 [CLS]/[SEP]),返回 token 数,-1 失败
static int tokenize_to(TinyEmb* h, const char* text, int* ids, int max_ids, size_t* n_out) {
    size_t n = 0;
    if (wp_encode(&h->wp, text, ids, &n, (size_t)max_ids) != 0) return -1;
    *n_out = n;
    return 0;
}

int tinyemb_encode_text(TinyEmb* h, const char* text, float* out, int out_cap) {
    if (!h || !text || !out) return -1;
    int dim = h->model.cfg.hidden_dim;
    if (out_cap < dim) return -1;

    int ids[1024];
    size_t n = 0;
    if (tokenize_to(h, text, ids, 1024, &n) != 0) return -1;

    return bert_encode(&h->model, ids, (int)n, out);
}

int tinyemb_encode_ids(TinyEmb* h, const int* ids, int n_ids, float* out, int out_cap) {
    if (!h || !ids || !out) return -1;
    if (out_cap < h->model.cfg.hidden_dim) return -1;
    return bert_encode(&h->model, ids, n_ids, out);
}

int tinyemb_encode_texts(TinyEmb* h, const char** texts, int n_texts, float* out, int out_cap) {
    if (!h || !texts || !out || n_texts <= 0) return -1;
    int dim = h->model.cfg.hidden_dim;
    if (out_cap < dim * n_texts) return -1;

    // 分词每句,拼接 id + 记录每句长度
    int cap = 1024 * n_texts > 4096 ? 1024 * n_texts : 4096;
    int* ids = malloc(cap * sizeof(int));
    int* lens = malloc(n_texts * sizeof(int));
    int total = 0;
    for (int s = 0; s < n_texts; s++) {
        size_t n = 0;
        // 只取每个文本的分词,写到 ids+total
        if (wp_encode(&h->wp, texts[s], ids + total, &n, (size_t)(cap - total)) != 0) {
            free(ids); free(lens); return -1;
        }
        lens[s] = (int)n;
        total += (int)n;
    }
    int rc = bert_encode_batch(&h->model, ids, lens, n_texts, out);
    free(ids); free(lens);
    return rc;
}

int tinyemb_token_count(TinyEmb* h, const char* text) {
    if (!h || !text) return 0;
    int ids[1024];
    size_t n = 0;
    if (tokenize_to(h, text, ids, 1024, &n) != 0) return 0;
    return (int)n;
}
