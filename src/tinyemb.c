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

int tinyemb_max_seq(TinyEmb* h) {
    return h ? h->model.cfg.max_seq : 0;
}

int tinyemb_set_threads(TinyEmb* h, int n_threads) {
    if (!h) return 0;
    h->model.cfg.n_threads = n_threads;
    return n_threads;
}

// 内部:文本 → id 序列(带 [CLS]/[SEP]),返回 token 数,-1 失败。
// truncate=1 超长自动截断(保证 n <= max_ids),=0 超长报 -1。
static int tokenize_to(TinyEmb* h, const char* text, int* ids, int max_ids,
                       int truncate, size_t* n_out) {
    size_t n = 0;
    if (wp_encode(&h->wp, text, ids, &n, (size_t)max_ids, truncate) != 0) return -1;
    *n_out = n;
    return 0;
}

// 单条文本编码,truncate 由调用方决定。超长按模型 max_seq 截断。
int tinyemb_encode_text_ex(TinyEmb* h, const char* text, int truncate,
                           float* out, int out_cap) {
    if (!h || !text || !out) return -1;
    int dim = h->model.cfg.hidden_dim;
    if (out_cap < dim) return -1;

    int max_seq = h->model.cfg.max_seq;
    int* ids = malloc((size_t)max_seq * sizeof(int));
    if (!ids) return -1;

    size_t n = 0;
    if (tokenize_to(h, text, ids, max_seq, truncate, &n) != 0) {
        free(ids);
        return -1;
    }
    int rc = bert_encode(&h->model, ids, (int)n, out);
    free(ids);
    return rc;
}

int tinyemb_encode_text(TinyEmb* h, const char* text, float* out, int out_cap) {
    // 默认 truncate=true(对齐 HF truncation=True)
    return tinyemb_encode_text_ex(h, text, 1, out, out_cap);
}

int tinyemb_encode_ids(TinyEmb* h, const int* ids, int n_ids, float* out, int out_cap) {
    if (!h || !ids || !out) return -1;
    if (out_cap < h->model.cfg.hidden_dim) return -1;
    // id 入口 = 调用方负责长度,超长直接报错(保留护栏,不静默截断)
    return bert_encode(&h->model, ids, n_ids, out);
}

int tinyemb_encode_texts(TinyEmb* h, const char** texts, int n_texts, float* out, int out_cap) {
    // 默认 truncate=true
    return tinyemb_encode_texts_ex(h, texts, n_texts, 1, out, out_cap);
}

int tinyemb_encode_texts_ex(TinyEmb* h, const char** texts, int n_texts, int truncate,
                            float* out, int out_cap) {
    if (!h || !texts || !out || n_texts <= 0) return -1;
    int dim = h->model.cfg.hidden_dim;
    if (out_cap < dim * n_texts) return -1;

    int max_seq = h->model.cfg.max_seq;

    // 分词每句,拼接 id + 记录每句长度。每句上限 = max_seq(截断后恒不超)
    int cap = max_seq * n_texts;
    int* ids = malloc((size_t)cap * sizeof(int));
    int* lens = malloc((size_t)n_texts * sizeof(int));
    if (!ids || !lens) { free(ids); free(lens); return -1; }

    int total = 0;
    for (int s = 0; s < n_texts; s++) {
        size_t n = 0;
        // 每句最多占 max_seq 个位置,剩余空间足够(不会再被截)
        if (tokenize_to(h, texts[s], ids + total, max_seq, truncate, &n) != 0) {
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
    int max_seq = h->model.cfg.max_seq;
    int* ids = malloc((size_t)max_seq * sizeof(int));
    if (!ids) return 0;

    size_t n = 0;
    // truncate=1:报告截断后的真实 token 数(=实际参与推理的长度)
    if (tokenize_to(h, text, ids, max_seq, 1, &n) != 0) {
        free(ids);
        return 0;
    }
    free(ids);
    return (int)n;
}
