// M5b:完整 BERT 前向实现
// ①嵌入层 → ②堆 N 层 Encoder → ③CLS 池化 → ④L2 归一化
#include "io/bert_model.h"
#include "io/tokenizer.h"
#include "ggml.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- 内部:取 loader 里某个张量的 float 指针 ---- */
// 为了零拷贝,我们不把张量数据搬走,而是直接读到模型自己的缓冲区。
// 这里简单起见:每次 read_floats 拷一份到模型内存,由 bert_free 统一释放。
static float* load_tensor(BertModel* m, const char* name, size_t expect_numel) {
    size_t n = 0;
    float* p = st_read_floats(&m->loader, name, &n);
    if (!p) {
        fprintf(stderr, "取张量失败: %s\n", name);
        return NULL;
    }
    if (expect_numel && n != expect_numel) {
        fprintf(stderr, "张量 %s 元素数不对: %zu != %zu\n", name, n, expect_numel);
        free(p);
        return NULL;
    }
    return p;   // 调用者(bert_free)负责 free
}

// 加载时把 F32 权重量化成 Q8_0,缓存到层结构(避免每次推理重复量化)
static void* prequant_q8_0(const float* src, int64_t nrows, int64_t ncols) {
    ggml_quantize_init(GGML_TYPE_Q8_0);
    size_t nbytes = ggml_row_size(GGML_TYPE_Q8_0, ncols) * (size_t)nrows;
    void* dst = malloc(nbytes);
    if (dst) ggml_quantize_chunk(GGML_TYPE_Q8_0, src, dst, 0, nrows, ncols, NULL);
    return dst;
}

int bert_load(BertModel* m, const char* model_dir) {
    memset(m, 0, sizeof(*m));

    // ---- 配置(按 bge-small-zh-v1.5 固定;以后可从 config.json 解析)----
    m->cfg.vocab_size = 21128;
    m->cfg.hidden_dim = 512;
    m->cfg.num_layers = 4;
    m->cfg.num_heads  = 8;
    m->cfg.max_seq    = 512;
    m->cfg.ffn_dim    = 2048;
    m->cfg.layer_norm_eps = 1e-12f;
    m->cfg.n_threads  = 0;   // 0 = 自动(默认单线程;可由上层设)
    // 权重量化:TINYEMB_QUANT=1 开启 Q8_0(降内存 ~4x,小 CPU 提速)
    m->cfg.quant = (getenv("TINYEMB_QUANT") && getenv("TINYEMB_QUANT")[0] == '1');

    int H = m->cfg.hidden_dim;

    // ---- 打开权重文件 ----
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    if (st_open(&m->loader, path) != 0) return -1;

    // ---- 嵌入层 ----
    m->word_emb = load_tensor(m, "embeddings.word_embeddings.weight",
                              (size_t)m->cfg.vocab_size * H);
    m->pos_emb  = load_tensor(m, "embeddings.position_embeddings.weight",
                              (size_t)m->cfg.max_seq * H);
    m->type_emb = load_tensor(m, "embeddings.token_type_embeddings.weight",
                              2 * (size_t)H);
    m->emb_ln_g = load_tensor(m, "embeddings.LayerNorm.weight", H);
    m->emb_ln_b = load_tensor(m, "embeddings.LayerNorm.bias", H);
    if (!m->word_emb || !m->pos_emb || !m->type_emb || !m->emb_ln_g || !m->emb_ln_b)
        return -1;

    // ---- 每层 Encoder 权重 ----
    for (int i = 0; i < m->cfg.num_layers; i++) {
        char base[256];
        BertLayerW* w = &m->layers[i];

        snprintf(base, sizeof(base), "encoder.layer.%d.attention.self.query", i);
        char name[320];
        #define LOAD(field, suffix, numel) do { \
            snprintf(name, sizeof(name), "%s%s", base, suffix); \
            w->field = load_tensor(m, name, numel); \
            if (!w->field) return -1; \
        } while (0)

        // 自注意力 Q/K/V
        LOAD(Wq, ".weight", (size_t)H*H); LOAD(bq, ".bias", H);
        snprintf(base, sizeof(base), "encoder.layer.%d.attention.self.key", i);
        LOAD(Wk, ".weight", (size_t)H*H); LOAD(bk, ".bias", H);
        snprintf(base, sizeof(base), "encoder.layer.%d.attention.self.value", i);
        LOAD(Wv, ".weight", (size_t)H*H); LOAD(bv, ".bias", H);
        // 输出投影 + 注意力后 LayerNorm
        snprintf(base, sizeof(base), "encoder.layer.%d.attention.output.dense", i);
        LOAD(Wo, ".weight", (size_t)H*H); LOAD(bo, ".bias", H);
        snprintf(base, sizeof(base), "encoder.layer.%d.attention.output.LayerNorm", i);
        LOAD(ln1_g, ".weight", H); LOAD(ln1_b, ".bias", H);
        // FFN
        snprintf(base, sizeof(base), "encoder.layer.%d.intermediate.dense", i);
        LOAD(W1, ".weight", (size_t)m->cfg.ffn_dim*H); LOAD(b1, ".bias", m->cfg.ffn_dim);
        snprintf(base, sizeof(base), "encoder.layer.%d.output.dense", i);
        LOAD(W2, ".weight", (size_t)H*m->cfg.ffn_dim); LOAD(b2, ".bias", H);
        snprintf(base, sizeof(base), "encoder.layer.%d.output.LayerNorm", i);
        LOAD(ln2_g, ".weight", H); LOAD(ln2_b, ".bias", H);
        #undef LOAD
        w->quant = m->cfg.quant;   // 该层是否走 Q8_0
        // 预量化:加载时算一次 Q8_0,层内直接引用(不重复量化)
        if (w->quant) {
            w->Wq_q = prequant_q8_0(w->Wq, H, H);
            w->Wk_q = prequant_q8_0(w->Wk, H, H);
            w->Wv_q = prequant_q8_0(w->Wv, H, H);
            w->Wo_q = prequant_q8_0(w->Wo, H, H);
            w->W1_q = prequant_q8_0(w->W1, m->cfg.ffn_dim, H);
            w->W2_q = prequant_q8_0(w->W2, H, m->cfg.ffn_dim);
        }
    }
    return 0;
}

void bert_free(BertModel* m) {
    free(m->word_emb); free(m->pos_emb); free(m->type_emb);
    free(m->emb_ln_g); free(m->emb_ln_b);
    for (int i = 0; i < m->cfg.num_layers; i++) {
        BertLayerW* w = &m->layers[i];
        free((void*)w->Wq); free((void*)w->bq);
        free((void*)w->Wk); free((void*)w->bk);
        free((void*)w->Wv); free((void*)w->bv);
        free((void*)w->Wo); free((void*)w->bo);
        free((void*)w->ln1_g); free((void*)w->ln1_b);
        free((void*)w->W1); free((void*)w->b1);
        free((void*)w->W2); free((void*)w->b2);
        free((void*)w->ln2_g); free((void*)w->ln2_b);
        // 释放预量化缓冲
        free(w->Wq_q); free(w->Wk_q); free(w->Wv_q);
        free(w->Wo_q); free(w->W1_q); free(w->W2_q);
    }
    st_close(&m->loader);
    memset(m, 0, sizeof(*m));
}

// 嵌入层:x[t] = word[id[t]] + pos[t] + type[0],再过 LayerNorm
static void embed(BertModel* m, const int* ids, int seq, float* x) {
    int H = m->cfg.hidden_dim;
    for (int t = 0; t < seq; t++) {
        float* xt = x + t * H;
        const float* w = m->word_emb + (size_t)ids[t] * H;
        const float* p = m->pos_emb + (size_t)t * H;
        const float* ty = m->type_emb;   // 句子类型 0
        for (int d = 0; d < H; d++)
            xt[d] = w[d] + p[d] + ty[d];
    }
    // LayerNorm 每行
    for (int t = 0; t < seq; t++) {
        float* xt = x + t * H;
        float mean = 0;
        for (int d = 0; d < H; d++) mean += xt[d];
        mean /= H;
        float var = 0;
        for (int d = 0; d < H; d++) { float e = xt[d] - mean; var += e*e; }
        var /= H;
        float inv = 1.0f / sqrtf(var + m->cfg.layer_norm_eps);
        for (int d = 0; d < H; d++)
            xt[d] = (xt[d] - mean) * inv * m->emb_ln_g[d] + m->emb_ln_b[d];
    }
}

int bert_encode(const BertModel* m, const int* ids, int seq, float* out) {
    return bert_encode_batch(m, ids, &seq, 1, out);
}

int bert_encode_batch(const BertModel* m, const int* ids,
                      const int* seq_lens, int n_seqs, float* out) {
    if (n_seqs <= 0) return -1;
    int H = m->cfg.hidden_dim;

    // 总 token 数 = 各句拼接(每句含 [CLS]/[SEP],由调用方保证)
    int total = 0;
    for (int s = 0; s < n_seqs; s++) {
        if (seq_lens[s] <= 0 || total + seq_lens[s] > m->cfg.max_seq) return -1;
        total += seq_lens[s];
    }

    int n = total * H;
    float* x = malloc(n * sizeof(float));
    float* y = malloc(n * sizeof(float));
    if (!x || !y) { free(x); free(y); return -1; }

    // ① 逐句嵌入(位置编码每句从 0 重新计),拼进 x
    {
        int off = 0;
        for (int s = 0; s < n_seqs; s++) {
            int len = seq_lens[s];
            float* xs = x + off * H;
            for (int t = 0; t < len; t++) {
                float* xt = xs + t * H;
                const float* we = m->word_emb + (size_t)ids[off + t] * H;
                const float* pe = m->pos_emb + (size_t)t * H;
                const float* te = m->type_emb;
                for (int d = 0; d < H; d++) xt[d] = we[d] + pe[d] + te[d];
            }
            for (int t = 0; t < len; t++) {
                float* xt = xs + t * H;
                float mean = 0;
                for (int d = 0; d < H; d++) mean += xt[d];
                mean /= H;
                float var = 0;
                for (int d = 0; d < H; d++) { float e = xt[d] - mean; var += e*e; }
                var /= H;
                float inv = 1.0f / sqrtf(var + m->cfg.layer_norm_eps);
                for (int d = 0; d < H; d++)
                    xt[d] = (xt[d] - mean) * inv * m->emb_ln_g[d] + m->emb_ln_b[d];
            }
            off += len;
        }
    }

    // ② 块对角注意力掩码:各句只看自己
    int* attn = malloc((size_t)total * total * sizeof(int));
    memset(attn, 0, (size_t)total * total * sizeof(int));
    {
        int off = 0;
        for (int s = 0; s < n_seqs; s++) {
            int len = seq_lens[s];
            for (int qi = 0; qi < len; qi++)
                for (int ki = 0; ki < len; ki++)
                    attn[(off + qi) * total + (off + ki)] = 1;
            off += len;
        }
    }

    // ③ 堆 N 层 Encoder
    for (int i = 0; i < m->cfg.num_layers; i++) {
        bert_layer(&m->layers[i], x, y, total, H, m->cfg.num_heads, attn, m->cfg.n_threads);
        memcpy(x, y, n * sizeof(float));
    }

    // ④ 每句取自己的 [CLS](第 0 个 token)+ L2 归一化
    {
        int off = 0;
        for (int s = 0; s < n_seqs; s++) {
            float* o = out + s * H;
            memcpy(o, x + off * H, H * sizeof(float));
            float sum2 = 0;
            for (int d = 0; d < H; d++) sum2 += o[d] * o[d];
            float norm = sqrtf(sum2);
            if (norm > 0) for (int d = 0; d < H; d++) o[d] /= norm;
            off += seq_lens[s];
        }
    }

    free(x); free(y); free(attn);
    return 0;
}
