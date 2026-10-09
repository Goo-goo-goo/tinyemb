// M5a/优化版:Transformer Encoder 层
// 优化要点:
//   1. 全部用 ggml 算子(mul_mat/gelu/norm/add),吃 NEON/AMX 向量化内核
//   2. 一层 = 一个 ggml 图(不是每个子操作各建一次图)
//   3. 支持多线程计算(n_threads)
#include "io/bert_layer.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =============================================================
 * Q8_0 权重量化辅助
 * ggml_quantize_chunk 把 F32 权重 [out][in] 转成 Q8_0。
 * Q8_0: 32 个 float/块 → 32 字节 int8 + 4 字节 scale ≈ 1.125 字节/元素。
 * ============================================================= */
static void* quantize_q8_0(const float* src, int64_t nrows, int64_t ncols) {
    ggml_quantize_init(GGML_TYPE_Q8_0);
    // 注意:ggml_quantize_chunk 不支持 dst=NULL 查大小,必须先按 row_size 算好分配
    size_t nbytes = ggml_row_size(GGML_TYPE_Q8_0, ncols) * (size_t)nrows;
    void* dst = malloc(nbytes);
    if (!dst) return NULL;
    ggml_quantize_chunk(GGML_TYPE_Q8_0, src, dst, 0, nrows, ncols, NULL);
    return dst;
}

/* =============================================================
 * 多头自注意力:整个一层的注意力做成一个 ggml 图
 * ============================================================= */
int self_attention(const float* q, const float* k, const float* v,
                   float* out, int seq, int dim, int n_head,
                   const int* attn_mask, int n_threads) {
    int head_dim = dim / n_head;
    int nt = n_threads > 0 ? n_threads : 1;

    size_t mem = 0;
    mem += (4 * seq * dim + 2 * (size_t)n_head * seq * seq + seq * seq)
           * ggml_type_size(GGML_TYPE_F32);
    mem += 32 * ggml_tensor_overhead();
    mem += 4 * ggml_graph_overhead();
    mem += 64 * 1024;

    struct ggml_init_params params = { mem, NULL, false };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return -1;

    struct ggml_tensor* tq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
    struct ggml_tensor* tk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
    struct ggml_tensor* tv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
    memcpy(tq->data, q, seq * dim * sizeof(float));
    memcpy(tk->data, k, seq * dim * sizeof(float));
    memcpy(tv->data, v, seq * dim * sizeof(float));

    struct ggml_tensor* q3 = ggml_reshape_3d(ctx, tq, head_dim, n_head, seq);
    struct ggml_tensor* k3 = ggml_reshape_3d(ctx, tk, head_dim, n_head, seq);
    struct ggml_tensor* v3 = ggml_reshape_3d(ctx, tv, head_dim, n_head, seq);

    struct ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q3, 0, 2, 1, 3));
    struct ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3));
    struct ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v3, 1, 2, 0, 3));

    struct ggml_tensor* scores = ggml_mul_mat(ctx, kp, qp);
    float scale = 1.0f / sqrtf((float)head_dim);
    scores = ggml_scale(ctx, scores, scale);

    if (attn_mask) {
        struct ggml_tensor* bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq, seq, 1);
        float* bd = (float*)bias->data;
        for (int qi = 0; qi < seq; qi++)
            for (int ki = 0; ki < seq; ki++)
                bd[qi * seq + ki] = (attn_mask[qi * seq + ki] == 0) ? -1e9f : 0.0f;
        scores = ggml_add(ctx, scores, bias);
    }

    struct ggml_tensor* probs = ggml_soft_max(ctx, scores);
    struct ggml_tensor* o = ggml_mul_mat(ctx, vp, probs);
    struct ggml_tensor* o2 = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));
    struct ggml_tensor* o3 = ggml_reshape_2d(ctx, o2, n_head * head_dim, seq);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o3);
    ggml_graph_compute_with_ctx(ctx, gf, nt);

    memcpy(out, o3->data, seq * dim * sizeof(float));
    ggml_free(ctx);
    return 0;
}

/* =============================================================
 * 一层 Encoder:整体做成一个 ggml 图
 * Post-LN:变换 → 残差相加 → LayerNorm
 * ============================================================= */
int bert_layer(const BertLayerW* w,
               const float* x, float* y,
               int seq, int dim, int n_head,
               const int* attn_mask, int n_threads) {
    int head_dim = dim / n_head;
    int ffn = dim * 4;
    int nt = n_threads > 0 ? n_threads : 1;

    // 内存:输入 + 每个中间结果 + 权重(常量)+ mask + 图
    // 这里把权重也放进图的常量张量,ggml_mul_mat 直接用
    size_t n_elem = (size_t)seq * dim;
    size_t mem = 0;
    // 激活:输入 + Q/K/V + att + sa + l1 + m1/ff1/m2/sa2/l2 等中间张量,留足
    mem += (n_elem * 20 + (size_t)seq * ffn * 6) * ggml_type_size(GGML_TYPE_F32);
    // 权重:Q/K/V/Wo 4×dim² + W1/W2 2×dim×ffn + LN/bias
    mem += ((size_t)dim*dim*4 + (size_t)dim*ffn*2 + dim*12) * ggml_type_size(GGML_TYPE_F32);
    // 注意力 scores/probs/bias:各 [seq,seq,n_head](批量 seq 大,seq² 主导)
    mem += (size_t)n_head * seq * seq * 6 * ggml_type_size(GGML_TYPE_F32);
    mem += 64 * ggml_tensor_overhead();
    mem += 8 * ggml_graph_overhead();
    mem += 2 * 1024 * 1024;   // 兜底余量(大 batch 安全)

    struct ggml_init_params params = { mem, NULL, false };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return -1;

    #define W2(name, src, out_, in_) do { \
        name = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_, out_); \
        memcpy(name->data, src, (size_t)(out_)*(in_)*sizeof(float)); \
    } while (0)
    #define B1(name, src, n_) \
        struct ggml_tensor* name = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_); \
        memcpy(name->data, src, (size_t)(n_)*sizeof(float))
    // 量化权重:Q8_0 张量,ggml_mul_mat 直接吃(Q8_0 内核,内存省 4x)
    // 量化权重:优先用预量化缓冲(加载时算好,不重复量化),否则现场量化
    #define W2Q(name, preq, src, out_, in_) do { \
        name = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, in_, out_); \
        if (preq) memcpy(name->data, preq, ggml_nbytes(name)); \
        else { void* q = quantize_q8_0(src, out_, in_); memcpy(name->data, q, ggml_nbytes(name)); free(q); } \
    } while (0)

    // 权重常量。大矩阵按需 Q8_0 量化(bias/LN 保持 F32)。
    struct ggml_tensor *tWq, *tWk, *tWv, *tWo, *tW1, *tW2;
    if (w->quant) {
        W2Q(tWq, w->Wq_q, w->Wq, dim, dim);  W2Q(tWk, w->Wk_q, w->Wk, dim, dim);
        W2Q(tWv, w->Wv_q, w->Wv, dim, dim);  W2Q(tWo, w->Wo_q, w->Wo, dim, dim);
        W2Q(tW1, w->W1_q, w->W1, ffn, dim);  W2Q(tW2, w->W2_q, w->W2, dim, ffn);
    } else {
        W2(tWq, w->Wq, dim, dim);   W2(tWk, w->Wk, dim, dim);
        W2(tWv, w->Wv, dim, dim);   W2(tWo, w->Wo, dim, dim);
        W2(tW1, w->W1, ffn, dim);   W2(tW2, w->W2, dim, ffn);
    }
    B1(tbq, w->bq, dim);   B1(tbk, w->bk, dim);
    B1(tbv, w->bv, dim);   B1(tbo, w->bo, dim);
    B1(tln1g, w->ln1_g, dim);   B1(tln1b, w->ln1_b, dim);
    B1(tb1, w->b1, ffn);   B1(tb2, w->b2, dim);
    B1(tln2g, w->ln2_g, dim);   B1(tln2b, w->ln2_b, dim);

    // 输入 x:[dim, seq]
    struct ggml_tensor* tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
    memcpy(tx->data, x, n_elem * sizeof(float));

    // ---- 子层 1:自注意力 ----
    // Q/K/V 投影:mul_mat(W, x) → [dim, seq];bias 用 [dim,1] 广播(ggml_add 自带广播)
    struct ggml_tensor* Q = ggml_add(ctx, ggml_mul_mat(ctx, tWq, tx), ggml_reshape_2d(ctx, tbq, dim, 1));
    struct ggml_tensor* K = ggml_add(ctx, ggml_mul_mat(ctx, tWk, tx), ggml_reshape_2d(ctx, tbk, dim, 1));
    struct ggml_tensor* V = ggml_add(ctx, ggml_mul_mat(ctx, tWv, tx), ggml_reshape_2d(ctx, tbv, dim, 1));

    // 多头注意力
    struct ggml_tensor* q3 = ggml_reshape_3d(ctx, Q, head_dim, n_head, seq);
    struct ggml_tensor* k3 = ggml_reshape_3d(ctx, K, head_dim, n_head, seq);
    struct ggml_tensor* v3 = ggml_reshape_3d(ctx, V, head_dim, n_head, seq);
    struct ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q3, 0, 2, 1, 3));
    struct ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3));
    struct ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v3, 1, 2, 0, 3));

    struct ggml_tensor* scores = ggml_scale(ctx, ggml_mul_mat(ctx, kp, qp), 1.0f / sqrtf((float)head_dim));
    if (attn_mask) {
        struct ggml_tensor* bias = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, seq, seq, 1);
        float* bd = (float*)bias->data;
        for (int qi = 0; qi < seq; qi++)
            for (int ki = 0; ki < seq; ki++)
                bd[qi * seq + ki] = (attn_mask[qi * seq + ki] == 0) ? -1e9f : 0.0f;
        scores = ggml_add(ctx, scores, bias);
    }
    struct ggml_tensor* probs = ggml_soft_max(ctx, scores);
    struct ggml_tensor* ao = ggml_mul_mat(ctx, vp, probs);
    struct ggml_tensor* ao2 = ggml_cont(ctx, ggml_permute(ctx, ao, 0, 2, 1, 3));
    struct ggml_tensor* att = ggml_reshape_2d(ctx, ao2, dim, seq);

    // 输出投影 + 残差
    struct ggml_tensor* sa = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, tWo, att), ggml_reshape_2d(ctx, tbo, dim, 1)), tx);

    // LN(子层1 后):norm → *gamma + beta
    struct ggml_tensor* l1 = ggml_norm(ctx, sa, 1e-12f);
    l1 = ggml_add(ctx, ggml_mul(ctx, l1, ggml_reshape_2d(ctx, tln1g, dim, 1)),
                  ggml_reshape_2d(ctx, tln1b, dim, 1));

    // ---- 子层 2:FFN ----
    // m1 = l1 @ W1^T → [ffn, seq];bias [ffn,1] 广播
    struct ggml_tensor* m1 = ggml_mul_mat(ctx, tW1, l1);
    struct ggml_tensor* ff1 = ggml_gelu_erf(ctx, ggml_add(ctx, m1, ggml_reshape_2d(ctx, tb1, ffn, 1)));
    struct ggml_tensor* m2 = ggml_mul_mat(ctx, tW2, ff1);
    struct ggml_tensor* sa2 = ggml_add(ctx, ggml_add(ctx, m2, ggml_reshape_2d(ctx, tb2, dim, 1)), l1);

    // LN(子层2 后)
    struct ggml_tensor* l2 = ggml_norm(ctx, sa2, 1e-12f);
    l2 = ggml_add(ctx, ggml_mul(ctx, l2, ggml_reshape_2d(ctx, tln2g, dim, 1)),
                  ggml_reshape_2d(ctx, tln2b, dim, 1));

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, l2);
    ggml_graph_compute_with_ctx(ctx, gf, nt);

    memcpy(y, l2->data, n_elem * sizeof(float));
    ggml_free(ctx);
    return 0;
}
