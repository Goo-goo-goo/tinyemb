// M4:用 ggml 实现 LayerNorm 和矩阵乘
// 重点不是这两个操作本身,而是展示 ggml 的"搭图 → 计算"范式
#include "io/ggml_ops.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * LayerNorm
 * 在 ggml 里就是 5 个节点:
 *   mean = mean(x)              → 求均值
 *   x2   = sqr(x - mean)        → 离均值平方
 *   var  = mean(x2)             → 求方差
 *   norm = (x - mean) / sqrt(var+eps)
 *   y    = norm * gamma + beta
 * 但 ggml 有现成的 ggml_norm() 一步做完前三步。
 * ============================================================ */

int layernorm(const float* x, const float* gamma, const float* beta,
              float* y, int dim, float eps) {
    // ---- 第一步:算出需要多大内存 ----
    // 3 个输入张量 + 3 个中间结果 + 3 个张量元数据 + 图 + 余量
    size_t mem = 0;
    mem += 3 * dim * ggml_type_size(GGML_TYPE_F32);  // x, gamma, beta
    mem += 3 * dim * ggml_type_size(GGML_TYPE_F32);  // norm, scaled, y
    mem += 6 * ggml_tensor_overhead();               // 张量元数据
    mem += ggml_graph_overhead();                    // 计算图
    mem += 1024;                                     // 余量

    // ---- 第二步:初始化上下文 ----
    struct ggml_init_params params = { mem, NULL, /*no_alloc=*/false };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return -1;

    // ---- 第三步:创建输入张量并填数据 ----
    // 注意 ne[0] 是"最快变化的维",一个长度 dim 的向量就是 ne = {dim}
    struct ggml_tensor* tx = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    struct ggml_tensor* tg = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    struct ggml_tensor* tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    memcpy(tx->data, x, dim * sizeof(float));
    memcpy(tg->data, gamma, dim * sizeof(float));
    memcpy(tb->data, beta, dim * sizeof(float));

    // ---- 第四步:搭图(只声明,不算)----
    struct ggml_tensor* norm   = ggml_norm(ctx, tx, eps);       // (x-mean)/sqrt(var+eps)
    struct ggml_tensor* scaled = ggml_mul(ctx, norm, tg);       // * gamma
    struct ggml_tensor* out    = ggml_add(ctx, scaled, tb);     // + beta

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // ---- 第五步:计算 ----
    ggml_graph_compute_with_ctx(ctx, gf, /*n_threads=*/1);

    // ---- 第六步:取结果 ----
    memcpy(y, out->data, dim * sizeof(float));

    ggml_free(ctx);
    return 0;
}

/* ============================================================
 * 矩阵乘 z = W @ x
 * ggml 的 ggml_mul_mat(A, B) 计算的是 B^T @ A 的效果:
 *   A: [in, out]  B: [in, n]  →  结果 [out, n]
 * 所以把 W 摆成 [in_dim, out_dim]、x 摆成 [in_dim, 1] 就得到 [out_dim, 1]。
 * ============================================================ */

int matmul(const float* W, const float* x, float* z, int out_dim, int in_dim) {
    size_t mem = 0;
    mem += (in_dim * out_dim + in_dim + out_dim) * ggml_type_size(GGML_TYPE_F32);
    mem += 3 * ggml_tensor_overhead();
    mem += ggml_graph_overhead();
    mem += 1024;

    struct ggml_init_params params = { mem, NULL, false };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) return -1;

    // W: ne = {in_dim, out_dim} —— 注意 ggml 的 ne[0] 是 in(列),ne[1] 是 out(行)
    struct ggml_tensor* tW = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, out_dim);
    // x: ne = {in_dim, 1}(一个向量)
    struct ggml_tensor* tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, 1);

    // W 是行主序 [out_dim][in_dim],ggml 也是 ne[0]=in 连续存,所以直接拷
    memcpy(tW->data, W, in_dim * out_dim * sizeof(float));
    memcpy(tx->data, x, in_dim * sizeof(float));

    // 搭图 + 计算
    struct ggml_tensor* out = ggml_mul_mat(ctx, tW, tx);   // → [out_dim, 1]
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    memcpy(z, out->data, out_dim * sizeof(float));

    ggml_free(ctx);
    return 0;
}
