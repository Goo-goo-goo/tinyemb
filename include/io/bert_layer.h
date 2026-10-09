// M5a:Transformer Encoder 层(用 ggml)
#pragma once
#include <stddef.h>
#include "io/safetensor_loader.h"

// 一层 Encoder 的权重(从 loader 里取出来放一起,方便传参)
typedef struct {
    // 自注意力
    const float* Wq;  const float* bq;   // query
    const float* Wk;  const float* bk;   // key
    const float* Wv;  const float* bv;   // value
    const float* Wo;  const float* bo;   // 输出投影
    // 第一个 LayerNorm(注意力前)
    const float* ln1_g; const float* ln1_b;
    // FFN
    const float* W1;  const float* b1;   // 512 → 2048
    const float* W2;  const float* b2;   // 2048 → 512
    // 第二个 LayerNorm(FFN 前)
    const float* ln2_g; const float* ln2_b;
    // 量化标志:非 0 时把 6 个大矩阵权重(Wq/Wk/Wv/Wo/W1/W2)在加载时转 Q8_0,
    // 内存降 ~4x,小 CPU 矩阵乘更快。bias/LN 保持 F32。
    int quant;          // 0=F32(默认), 1=Q8_0
    // 预量化的权重缓冲(quant=1 时有效,由 bert_load 分配,bert_free 释放)
    void* Wq_q;  void* Wk_q;  void* Wv_q;  void* Wo_q;  void* W1_q;  void* W2_q;
} BertLayerW;

// 跑一层 Encoder:
//   x: [seq][dim] 行主序(输入,seq 个 token,每个 dim 维)
//   y: [seq][dim](输出,调用者分配)
//   attn_mask: [seq][seq] 的二维注意力掩码(行=q,列=k),1=可看,0=屏蔽;
//              NULL 表示全连接(无 padding)。批量推理用块对角掩码隔离各句。
//   n_threads: ggml 计算线程数(0 = 自动)
// dim 必须等于 n_head * head_dim
// 返回 0 成功
int bert_layer(const BertLayerW* w,
               const float* x, float* y,
               int seq, int dim, int n_head,
               const int* attn_mask, int n_threads);

// 多头自注意力(单独出来,方便测试):
//   q/k/v: [seq][dim] 已经算好的 Q/K/V
//   out:   [seq][dim] 注意力输出(调用者分配)
int self_attention(const float* q, const float* k, const float* v,
                   float* out, int seq, int dim, int n_head,
                   const int* attn_mask, int n_threads);
