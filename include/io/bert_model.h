// M5b:完整 BERT embedding 模型
#pragma once
#include <stddef.h>
#include "io/safetensor_loader.h"
#include "io/bert_layer.h"

// 模型配置(从 config.json 读,或手动填)
typedef struct {
    int vocab_size;    // 21128
    int hidden_dim;    // 512
    int num_layers;    // 4
    int num_heads;     // 8
    int max_seq;       // 512
    int ffn_dim;       // 2048 = hidden*4
    float layer_norm_eps;  // 1e-12
    int n_threads;     // ggml 计算线程数(0=自动)
    int quant;         // 0=F32, 1=Q8_0 权重量化(降内存提速)
} BertConfig;

// 整个模型:配置 + 所有权重指针(指向 loader 读出的数据)
typedef struct {
    BertConfig cfg;
    ST_Loader loader;

    // 嵌入层
    float* word_emb;    // [vocab][hidden]
    float* pos_emb;     // [max_seq][hidden]
    float* type_emb;    // [2][hidden]
    float* emb_ln_g;    // [hidden]
    float* emb_ln_b;    // [hidden]

    // 4 层 Encoder 的权重
    BertLayerW layers[4];

    // 最终池化层(可选,BERT 有个 pooler.dense,但 sentence-transformers 用 CLS 不用它)
} BertModel;

// 从模型目录加载(config.json + model.safetensors)。成功返回 0
int  bert_load(BertModel* m, const char* model_dir);
void bert_free(BertModel* m);

// 推理:把 token id 序列编码成 embedding
//   ids: [seq] 个 token id(已含 [CLS]/[SEP])
//   seq: token 个数
//   out: [hidden] 个 float(调用者分配),L2 归一化后的向量
// 返回 0 成功
int bert_encode(const BertModel* m, const int* ids, int seq, float* out);

// 批量推理:多句一次算(共享矩阵乘,吞吐大增)
//   ids: 拼接的 token id 数组(每句含 [CLS]/[SEP])
//   seq_lens: [n_seqs] 每句的 token 数
//   n_seqs: 句子数
//   out: [n_seqs][hidden] 行主序(调用者分配)
// 返回 0 成功
int bert_encode_batch(const BertModel* m, const int* ids,
                      const int* seq_lens, int n_seqs, float* out);
