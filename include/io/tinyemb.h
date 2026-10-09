// tinyemb 共享库导出 API:给 Python/其他语言调用的干净接口
// 设计原则:调用方只碰字符串和浮点数组,不碰内部结构体
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 不透明句柄(隐藏内部 BertModel / WordPiece 细节)
typedef struct TinyEmb TinyEmb;

// 加载模型(模型目录需含 model.safetensors + vocab.txt)。成功返回句柄,失败 NULL
TinyEmb* tinyemb_load(const char* model_dir);

// 释放句柄
void tinyemb_free(TinyEmb* h);

// 隐藏维度(=512)
int tinyemb_dim(TinyEmb* h);

// 设置推理线程数(0=自动,建议用物理性能核数)。返回实际设置值
int tinyemb_set_threads(TinyEmb* h, int n_threads);

// 把一段文本编码成 embedding
//   text: UTF-8 文本
//   out: 调用者分配的 float 数组,长度 >= tinyemb_dim(h)
//   out_n: 传入 out 容量,返回实际写入维度
// 返回 0 成功,-1 失败
int tinyemb_encode_text(TinyEmb* h, const char* text, float* out, int out_cap);

// 编码一段已分词的 id 序列(高级用法,一般用 encode_text)
int tinyemb_encode_ids(TinyEmb* h, const int* ids, int n_ids, float* out, int out_cap);

// 批量编码多段文本(共享矩阵乘,吞吐远高于逐句)
//   texts: [n_texts] 个 UTF-8 字符串
//   out: 调用者分配 [n_texts][dim] 行主序
// 返回 0 成功
int tinyemb_encode_texts(TinyEmb* h, const char** texts, int n_texts, float* out, int out_cap);

// 查询文本的 token 数(用于 usage 统计)
int tinyemb_token_count(TinyEmb* h, const char* text);

#ifdef __cplusplus
}
#endif
