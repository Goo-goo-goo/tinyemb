// M4:ggml 基础热身 —— LayerNorm + 矩阵乘
// 目标:理解张量(tensor)和计算图(graph),不碰 BERT
#pragma once
#include <stddef.h>

// 对一整行做 LayerNorm:
//   y[i] = (x[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
// x/gamma/beta 都是 dim 个 float,y 是输出(调用者分配 dim 个 float)
// 返回 0 成功
int layernorm(const float* x, const float* gamma, const float* beta,
              float* y, int dim, float eps);

// 矩阵乘:z = W @ x
// W: out_dim × in_dim(行主序,W[r*in_dim + c])
// x: in_dim 个
// z: out_dim 个(调用者分配)
// 返回 0 成功
int matmul(const float* W, const float* x, float* z, int out_dim, int in_dim);
