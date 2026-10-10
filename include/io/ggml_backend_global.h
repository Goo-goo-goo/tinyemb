// ggml 后端计算封装
// 为什么需要:GGML_CPU_ALL_VARIANTS + GGML_BACKEND_DL 模式下,ggml-cpu
// 变体是 MODULE 插件(ggml_graph_compute_with_ctx 不直接导出给 libtinyemb)。
// 正确方式是走后端 API(ggml_backend_graph_compute,从主库导出)。
// 这里封装一个统一入口,替代所有 ggml_graph_compute_with_ctx 调用。
#pragma once
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// 懒初始化全局 CPU 后端(线程数在每次计算前用 tinyemb_set_n_threads 设置)
// 返回 0 成功
int  ggml_backend_global_init(void);
void ggml_backend_global_free(void);

// 设置全局后端的线程数(替代原来传给 ggml_graph_compute_with_ctx 的 n_threads)
void ggml_backend_global_set_threads(int n_threads);

// 用全局 CPU 后端计算图(替代 ggml_graph_compute_with_ctx(ctx, gf, nt))
// 自动管理工作缓冲。返回 0 成功。
int ggml_backend_global_compute(struct ggml_cgraph* gf);

#ifdef __cplusplus
}
#endif
