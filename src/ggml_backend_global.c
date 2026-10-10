// ggml 后端计算封装实现
// 全变体 + GGML_BACKEND_DL 模式下,CPU 后端是动态加载的插件,不能直接调用
// ggml_backend_cpu_init(那是 MODULE 插件符号)。正确方式:走动态注册表 API
// (全部 GGML_API 从主库导出,可直接链接):
//   ggml_backend_init_by_type(CPU)            → 拿 CPU 后端
//   ggml_backend_reg_get_proc_address(...)   → 拿线程设置函数指针
//   ggml_backend_graph_compute(backend, gf)  → 计算(自动管 work buffer)
#include "io/ggml_backend_global.h"
#include "ggml-backend.h"

#include <stdio.h>

// CPU 后端设置线程数的函数指针类型(从注册表动态获取)
typedef void (*ggml_set_n_threads_t)(ggml_backend_t, int);

static ggml_backend_t     g_backend      = NULL;
static ggml_set_n_threads_t g_set_threads = NULL;

int ggml_backend_global_init(void) {
    if (g_backend) return 0;

    // 关键:先加载所有后端插件(含 CPU 变体)进注册表。
    // 全变体模式下 CPU 后端是动态插件,不 load 就不在注册表里,
    // init_by_type 会找不到设备返回 NULL。
    ggml_backend_load_all();

    // 从注册表拿 CPU 后端(运行时按 CPU 自动选最优变体)
    g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!g_backend) {
        fprintf(stderr, "ggml_backend_init_by_type(CPU) 失败\n");
        return -1;
    }

    // 拿 CPU 后端的 set_n_threads 函数指针(可选,拿不到就用默认线程)
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
    if (reg) {
        g_set_threads = (ggml_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    }
    return 0;
}

void ggml_backend_global_free(void) {
    if (g_backend) {
        ggml_backend_free(g_backend);
        g_backend = NULL;
    }
    g_set_threads = NULL;
}

void ggml_backend_global_set_threads(int n_threads) {
    if (g_backend && g_set_threads) {
        g_set_threads(g_backend, n_threads);
    }
}

int ggml_backend_global_compute(struct ggml_cgraph* gf) {
    if (ggml_backend_global_init() != 0) return -1;
    enum ggml_status st = ggml_backend_graph_compute(g_backend, gf);
    return (st == GGML_STATUS_SUCCESS) ? 0 : -1;
}
