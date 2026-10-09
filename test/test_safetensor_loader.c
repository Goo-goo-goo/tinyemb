// 单元测试:safetensors loader
// 编译运行:cmake --build build --target step1 -j && ./build/step1
// 期望:全部 PASS,退出码 0。任何一个失败会打印 FAIL 并返回非 0。
#include "io/safetensor_loader.h"

#include <assert.h>   // 断言宏:条件为假就立刻报错终止
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- 极简测试框架(约 20 行,不引入任何外部依赖)----
static int g_pass = 0, g_fail = 0;

// 每个测试是一个"返回 0 表示通过"的函数
#define RUN_TEST(fn) do {                                  \
    printf("[RUN ] %s\n", #fn);          /* #fn 把参数变成字符串,打印函数名 */ \
    if ((fn)() == 0) { g_pass++; printf("[ OK ] %s\n", #fn); } \
    else             { g_fail++; printf("[FAIL] %s\n", #fn); } \
} while (0)

// 断言:失败时打印文件:行号和表达式,并让测试函数返回 1(失败)
#define CHECK(cond) do {                                   \
    if (!(cond)) {                                         \
        printf("       断言失败: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1;                                          \
    }                                                      \
} while (0)

// 浮点比较:两个数差的绝对值小于 eps 才算相等(浮点数不能直接 ==)
static int feq(float a, float b, float eps) {
    float d = a - b;
    if (d < 0) d = -d;
    return d < eps;
}

// ---- 测试用例 ----

// 模型文件路径(所有测试共享)
#define MODEL "models/bge-small-zh-v1.5/model.safetensors"

// 测 1:能打开文件,张量个数正确
static int test_open_count(void) {
    ST_Loader L;
    CHECK(st_open(&L, MODEL) == 0);          // 打开必须成功
    CHECK(L.header_len == 7864);             // 头部长度(实测值)
    CHECK(L.n_tensors == 72);                // 张量总数
    st_close(&L);
    return 0;
}

// 测 2:能按名字找到张量,元信息正确
static int test_find_tensor(void) {
    ST_Loader L;
    CHECK(st_open(&L, MODEL) == 0);

    const TensorInfo* t = st_find(&L, "embeddings.word_embeddings.weight");
    CHECK(t != NULL);                        // 找到了
    CHECK(strcmp(t->dtype, "F32") == 0);     // 类型是 F32
    CHECK(t->shape_len == 2);                // 二维张量
    CHECK(t->shape[0] == 21128);             // 21128 行(词表大小)
    CHECK(t->shape[1] == 512);               // 512 列(隐藏维度)
    CHECK(t->offset_begin == 1060864);       // 字节偏移
    CHECK(t->offset_end == 44331008);

    // 不存在的名字应该返回 NULL
    CHECK(st_find(&L, "不存在的张量") == NULL);

    st_close(&L);
    return 0;
}

// 测 3:读出的浮点数个数和数值正确(和字节级基准对拍)
static int test_read_floats(void) {
    ST_Loader L;
    CHECK(st_open(&L, MODEL) == 0);

    size_t n = 0;
    float* w = st_read_floats(&L, "embeddings.word_embeddings.weight", &n);
    CHECK(w != NULL);
    CHECK(n == 10817536);                    // 21128 × 512

    // [CLS] (id=101) 那一行前 4 个数。
    // 期望值来自 torch 权威读取(safetensors.torch.load_file)。
    // 注意:data_offsets 是相对"数据区起点"(8+header_len),不是文件头。
    const float expect[4] = { -0.148682f, 0.057526f, -0.026657f, -0.045593f };
    for (int i = 0; i < 4; i++)
        CHECK(feq(w[101 * 512 + i], expect[i], 1e-5f));

    free(w);                                 // 谁 malloc 谁 free
    st_close(&L);
    return 0;
}

// 测 4:错误路径 —— 不存在的文件、不存在的张量都要安全失败,不能崩溃
static int test_error_paths(void) {
    ST_Loader L;
    CHECK(st_open(&L, "不存在的文件.safetensors") != 0);  // 打开失败应返回非 0

    CHECK(st_open(&L, MODEL) == 0);
    size_t n;
    CHECK(st_read_floats(&L, "没有这个名字", &n) == NULL); // 找不到应返回 NULL
    st_close(&L);
    return 0;
}

int main(void) {
    RUN_TEST(test_open_count);
    RUN_TEST(test_find_tensor);
    RUN_TEST(test_read_floats);
    RUN_TEST(test_error_paths);

    // 汇总
    printf("\n===== 结果: %d 通过, %d 失败 =====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;              // 有失败就返回非 0(自动化脚本可判断)
}
