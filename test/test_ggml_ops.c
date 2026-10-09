// M4 单元测试:LayerNorm + 矩阵乘,和手算值对拍
#include "io/ggml_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_pass = 0, g_fail = 0;

#define RUN_TEST(fn) do {                                  \
    printf("[RUN ] %s\n", #fn);                            \
    if ((fn)() == 0) { g_pass++; printf("[ OK ] %s\n", #fn); } \
    else             { g_fail++; printf("[FAIL] %s\n", #fn); } \
} while (0)

#define CHECK(cond) do {                                   \
    if (!(cond)) {                                         \
        printf("       断言失败: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1;                                          \
    }                                                      \
} while (0)

static int feq(float a, float b, float eps) {
    float d = a - b; if (d < 0) d = -d;
    return d < eps;
}

// 测 1:LayerNorm。用 4 个数的小例子手算。
// x = [1, 2, 3, 4], gamma=1, beta=0, eps=1e-5
// mean = 2.5, var = [(1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2]/4 = (2.25+0.25+0.25+2.25)/4 = 1.25
// y[i] = (x[i]-2.5)/sqrt(1.25)
// sqrt(1.25)≈1.118034
// y = [-1.3416, -0.4472, 0.4472, 1.3416]
static int test_layernorm(void) {
    float x[4] = { 1, 2, 3, 4 };
    float gamma[4] = { 1, 1, 1, 1 };
    float beta[4]  = { 0, 0, 0, 0 };
    float y[4];
    CHECK(layernorm(x, gamma, beta, y, 4, 1e-5f) == 0);

    const float expect[4] = { -1.341641f, -0.447214f, 0.447214f, 1.341641f };
    for (int i = 0; i < 4; i++) {
        if (!feq(y[i], expect[i], 1e-3f)) {
            printf("       y[%d]=%.6f 期望 %.6f\n", i, y[i], expect[i]);
            CHECK(feq(y[i], expect[i], 1e-3f));
        }
    }
    return 0;
}

// 测 2:LayerNorm 带 gamma/beta(验证仿射变换)
// x=[1,2,3,4] 归一化后同上,再 * gamma=2 + beta=1
static int test_layernorm_affine(void) {
    float x[4] = { 1, 2, 3, 4 };
    float gamma[4] = { 2, 2, 2, 2 };
    float beta[4]  = { 1, 1, 1, 1 };
    float y[4];
    CHECK(layernorm(x, gamma, beta, y, 4, 1e-5f) == 0);

    // expect = norm*2 + 1
    const float expect[4] = { -1.683282f, 0.105573f, 1.894427f, 3.683282f };
    for (int i = 0; i < 4; i++) CHECK(feq(y[i], expect[i], 1e-3f));
    return 0;
}

// 测 3:矩阵乘 z = W @ x
// W = [[1,2],[3,4]] (2×2),x = [5,6]
// z = [1*5+2*6, 3*5+4*6] = [17, 39]
static int test_matmul(void) {
    float W[4] = { 1, 2, 3, 4 };   // 行主序:W[0]={1,2}, W[1]={3,4}
    float x[2] = { 5, 6 };
    float z[2];
    CHECK(matmul(W, x, z, /*out*/2, /*in*/2) == 0);
    CHECK(feq(z[0], 17.0f, 1e-4f));
    CHECK(feq(z[1], 39.0f, 1e-4f));
    return 0;
}

// 测 4:矩阵乘非方阵(3×2 @ 2 向量 → 3)
// W = [[1,2],[3,4],[5,6]] (3 行 2 列),x=[1,1]
// z = [3, 7, 11]
static int test_matmul_rect(void) {
    float W[6] = { 1, 2, 3, 4, 5, 6 };
    float x[2] = { 1, 1 };
    float z[3];
    CHECK(matmul(W, x, z, /*out*/3, /*in*/2) == 0);
    CHECK(feq(z[0], 3.0f, 1e-4f));
    CHECK(feq(z[1], 7.0f, 1e-4f));
    CHECK(feq(z[2], 11.0f, 1e-4f));
    return 0;
}

int main(void) {
    RUN_TEST(test_layernorm);
    RUN_TEST(test_layernorm_affine);
    RUN_TEST(test_matmul);
    RUN_TEST(test_matmul_rect);
    printf("\n===== 结果: %d 通过, %d 失败 =====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
