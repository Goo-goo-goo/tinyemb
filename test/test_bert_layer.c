// M5a 单元测试:多头自注意力 + 单层 Encoder
// 关键:验证批量切头的维度变换正确(这是最容易错的地方)
#include "io/bert_layer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// 测 1:单头、无 mask 的最小注意力,手算对拍
// seq=2, dim=2(1 个头,head_dim=2)
// q = k = v = [[1,0],[0,1]]
// scores = q@k^T / sqrt(2) = [[0.5,0],[0,0.5]]... 实际:
//   q0·k0=1, q0·k1=0, q1·k0=0, q1·k1=1 → [[1,0],[0,1]]/sqrt(2)
// softmax 沿 k:
//   行0: softmax([1,0]/sqrt2)=softmax([0.7071,0])=[0.6704,0.3296]
//   行1: softmax([0,1]/sqrt2)=[0.3296,0.6704]
// out = probs@v = probs@I = probs
static int test_attn_identity(void) {
    int seq = 2, dim = 2, n_head = 1;
    float q[4] = { 1, 0,  0, 1 };
    float k[4] = { 1, 0,  0, 1 };
    float v[4] = { 1, 0,  0, 1 };
    float out[4];
    CHECK(self_attention(q, k, v, out, seq, dim, n_head, NULL, 1) == 0);

    // softmax([0.7071, 0]) = [0.6704, 0.3296]
    CHECK(feq(out[0], 0.6704f, 1e-3f));
    CHECK(feq(out[1], 0.3296f, 1e-3f));
    CHECK(feq(out[2], 0.3296f, 1e-3f));
    CHECK(feq(out[3], 0.6704f, 1e-3f));
    return 0;
}

// 测 2:多头(2 个头)不互相干扰 —— 验证切头逻辑
// dim=4, n_head=2, head_dim=2
// 造 q/k/v 让两个头独立:头0 用前 2 维,头1 用后 2 维
// 如果切头正确,头0 输出只依赖前 2 维,头1 只依赖后 2 维
static int test_attn_two_heads(void) {
    int seq = 2, dim = 4, n_head = 2;
    // q: token0=[1,0, 0,0], token1=[0,0, 0,1]
    // 头0: t0=[1,0], t1=[0,0];头1: t0=[0,0], t1=[0,1]
    float q[8] = { 1,0,0,0,  0,0,0,1 };
    float k[8] = { 1,0,0,0,  0,0,0,1 };
    float v[8] = { 1,0,0,0,  0,0,0,1 };
    float out[8];
    CHECK(self_attention(q, k, v, out, seq, dim, n_head, NULL, 1) == 0);

    // 头0(前2维):q0=[1,0],q1=[0,0],k0=[1,0],k1=[0,0]
    //   scores=[[q0·k0, q0·k1],[q1·k0,q1·k1]]/sqrt2 = [[1,0],[0,0]]/sqrt2
    //   softmax 行0=softmax([0.7071,0])=[0.6704,0.3296]
    //   softmax 行1=softmax([0,0])=[0.5,0.5]
    //   out_h0 = probs@v0, v0=[[1,0],[0,0]]
    //     行0 = 0.6704*[1,0]+0.3296*[0,0] = [0.6704, 0]
    //     行1 = 0.5*[1,0]+0.5*[0,0] = [0.5, 0]
    CHECK(feq(out[0], 0.6704f, 1e-3f));  // t0 头0 维0
    CHECK(feq(out[1], 0.0f,    1e-3f));  // t0 头0 维1
    CHECK(feq(out[4], 0.5f,    1e-3f));  // t1 头0 维0
    CHECK(feq(out[5], 0.0f,    1e-3f));  // t1 头0 维1

    // 头1(后2维):q0=[0,0],q1=[0,1],k0=[0,0],k1=[0,1]
    //   scores=[[0,0],[0,1]]/sqrt2
    //   行0=softmax([0,0])=[0.5,0.5], 行1=softmax([0,0.7071])=[0.3296,0.6704]
    //   out_h1 = probs@v1, v1=[[0,0],[0,1]]
    //     行0=0.5*[0,0]+0.5*[0,1]=[0,0.5]
    //     行1=0.3296*[0,0]+0.6704*[0,1]=[0,0.6704]
    CHECK(feq(out[2], 0.0f,    1e-3f));  // t0 头1 维0
    CHECK(feq(out[3], 0.5f,    1e-3f));  // t0 头1 维1
    CHECK(feq(out[6], 0.0f,    1e-3f));  // t1 头1 维0
    CHECK(feq(out[7], 0.6704f, 1e-3f));  // t1 头1 维1
    return 0;
}

// 测 3:mask —— padding 位置应该不被关注
// seq=3, 但 mask=[1,1,0],token2 是 padding
// 单头 dim=2,q=k=v=I3 的前 2 维
// token2 作为 key 被 mask,softmax 后它的权重≈0
static int test_attn_mask(void) {
    int seq = 3, dim = 2, n_head = 1;
    float q[6] = { 1,0,  0,1,  1,1 };
    float k[6] = { 1,0,  0,1,  1,1 };
    float v[6] = { 1,0,  0,1,  1,1 };
    // 2D 注意力掩码 [seq][seq](行=q,列=k):屏蔽 key=2(padding),所有 query 都不看它
    int mask[9];
    for (int qi = 0; qi < seq; qi++)
        for (int ki = 0; ki < seq; ki++)
            mask[qi * seq + ki] = (ki == 2) ? 0 : 1;
    float out[6];
    CHECK(self_attention(q, k, v, out, seq, dim, n_head, mask, 1) == 0);

    // token0=q[1,0]:只看 k0,k1(因为 k2 被 mask)
    //   q0·k0=1, q0·k1=0, q0·k2=mask→-1e9
    //   softmax([0.7071, 0, -1e9]) ≈ [0.6704, 0.3296, 0]
    //   out0 = 0.6704*v0 + 0.3296*v1 + 0 = [0.6704, 0.3296]
    CHECK(feq(out[0], 0.6704f, 1e-3f));
    CHECK(feq(out[1], 0.3296f, 1e-3f));
    return 0;
}

int main(void) {
    RUN_TEST(test_attn_identity);
    RUN_TEST(test_attn_two_heads);
    RUN_TEST(test_attn_mask);
    printf("\n===== 结果: %d 通过, %d 失败 =====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
