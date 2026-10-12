// 模型层 truncate 语义测试(临时,用于验证 tinyemb API)
#include "io/tinyemb.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { if(!(cond)){printf("       失败 %s:%d %s\n",__FILE__,__LINE__,#cond);return 1;} } while(0)
#define RUN(fn) do { printf("[RUN ] %s\n",#fn); if((fn)()==0){g_pass++;printf("[ OK ] %s\n",#fn);}else{g_fail++;printf("[FAIL] %s\n",#fn);} } while(0)

#define DIR "models/bge-small-zh-v1.5"

static char* make_text(size_t n) {   // n 个 "的"
    char* s = malloc(n*3+1);
    for (size_t i=0;i<n;i++) memcpy(s+i*3,"\xe7\x9a\x84",3);
    s[n*3]=0; return s;
}

// 超长文本:truncate=1 必须成功出向量,truncate=0 必须 -1
static int test_overlong(void) {
    TinyEmb* h = tinyemb_load(DIR);
    CHECK(h != NULL);
    int dim = tinyemb_dim(h);
    CHECK(dim == 512);
    CHECK(tinyemb_max_seq(h) == 512);

    char* text = make_text(600);            // 600 正文 token >> 512
    float* v = malloc(dim*sizeof(float));

    CHECK(tinyemb_encode_text(h, text, v, dim) == 0);           // 默认 truncate=true
    CHECK(tinyemb_encode_text_ex(h, text, 1, v, dim) == 0);
    CHECK(tinyemb_encode_text_ex(h, text, 0, v, dim) == -1);    // 不截断 → 报错

    // 向量必须是有限值 + L2 归一化(模长≈1)
    float norm = 0;
    for (int i=0;i<dim;i++){ CHECK(isfinite(v[i])); norm += v[i]*v[i]; }
    norm = sqrtf(norm);
    CHECK(fabsf(norm-1.0f) < 1e-3f);

    // token_count 报告截断后长度 <= 512
    int cnt = tinyemb_token_count(h, text);
    CHECK(cnt == 512);

    free(v); free(text); tinyemb_free(h);
    return 0;
}

// 短文本不受影响:截断开关都应成功,且两者结果一致
static int test_short(void) {
    TinyEmb* h = tinyemb_load(DIR);
    CHECK(h != NULL);
    int dim = tinyemb_dim(h);
    float* a = malloc(dim*sizeof(float));
    float* b = malloc(dim*sizeof(float));

    const char* t = "北京的天气不错";
    CHECK(tinyemb_encode_text_ex(h, t, 1, a, dim) == 0);
    CHECK(tinyemb_encode_text_ex(h, t, 0, b, dim) == 0);
    for (int i=0;i<dim;i++) CHECK(a[i]==b[i]);   // 短文本两模式结果完全相同

    int cnt = tinyemb_token_count(h, t);
    CHECK(cnt == 9);                            // [CLS]+7字+[SEP]
    free(a); free(b); tinyemb_free(h);
    return 0;
}

// 批量:含超长句,truncate=1 应整体成功
static int test_batch(void) {
    TinyEmb* h = tinyemb_load(DIR);
    CHECK(h != NULL);
    int dim = tinyemb_dim(h);
    char* long_t = make_text(600);
    const char* texts[3] = { "北京的天气不错", long_t, "今天是星期天" };
    float* out = malloc(3*dim*sizeof(float));

    CHECK(tinyemb_encode_texts(h, texts, 3, out, 3*dim) == 0);
    CHECK(tinyemb_encode_texts_ex(h, texts, 3, 1, out, 3*dim) == 0);
    CHECK(tinyemb_encode_texts_ex(h, texts, 3, 0, out, 3*dim) == -1);   // 长句存在 → 报错

    for (int r=0;r<3;r++) {
        float norm=0;
        for (int i=0;i<dim;i++){ CHECK(isfinite(out[r*dim+i])); norm+=out[r*dim+i]*out[r*dim+i]; }
        CHECK(fabsf(sqrtf(norm)-1.0f) < 1e-3f);
    }
    free(out); free(long_t); tinyemb_free(h);
    return 0;
}

// 截断后向量 = 对应前缀文本的向量(HF truncation=True 语义:只看前 510 个 token)
static int test_trunc_matches_prefix(void) {
    TinyEmb* h = tinyemb_load(DIR);
    CHECK(h != NULL);
    int dim = tinyemb_dim(h);

    // 长文本:600 个 "的",截断后 = [CLS] + 510个"的" + [SEP]
    char* long_t = make_text(600);
    // 构造等价短文本:510 个 "的"(截断后实际内容)
    char* short_t = make_text(510);
    float* a = malloc(dim*sizeof(float));
    float* b = malloc(dim*sizeof(float));
    CHECK(tinyemb_encode_text(h, long_t, a, dim) == 0);
    CHECK(tinyemb_encode_text(h, short_t, b, dim) == 0);

    // 两者 token 序列相同 → 向量必须逐位相等
    for (int i=0;i<dim;i++) CHECK(a[i]==b[i]);
    free(a); free(b); free(long_t); free(short_t); tinyemb_free(h);
    return 0;
}

static int test_batch_matches_single(void);
static int test_batch_full_length(void);

int main(void) {
    RUN(test_overlong);
    RUN(test_short);
    RUN(test_batch);
    RUN(test_trunc_matches_prefix);
    RUN(test_batch_matches_single);
    RUN(test_batch_full_length);
    printf("\n===== 模型层 truncate: %d 通过, %d 失败 =====\n", g_pass, g_fail);
    return g_fail==0?0:1;
}

// 测 5:批量 vs 单条一致性 —— batch 里每句的结果必须 == 单独 encode 同一句
// (证明句间注意力真隔离 + 位置编码每句从 0 计数,即 bert_encode_batch 修复的正确性)
static int test_batch_matches_single(void) {
    TinyEmb* h = tinyemb_load(DIR);
    CHECK(h != NULL);
    int dim = tinyemb_dim(h);

    char* long_t = make_text(600);          // 截断到 512
    const char* texts[3] = { "北京的天气不错", long_t, "今天是星期天" };

    float* batch = malloc(3*dim*sizeof(float));
    float* single = malloc(dim*sizeof(float));
    CHECK(tinyemb_encode_texts(h, texts, 3, batch, 3*dim) == 0);

    for (int s = 0; s < 3; s++) {
        CHECK(tinyemb_encode_text(h, texts[s], single, dim) == 0);
        for (int i = 0; i < dim; i++) {
            // 批量矩阵乘的浮点累加顺序与单条不同,结果差 1 ULP(~6e-8)是正常的,
            // 故用容差比较(而非位级相等)。这里实测最大差 ~6e-8,取 1e-5 足够宽松。
            CHECK(fabsf(batch[s*dim + i] - single[i]) < 1e-5f);
        }
    }
    free(batch); free(single); free(long_t); tinyemb_free(h);
    return 0;
}

// 测 6:批量全是满长 512 的句子,压内存边界(seq² 注意力内存)
// 4 句 × 512 = 2048 token → 若 bert_layer 按 total² 分配,会到 ~1.6GB 量级。
// 通过即证明批量护栏"每句<=max_seq"在真实负载下可用,不 OOM 不崩。
static int test_batch_full_length(void) {
    TinyEmb* h = tinyemb_load(DIR);
    CHECK(h != NULL);
    int dim = tinyemb_dim(h);
    const int n = 4;
    char* t = make_text(600);               // 每句都截断到 512
    const char* texts[4] = { t, t, t, t };
    float* out = malloc((size_t)n*dim*sizeof(float));
    CHECK(tinyemb_encode_texts(h, texts, n, out, n*dim) == 0);
    for (int r = 0; r < n; r++) {
        float norm = 0;
        for (int i = 0; i < dim; i++) { CHECK(isfinite(out[r*dim+i])); norm += out[r*dim+i]*out[r*dim+i]; }
        CHECK(fabsf(sqrtf(norm)-1.0f) < 1e-3f);
    }
    free(out); free(t); tinyemb_free(h);
    return 0;
}
