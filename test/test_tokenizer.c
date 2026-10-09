// 单元测试:WordPiece 分词器
// 基准来自 Python transformers 对拍结果(北京的天气不错 → [101,1266,776,4638,1921,3698,679,7231,102])
#include "io/tokenizer.h"

#include <assert.h>
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

#define VOCAB "models/bge-small-zh-v1.5/vocab.txt"

// 测 1:词表加载,基本 token 的 id 正确
static int test_vocab_load(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);
    CHECK(wp.n_tokens == 21128);            // 词表大小
    CHECK(wp_token_to_id(&wp, "[PAD]") == 0);
    CHECK(wp_token_to_id(&wp, "[UNK]") == 100);
    CHECK(wp_token_to_id(&wp, "[CLS]") == 101);
    CHECK(wp_token_to_id(&wp, "[SEP]") == 102);
    // 中文在 BERT 里按单字切分,词表里是单字。已对拍验证:
    CHECK(wp_token_to_id(&wp, "北") == 1266);
    CHECK(wp_token_to_id(&wp, "京") == 776);
    CHECK(wp_token_to_id(&wp, "的") == 4638);
    // "北京" 整词不在词表里,应返回 -1
    CHECK(wp_token_to_id(&wp, "北京") == -1);
    wp_free(&wp);
    return 0;
}

// 测 2:中文句子完整编码(核心基准)
static int test_encode_chinese(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    int ids[64];
    size_t n = 0;
    CHECK(wp_encode(&wp, "北京的天气不错", ids, &n, 64) == 0);

    const int expect[] = { 101, 1266, 776, 4638, 1921, 3698, 679, 7231, 102 };
    CHECK(n == sizeof(expect)/sizeof(expect[0]));
    for (size_t i = 0; i < n; i++) {
        if (ids[i] != expect[i]) {
            printf("       第 %zu 个:得到 %d,期望 %d\n", i, ids[i], expect[i]);
            CHECK(ids[i] == expect[i]);
        }
    }
    wp_free(&wp);
    return 0;
}

// 测 3:空字符串 / 只有特殊标记的情况
static int test_empty(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    int ids[64];
    size_t n = 0;
    CHECK(wp_encode(&wp, "", ids, &n, 64) == 0);
    CHECK(n == 2);                          // 只有 [CLS] [SEP]
    CHECK(ids[0] == 101);
    CHECK(ids[1] == 102);
    wp_free(&wp);
    return 0;
}

// 测 4:英文 + 混合(验证不区分大小写 do_lower_case=false)
static int test_english(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    int ids[64];
    size_t n = 0;
    CHECK(wp_encode(&wp, "hello", ids, &n, 64) == 0);
    CHECK(n == 3);                          // [CLS] hello [SEP]
    CHECK(ids[0] == 101);
    CHECK(ids[2] == 102);
    // "hello" 若在词表里应是有效 id(>=0),否则是 [UNK]=100
    CHECK(ids[1] >= 0);
    wp_free(&wp);
    return 0;
}

int main(void) {
    RUN_TEST(test_vocab_load);
    RUN_TEST(test_encode_chinese);
    RUN_TEST(test_empty);
    RUN_TEST(test_english);

    printf("\n===== 结果: %d 通过, %d 失败 =====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
