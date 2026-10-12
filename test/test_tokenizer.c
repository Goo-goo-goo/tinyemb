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
    CHECK(wp_encode(&wp, "北京的天气不错", ids, &n, 64, 1) == 0);

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
    CHECK(wp_encode(&wp, "", ids, &n, 64, 1) == 0);
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
    CHECK(wp_encode(&wp, "hello", ids, &n, 64, 1) == 0);
    CHECK(n == 3);                          // [CLS] hello [SEP]
    CHECK(ids[0] == 101);
    CHECK(ids[2] == 102);
    // "hello" 若在词表里应是有效 id(>=0),否则是 [UNK]=100
    CHECK(ids[1] >= 0);
    wp_free(&wp);
    return 0;
}

// ==================== 截断(truncate)测试 ====================

// 造一段长度确定的中文正文:每个汉字 1 个 token,共 n 个汉字。
// 用词表里确定存在的单字拼(选 "的"=4638,最常见),避免 [UNK] 干扰。
static char* make_text(size_t n_chars) {
    // "的" 是 UTF-8 三字节
    char* s = malloc(n_chars * 3 + 1);
    for (size_t i = 0; i < n_chars; i++) memcpy(s + i * 3, "\xe7\x9a\x84", 3);
    s[n_chars * 3] = '\0';
    return s;
}

// 测 5:truncate=1,超长自动截断到 max_out,且 [SEP] 恒在
static int test_truncate_long(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    // 造 300 个汉字 → 正文 300 token,超过 max_out=10 的容量
    char* text = make_text(300);
    int ids[10];
    size_t n = 0;
    CHECK(wp_encode(&wp, text, ids, &n, 10, 1) == 0);
    CHECK(n == 10);                 // 截断后恰好占满
    CHECK(ids[0] == 101);           // [CLS]
    CHECK(ids[9] == 102);           // [SEP] 恒在(这是核心回归点)
    for (size_t i = 1; i < 9; i++) CHECK(ids[i] == 4638);   // 正文全是 "的"
    free(text);
    wp_free(&wp);
    return 0;
}

// 测 6:truncate=0,超长报 -1(不静默截断)
static int test_no_truncate_rejects(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    char* text = make_text(300);
    int ids[10];
    size_t n = 999;
    CHECK(wp_encode(&wp, text, ids, &n, 10, 0) == -1);
    free(text);
    wp_free(&wp);
    return 0;
}

// 测 7:刚好放得下(不截断边界)—— n_chars = max_out-2 时恒不截断
static int test_exact_fit(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    // max_out=10 → 正文最多 8 个。造 8 个汉字,刚好放满。
    char* text = make_text(8);
    int ids[10];
    size_t n = 0;
    CHECK(wp_encode(&wp, text, ids, &n, 10, 1) == 0);
    CHECK(n == 10);
    CHECK(ids[0] == 101);
    CHECK(ids[9] == 102);

    // 而且 truncate=0 在"刚好放得下"时也应该成功(不报错)
    size_t n2 = 0;
    CHECK(wp_encode(&wp, text, ids, &n2, 10, 0) == 0);
    CHECK(n2 == 10);
    free(text);
    wp_free(&wp);
    return 0;
}

// 测 8:超 1 个(9 个汉字 → 正文 9 > 8)就必须截断,truncate=0 报错
static int test_overflow_by_one(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    char* text = make_text(9);
    int ids[10];
    size_t n = 0;
    CHECK(wp_encode(&wp, text, ids, &n, 10, 1) == 0);
    CHECK(n == 10);
    CHECK(ids[0] == 101);
    CHECK(ids[9] == 102);

    size_t n2 = 0;
    CHECK(wp_encode(&wp, text, ids, &n2, 10, 0) == -1);
    free(text);
    wp_free(&wp);
    return 0;
}

// 测 9:max_out 边界值:2(只剩 [CLS][SEP],无正文)、1、0
static int test_max_out_boundary(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    int ids[8];
    size_t n = 0;

    // max_out=2:只能放 [CLS][SEP],正文被截到 0,但 [SEP] 仍必须在
    char* text = make_text(50);
    CHECK(wp_encode(&wp, text, ids, &n, 2, 1) == 0);
    CHECK(n == 2);
    CHECK(ids[0] == 101);
    CHECK(ids[1] == 102);

    // max_out=1 / 0:放不下 [CLS]+[SEP],恒报 -1
    CHECK(wp_encode(&wp, text, ids, &n, 1, 1) == -1);
    CHECK(wp_encode(&wp, text, ids, &n, 0, 1) == -1);
    free(text);
    wp_free(&wp);
    return 0;
}

// 测 10:未知 token 兜底成 [UNK],不写入 -1
static int test_unk_fallback(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    // 一串 emoji(词表肯定没有),每个会被整块变 [UNK]
    int ids[16];
    size_t n = 0;
    CHECK(wp_encode(&wp, "\xf0\x9f\x98\x80\xf0\x9f\x98\x81", ids, &n, 16, 1) == 0);
    CHECK(n >= 3);                      // 至少 [CLS] [UNK]... [SEP]
    CHECK(ids[0] == 101);
    CHECK(ids[n-1] == 102);
    for (size_t i = 0; i < n; i++) {
        CHECK(ids[i] >= 0);             // 绝不能是 -1(会越界读 word_emb)
        CHECK(ids[i] < (int)wp.n_tokens);
    }
    wp_free(&wp);
    return 0;
}

// 测 11:截断后的内容 = 未截断序列的前缀(HF truncation=True 语义)
static int test_truncate_is_prefix(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    char* text = make_text(20);         // 正文 20 个 "的"

    // 大缓冲拿全序列(含 [CLS][SEP])
    int full[32];
    size_t n_full = 0;
    CHECK(wp_encode(&wp, text, full, &n_full, 32, 1) == 0);
    CHECK(n_full == 22);                // 20 正文 + CLS + SEP

    // 小缓冲截断到 10
    int trunc[10];
    size_t n_trunc = 0;
    CHECK(wp_encode(&wp, text, trunc, &n_trunc, 10, 1) == 0);
    CHECK(n_trunc == 10);

    // 截断序列 = [CLS] + full[1..8] + [SEP],即正文是 full 的前 8 个
    CHECK(trunc[0] == full[0]);
    for (size_t i = 1; i < 9; i++) CHECK(trunc[i] == full[i]);
    CHECK(trunc[9] == full[n_full - 1]);   // [SEP] 与全序列末尾一致
    free(text);
    wp_free(&wp);
    return 0;
}

// 测 12:空文本 + 截断,仍应得到合法的 [CLS][SEP]
static int test_truncate_empty(void) {
    WordPiece wp;
    CHECK(wp_load(&wp, VOCAB) == 0);

    int ids[4];
    size_t n = 0;
    CHECK(wp_encode(&wp, "", ids, &n, 2, 1) == 0);
    CHECK(n == 2);
    CHECK(ids[0] == 101);
    CHECK(ids[1] == 102);
    wp_free(&wp);
    return 0;
}

int main(void) {
    RUN_TEST(test_vocab_load);
    RUN_TEST(test_encode_chinese);
    RUN_TEST(test_empty);
    RUN_TEST(test_english);

    // 截断语义
    RUN_TEST(test_truncate_long);
    RUN_TEST(test_no_truncate_rejects);
    RUN_TEST(test_exact_fit);
    RUN_TEST(test_overflow_by_one);
    RUN_TEST(test_max_out_boundary);
    RUN_TEST(test_unk_fallback);
    RUN_TEST(test_truncate_is_prefix);
    RUN_TEST(test_truncate_empty);

    printf("\n===== 结果: %d 通过, %d 失败 =====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
