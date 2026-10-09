// WordPiece 分词器实现
// 分三步:①基础分词(切词块) ②子词切分(贪心最长匹配) ③加特殊标记
#include "io/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================= 小工具 ================= */

static char* xstrndup(const char* s, size_t n) {
    char* p = malloc(n + 1);
    if (p) { memcpy(p, s, n); p[n] = '\0'; }
    return p;
}

static char* xstrdup(const char* s) {
    return xstrndup(s, strlen(s));
}

/* ================= 词表加载 ================= */

int wp_load(WordPiece* wp, const char* vocab_path) {
    wp->tokens = NULL; wp->ids = NULL; wp->n_tokens = 0;

    FILE* f = fopen(vocab_path, "rb");
    if (!f) { fprintf(stderr, "打不开词表 %s\n", vocab_path); return -1; }

    size_t cap = 256;
    wp->tokens = malloc(cap * sizeof(char*));
    wp->ids    = malloc(cap * sizeof(int));

    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        // 去掉行尾的 \n / \r
        size_t len = strlen(buf);
        while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) buf[--len] = '\0';

        if (wp->n_tokens == cap) {
            cap *= 2;
            wp->tokens = realloc(wp->tokens, cap * sizeof(char*));
            wp->ids    = realloc(wp->ids,    cap * sizeof(int));
        }
        wp->tokens[wp->n_tokens] = xstrndup(buf, len);
        wp->ids[wp->n_tokens]    = (int)wp->n_tokens;   // 行号 = id
        wp->n_tokens++;
    }
    fclose(f);
    return 0;
}

void wp_free(WordPiece* wp) {
    if (wp->tokens) {
        for (size_t i = 0; i < wp->n_tokens; i++) free(wp->tokens[i]);
        free(wp->tokens);
    }
    free(wp->ids);
    wp->tokens = NULL; wp->ids = NULL; wp->n_tokens = 0;
}

int wp_token_to_id(const WordPiece* wp, const char* token) {
    // 词表按 id 顺序存,这里线性查找。词表 2 万行,对单条句子足够快;
    // 以后要提速可换成哈希表。
    for (size_t i = 0; i < wp->n_tokens; i++)
        if (strcmp(wp->tokens[i], token) == 0)
            return wp->ids[i];
    return -1;
}

/* ================= 字符分类(UTF-8) ================= */

// 中文字符判定:粗略地把常见 CJK 区段当作"需要前后加空格的字符"
static int utf8_is_cjk(unsigned c) {
    return (c >= 0x4E00 && c <= 0x9FFF)   // 基本汉字
        || (c >= 0x3400 && c <= 0x4DBF)   // 扩展 A
        || (c >= 0x20000 && c <= 0x2A6DF); // 扩展 B(基本够用)
}

// 读取 text[*p] 处一个 UTF-8 字符,返回码点并前进 *p
static unsigned utf8_next(const char* text, size_t* p) {
    unsigned char c = (unsigned char)text[*p];
    unsigned cp;
    int extra;
    if (c < 0x80)             { cp = c;           extra = 0; }
    else if ((c >> 5) == 6)   { cp = c & 0x1F;    extra = 1; }
    else if ((c >> 4) == 14)  { cp = c & 0x0F;    extra = 2; }
    else if ((c >> 3) == 30)  { cp = c & 0x07;    extra = 3; }
    else                      { (*p)++; return 0xFFFD; } // 非法字节

    size_t i = *p + 1;
    for (int k = 0; k < extra; k++) {
        unsigned char cc = (unsigned char)text[i];
        if ((cc >> 6) != 2) { (*p)++; return 0xFFFD; }
        cp = (cp << 6) | (cc & 0x3F);
        i++;
    }
    *p = i;
    return cp;
}

static int is_ascii_punct(unsigned c) {
    return (c >= 33 && c <= 47) || (c >= 58 && c <= 64)
        || (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
}

static int is_whitespace_cp(unsigned c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x3000;
}

/* ================= ①基础分词:切成词块 ================= */

// 把一段文本切成"词块"(不做子词切分),输出到 out,返回个数
static size_t basic_tokenize(const char* text, char** out, size_t max_out) {
    // 第一步:逐字符处理,把中文字符、ASCII 标点前后加空格,统一空白
    char* tmp = malloc(strlen(text) * 3 + 8);  // 最坏每字节变 3 字符,足够
    size_t ti = 0;
    size_t p = 0;
    while (text[p]) {
        size_t before = p;
        unsigned cp = utf8_next(text, &p);
        size_t clen = p - before;

        if (cp == 0) break;

        if (is_whitespace_cp(cp)) {
            tmp[ti++] = ' ';
        } else if (utf8_is_cjk(cp) || is_ascii_punct(cp)) {
            // 中文字符 / 标点:前后加空格,让它独立成块
            tmp[ti++] = ' ';
            memcpy(tmp + ti, text + before, clen); ti += clen;
            tmp[ti++] = ' ';
        } else {
            memcpy(tmp + ti, text + before, clen); ti += clen;
        }
    }
    tmp[ti] = '\0';

    // 第二步:按空格切分(连续空格视为一个分隔)
    size_t n = 0;
    size_t start = 0;
    int in_tok = 0;
    for (size_t i = 0; i <= ti; i++) {
        if (i < ti && tmp[i] != ' ') {
            if (!in_tok) { start = i; in_tok = 1; }
        } else {
            if (in_tok) {
                if (n < max_out) out[n++] = xstrndup(tmp + start, i - start);
                in_tok = 0;
            }
        }
    }
    free(tmp);
    return n;
}

/* ================= ②子词切分:贪心最长匹配 ================= */

// 对单个词块做 WordPiece 切分,结果追加到 out。返回新增个数。
static size_t wordpiece_split(const WordPiece* wp,
                              const char* word, char** out, size_t max_out) {
    size_t wlen = strlen(word);
    if (wlen == 0) return 0;

    char* buf = malloc(wlen * 4 + 8);   // "##" 前缀留余量
    size_t n = 0;

    size_t start = 0;
    int first = 1;
    while (start < wlen) {
        size_t end = wlen;
        int found = 0;
        char* piece = NULL;

        // 从最长开始试,直到找到词表里的子词(至少留 1 字符)
        while (start < end) {
            size_t plen = end - start;
            size_t off = 0;
            if (!first) { memcpy(buf, "##", 2); off = 2; }  // 非首段加 ##
            memcpy(buf + off, word + start, plen);
            buf[off + plen] = '\0';

            if (wp_token_to_id(wp, buf) >= 0) {
                piece = xstrndup(buf, off + plen);
                found = 1;
                break;
            }
            // 收缩一个 UTF-8 字符(不能切坏多字节)
            end--;
            while (end > start && ((unsigned char)word[end] & 0xC0) == 0x80) end--;
        }

        if (!found) {
            // 整块失败 → 整块变 [UNK]
            for (size_t i = 0; i < n; i++) free(out[i]);
            n = 0;
            if (n < max_out) out[n++] = xstrdup("[UNK]");
            free(buf);
            return n;
        }
        if (n < max_out) out[n++] = piece;
        start = end;
        first = 0;
    }
    free(buf);
    return n;
}

/* ================= 主入口 ================= */

int wp_tokenize(const WordPiece* wp,
                const char* text,
                char** out_tokens, size_t* out_n, size_t max_out) {
    // 先基础分词(上限放宽,词块数一般不多)
    size_t cap = 512;
    char** chunks = malloc(cap * sizeof(char*));
    size_t n_chunks = basic_tokenize(text, chunks, cap);

    // 再对每个词块做子词切分
    size_t n = 0;
    for (size_t i = 0; i < n_chunks; i++) {
        n += wordpiece_split(wp, chunks[i], out_tokens + n, max_out - n);
        free(chunks[i]);
    }
    free(chunks);

    *out_n = n;
    return 0;
}

int wp_encode(const WordPiece* wp,
              const char* text,
              int* ids, size_t* n_ids, size_t max_out) {
    char** toks = malloc(max_out * sizeof(char*));
    size_t n = 0;
    wp_tokenize(wp, text, toks, &n, max_out);

    size_t m = 0;
    if (m < max_out) ids[m++] = wp_token_to_id(wp, "[CLS]");   // 101
    for (size_t i = 0; i < n && m < max_out; i++) {
        ids[m++] = wp_token_to_id(wp, toks[i]);
    }
    if (m < max_out) ids[m++] = wp_token_to_id(wp, "[SEP]");   // 102

    wp_free_tokens(toks, n);
    free(toks);
    *n_ids = m;
    return 0;
}

void wp_free_tokens(char** tokens, size_t n) {
    for (size_t i = 0; i < n; i++) free(tokens[i]);
}
