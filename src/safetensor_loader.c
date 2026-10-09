#include "io/safetensor_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- 小工具 ---------- */

// 复制一个 C 字符串(C 标准库没有,自己造)
static char* xstrdup(const char* s) {
    size_t n = strlen(s) + 1;
    char* p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

// 解析 "整数个数" —— 把 shape 各维乘起来
static uint64_t count_numel(const TensorInfo* t) {
    uint64_t n = 1;
    for (int i = 0; i < t->shape_len; i++) n *= (uint64_t)t->shape[i];
    return n;
}

// 解析 [1,2,3];*p 是下标,进入时指向 '[',出来时停在 ']' 之后
// 返回 malloc 出来的数组,元素个数写进 *count;失败返回 NULL
static int64_t* parse_int_array(const char* s, size_t* p, int* count) {
    if (s[*p] != '[') return NULL;
    (*p)++;

    int cap = 4, n = 0;
    int64_t* a = malloc(cap * sizeof(int64_t));
    if (!a) return NULL;

    for (;;) {
        char* end = NULL;
        long long v = strtoll(s + *p, &end, 10);   // 自动跳过空白,读一个整数
        if (end == s + *p) {                        // 这里读不到数字
            if (s[*p] == '\0') { free(a); return NULL; }  // 防越界
            if (s[*p] == ']') { (*p)++; break; }    // 数组结束
            (*p)++;                                 // 跳过 ',' '[' 等
            continue;
        }
        if (n == cap) {                             // 满了就翻倍扩容
            int64_t* a2 = realloc(a, cap * 2 * sizeof(int64_t));
            if (!a2) { free(a); return NULL; }
            a = a2; cap *= 2;
        }
        a[n++] = (int64_t)v;
        *p = (size_t)(end - s);
    }
    *count = n;
    return a;
}

/* ---------- 主流程 ---------- */

int st_open(ST_Loader* L, const char* path) {
    memset(L, 0, sizeof(*L));                 // 全部清零,防止野指针
    L->path = xstrdup(path);

    FILE* f = fopen(path, "rb");              // 二进制模式
    if (!f) { fprintf(stderr, "打不开 %s\n", path); return -1; }

    // 第一段:8 字节头部长度(小端,直接读进 uint64_t 即可)
    uint64_t header_len = 0;
    if (fread(&header_len, 1, 8, f) != 8) {
        fprintf(stderr, "读头部长度失败\n"); fclose(f); return -1;
    }
    L->header_len = header_len;

    // 第二段:头部 JSON。多分配 1 字节放 '\0',
    // 这样它就是一个合法的 C 字符串,能用 strstr/strtoll 随便扫。
    char* header = malloc(header_len + 1);
    if (!header) { fclose(f); return -1; }
    if (fread(header, 1, header_len, f) != header_len) {
        fprintf(stderr, "读头部 JSON 失败\n");
        free(header); fclose(f); return -1;
    }
    header[header_len] = '\0';
    fclose(f);

    // 第三段:扫描
    size_t cap = 128;
    L->tensors = malloc(cap * sizeof(TensorInfo));
    if (!L->tensors) { free(header); return -1; }
    L->n_tensors = 0;

    const char* scan = header;
    const char* hit;
    while ((hit = strstr(scan, "\"dtype\":")) != NULL) {
        // 头部每条记录格式固定:
        //   "张量名":{"dtype":"F32","shape":[..],"data_offsets":[a,b]}
        //         ↑↑↑-3 -2 -1↑hit
        //   结束引号 -3，中间隔着 ':'(-2) 和 '{'(-1)，hit 是 "dtype" 的开头引号
        if (hit < header + 3) break;
        const char* key_close = hit - 3;               // 张量名结束引号
        const char* key_open  = key_close - 1;         // 从这里往回找起始引号
        while (key_open > header && *key_open != '"') key_open--;

        const char* val  = hit + 9;                    // 跳过 "dtype":"
        const char* val_end = strchr(val, '"');
        if (!val_end) break;

        const char* sh = strstr(val_end, "\"shape\":[");
        if (!sh) break;
        size_t idx = (size_t)(sh - header) + 8;        // 指向 '['
        int shape_len = 0;
        int64_t* shape = parse_int_array(header, &idx, &shape_len);

        const char* dp = strstr(val_end, "\"data_offsets\":[");
        if (!dp) { free(shape); break; }
        size_t idx2 = (size_t)(dp - header) + 15;      // 指向 '['
        int noff = 0;
        int64_t* offs = parse_int_array(header, &idx2, &noff);
        if (!shape || !offs || noff != 2) {
            free(shape); free(offs);
            fprintf(stderr, "解析失败\n");
            st_close(L); free(header); return -1;
        }

        if (L->n_tensors == cap) {                    // 张量表满就翻倍
            cap *= 2;
            TensorInfo* t2 = realloc(L->tensors, cap * sizeof(TensorInfo));
            if (!t2) { free(shape); free(offs); free(header); st_close(L); return -1; }
            L->tensors = t2;
        }

        TensorInfo* t = &L->tensors[L->n_tensors++];
        size_t nl = (size_t)(key_close - key_open - 1);   // 张量名长度
        t->name = malloc(nl + 1);
        memcpy(t->name, key_open + 1, nl);
        t->name[nl] = '\0';

        size_t dl = (size_t)(val_end - val);              // "F32" 的长度
        if (dl >= sizeof(t->dtype)) dl = sizeof(t->dtype) - 1;
        memcpy(t->dtype, val, dl);
        t->dtype[dl] = '\0';

        t->shape        = shape;
        t->shape_len    = shape_len;
        t->offset_begin = (uint64_t)offs[0];
        t->offset_end   = (uint64_t)offs[1];
        free(offs);

        scan = val_end + 1;                               // 继续往后扫
        // (__metadata__ 没有 "dtype",天然被跳过)
    }

    free(header);
    return 0;
}

void st_close(ST_Loader* L) {
    if (L->tensors) {
        for (size_t i = 0; i < L->n_tensors; i++) {
            free(L->tensors[i].name);
            free(L->tensors[i].shape);
        }
        free(L->tensors);
    }
    free(L->path);
    memset(L, 0, sizeof(*L));
}

const TensorInfo* st_find(const ST_Loader* L, const char* name) {
    for (size_t i = 0; i < L->n_tensors; i++)
        if (strcmp(L->tensors[i].name, name) == 0)   // strcmp==0 表示相等
            return &L->tensors[i];
    return NULL;
}

float* st_read_floats(const ST_Loader* L, const char* name, size_t* n) {
    const TensorInfo* t = st_find(L, name);
    if (!t)                   { fprintf(stderr, "没有张量 %s\n", name); return NULL; }
    if (strcmp(t->dtype, "F32") != 0) { fprintf(stderr, "%s 不是 F32\n", name); return NULL; }

    uint64_t count = count_numel(t);
    float* out = malloc(count * sizeof(float));
    if (!out) return NULL;

    FILE* f = fopen(L->path, "rb");
    if (!f) { free(out); return NULL; }
    // 关键:data_offsets 是相对"数据区起点"的偏移,不是相对文件开头!
    // 数据区起点 = 8 字节头部长度 + header_len 字节 JSON
    uint64_t data_start = 8 + L->header_len;
    fseek(f, (long)(data_start + t->offset_begin), SEEK_SET);
    if (fread(out, sizeof(float), count, f) != count) {
        fprintf(stderr, "读 %s 数据失败\n", name);
        fclose(f); free(out); return NULL;
    }
    fclose(f);
    *n = (size_t)count;
    return out;
}

void st_dump(const ST_Loader* L) {
    printf("header_len = %llu, tensors = %zu\n",
           (unsigned long long)L->header_len, L->n_tensors);
    printf("%-60s %-4s %12s %14s\n", "name", "type", "numel", "offset");
    for (size_t i = 0; i < L->n_tensors; i++) {
        const TensorInfo* t = &L->tensors[i];
        printf("%-60s %-4s %12llu %7llu..%7llu\n",
               t->name, t->dtype,
               (unsigned long long)count_numel(t),
               (unsigned long long)t->offset_begin,
               (unsigned long long)t->offset_end);
    }
}