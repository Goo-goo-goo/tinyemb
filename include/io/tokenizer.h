// WordPiece 分词器(纯 C,零依赖)
// 对应 BERT 的 BertTokenizer + BasicTokenizer + WordpieceTokenizer
#pragma once
#include <stddef.h>

typedef struct {
    char** tokens;    // tokens[i] = 词表第 i 行(即 id = i)
    int*   ids;       // 冗余存一份 id(= 下标),方便遍历
    size_t n_tokens;  // 词表大小(行数)
} WordPiece;

// 加载 vocab.txt:行号(从 0 开始)就是 id。成功返回 0,失败 -1
int  wp_load(WordPiece* wp, const char* vocab_path);
void wp_free(WordPiece* wp);

// 查 id:找到返回 id,找不到返回 -1
int  wp_token_to_id(const WordPiece* wp, const char* token);

// 分词主入口:把一段文本变成 token 字符串序列,写进 out_tokens。
// *out_n 是输出个数。out_tokens 里每个字符串需要调用者逐个 free。
// max_out 是 out_tokens 的容量(防止越界)。
// 返回 0 成功,-1 出错。
int wp_tokenize(const WordPiece* wp,
                const char* text,
                char** out_tokens, size_t* out_n, size_t max_out);

// 便捷封装:直接输出 id 序列(自动加 [CLS]/[SEP])。
// ids 容量 max_out。返回 0 成功,-1 出错。
//
// truncate 决定超长(正文 token 数 > max_out-2)时的行为,语义对齐 HF truncation=True:
//   truncate != 0(默认):截断正文到 max_out-2 个,补 [SEP],总长恰好 = max_out。
//                       恒返回 0,[SEP] 永不丢失。
//   truncate == 0     :不截断,若正文放不下就返回 -1(ids/n_ids 未定义)。
//                       max_out < 2 时恒返回 -1。
// 未知 token 一律映射成 [UNK](id=100),不会写入 -1 之类的非法 id。
int wp_encode(const WordPiece* wp,
              const char* text,
              int* ids, size_t* n_ids, size_t max_out,
              int truncate);

// 释放 wp_tokenize 产生的一批字符串
void wp_free_tokens(char** tokens, size_t n);
