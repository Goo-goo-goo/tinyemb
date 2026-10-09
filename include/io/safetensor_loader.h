#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct {
    char*    name;          // 需要 free
    char     dtype[8];      // "F32" 或 "I64",定长数组放得下就行
    int64_t* shape;         // 需要 free
    int      shape_len;
    uint64_t offset_begin;
    uint64_t offset_end;
} TensorInfo;

typedef struct {
    char*       path;       // 需要 free
    uint64_t    header_len;
    TensorInfo* tensors;    // 需要 free(里面每项的 name/shape 也要 free)
    size_t      n_tensors;
} ST_Loader;

int           st_open(ST_Loader* L, const char* path);   // 0=成功, -1=失败
void          st_close(ST_Loader* L);                    // 释放全部内存
void          st_dump(const ST_Loader* L);               // 打印目录
const TensorInfo* st_find(const ST_Loader* L, const char* name);
float*        st_read_floats(const ST_Loader* L, const char* name, size_t* n);
                                                          // 调用者负责 free()