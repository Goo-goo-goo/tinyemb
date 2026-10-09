cmake_minimum_required(VERSION 3.20)
# 通用 x86 CPU 构建:不绑定任何特定 CPU/Mac,纯 CPU 推理。
# 语言:C 为主 + C++(ggml 部分用);ASM 在 Apple Silicon 才需要(Metal 汇编),x86 用不到。
project(tinyembbedding C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # 默认 Release(推理必须开优化)
endif()

# ---- 可移植的 CPU 优化 ----
# 不用 -march=native(在编译机上最优但换 CPU 可能非法指令崩溃)。
# 用 x86-64-v2 基线(SSE4.2/SSSE3,2009+ 的 CPU 都支持,含 J3160/Braswell)。
# 想要更强可在部署机上加 -DCMAKE_C_FLAGS="-march=native"。
include(CheckCCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  check_c_compiler_flag("-march=x86-64-v2" HAS_X86V2)
  if(HAS_X86V2)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-march=x86-64-v2>)
    message(STATUS "CPU 目标:x86-64-v2(通用 x86 基线)")
  endif()
endif()

# --- ggml 设置(必须在 add_subdirectory 之前)---
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CPU_KLEIDIAI   OFF CACHE BOOL "" FORCE)   # 关掉联网下载,保证离线可编译
# 纯 CPU:强制关掉所有加速后端,让 ggml::ggml 只带 ggml-cpu。
# 这样在任意 x86 Linux/NAS 上都不会因缺 Metal/CUDA/BLAS 依赖而失败。
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA  OFF CACHE BOOL "" FORCE)
set(GGML_BLAS  OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_SDL   OFF CACHE BOOL "" FORCE)
# 允许 ggml 在运行时按 CPU 分发内核(x86 上按 SSE/AVX2/AVX512 分发,但缺指令集会回退)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml)

# ggml-metal 的 <memory> 预包含 hack:只在真的构建了 Metal(即 Apple)时才需要。
# x86 纯 CPU 不会生成 ggml-metal target,用 if(TARGET) 守卫,否则 configure 就报错。
if(TARGET ggml-metal)
  target_compile_options(ggml-metal PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-include;memory>)
endif()

# ---- 通用小工具函数:建一个测试可执行(带 ASan)----
function(tinyemb_add_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  set_target_properties(${name} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  # ASan 只给测试用(生产库 tinyemb 不加)
  target_compile_options(${name} PRIVATE -fsanitize=address -g)
  target_link_options(${name} PRIVATE -fsanitize=address)
endfunction()

# ---- 生产共享库(无 ASan,通用 x86)----
# 供 Python(ctypes)/其他语言调用的推理引擎。
add_library(tinyemb SHARED
  src/tinyemb.c
  src/bert_model.c
  src/bert_layer.c
  src/tokenizer.c
  src/safetensor_loader.c)
target_include_directories(tinyemb PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tinyemb PRIVATE ggml::ggml)
set_target_properties(tinyemb PROPERTIES
  C_STANDARD 11 C_STANDARD_REQUIRED ON
  C_VISIBILITY_PRESET default
  # Linux 上叫 libtinyemb.so,macOS 上叫 libtinyemb.dylib,CMake 自动处理后缀
  POSITION_INDEPENDENT_CODE ON)

# ---- 测试目标 ----
tinyemb_add_test(step1 test/test_safetensor_loader.c src/safetensor_loader.c)

tinyemb_add_test(test_tokenizer test/test_tokenizer.c src/tokenizer.c)

tinyemb_add_test(test_ggml_ops test/test_ggml_ops.c src/ggml_ops.c)
target_link_libraries(test_ggml_ops PRIVATE ggml::ggml)

tinyemb_add_test(test_bert_layer test/test_bert_layer.c src/bert_layer.c)
target_link_libraries(test_bert_layer PRIVATE ggml::ggml)

tinyemb_add_test(test_bert_model
  test/test_bert_model.c
  src/bert_model.c src/bert_layer.c src/tokenizer.c src/safetensor_loader.c)
target_link_libraries(test_bert_model PRIVATE ggml::ggml)

# ---- 性能基准(无 ASan,Release 语义)----
add_executable(bench
  test/bench.c
  src/tinyemb.c src/bert_model.c src/bert_layer.c src/tokenizer.c src/safetensor_loader.c)
target_include_directories(bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench PRIVATE ggml::ggml)
set_target_properties(bench PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
