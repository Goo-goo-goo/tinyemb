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
# 关键:目标是老 x86 CPU(如 Intel J3160/Braswell),只支持到 SSE4.2,
# 不支持 AVX/AVX2/AVX-512。所以编译目标必须是 -msse4.2,绝不能用:
#   -march=native  → 换机器 Illegal instruction
#   -march=x86-64-v2 → 含 AVX,J3160 不支持,仍会 Illegal instruction
# 若确定部署机是新 CPU(支持 AVX2+),可加 -DCMAKE_C_FLAGS="-march=native" 提速。
include(CheckCCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  check_c_compiler_flag("-msse4.2" HAS_SSE42)
  if(HAS_SSE42)
    add_compile_options($<$<COMPILE_LANGUAGE:C>:-msse4.2>)
    message(STATUS "CPU 目标:SSE4.2(兼容 J3160/Braswell 等老 x86)")
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
# 指令集:用 ggml 官方的多变体编译(GGML_CPU_ALL_VARIANTS)。
# 这是"多指令集兼容 + 各用最优"的正确机制:
#   为每个 CPU 微架构编译独立变体(x64 兜底 / sse42 / haswell AVX2 / skylakex AVX512 ...)。
#   运行时按 CPU 自动选最优变体:
#     J3160/Braswell(仅SSE4.2) → 用 sse42 变体,不崩
#     Xeon 5218(AVX2)          → 用 haswell 变体,更快
#     更新 CPU(AVX512)         → 用 skylakex+ 变体,最快
#   x64 变体兜底,任何 x86 都能跑。
# 注意:GGML_CPU_ALL_VARIANTS 要求 GGML_BACKEND_DL(变体运行时动态加载)。
# 严禁单独开 GGML_AVX2 等(不隔离到变体,会把 AVX 编进公共代码,老 CPU 崩)。
set(GGML_BACKEND_DL        ON  CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS  ON  CACHE BOOL "" FORCE)
set(GGML_NATIVE            OFF CACHE BOOL "" FORCE)
# 关闭单独的指令集开关(交给多变体机制统一管理)
set(GGML_SSE42 OFF CACHE BOOL "" FORCE)
set(GGML_AVX   OFF CACHE BOOL "" FORCE)
set(GGML_AVX2  OFF CACHE BOOL "" FORCE)
set(GGML_BMI2  OFF CACHE BOOL "" FORCE)
set(GGML_FMA   OFF CACHE BOOL "" FORCE)
set(GGML_F16C  OFF CACHE BOOL "" FORCE)

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
  src/safetensor_loader.c
  src/ggml_backend_global.c)
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

tinyemb_add_test(test_ggml_ops test/test_ggml_ops.c src/ggml_ops.c src/ggml_backend_global.c)
target_link_libraries(test_ggml_ops PRIVATE ggml::ggml)

tinyemb_add_test(test_bert_layer test/test_bert_layer.c src/bert_layer.c src/ggml_backend_global.c)
target_link_libraries(test_bert_layer PRIVATE ggml::ggml)

tinyemb_add_test(test_bert_model
  test/test_bert_model.c
  src/bert_model.c src/bert_layer.c src/tokenizer.c src/safetensor_loader.c src/ggml_backend_global.c)
target_link_libraries(test_bert_model PRIVATE ggml::ggml)

# ---- 性能基准(无 ASan,Release 语义)----
add_executable(bench
  test/bench.c
  src/tinyemb.c src/bert_model.c src/bert_layer.c src/tokenizer.c src/safetensor_loader.c src/ggml_backend_global.c)
target_include_directories(bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench PRIVATE ggml::ggml)
set_target_properties(bench PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
