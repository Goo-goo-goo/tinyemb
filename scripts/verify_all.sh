#!/bin/bash
# 一键全量校验:单元测试 + HF 数值对拍(含截断)
# 用法: ./scripts/verify_all.sh
set -e
cd "$(dirname "$0")/.."
echo "==== 1. 单元/集成测试(带 ASan) ===="
for t in test_tokenizer test_truncate test_ggml_ops test_bert_layer step1; do
  printf "%-18s " "$t"; ./build/$t 2>/dev/null | grep -E "结果|通过" | tail -1
done
echo
echo "==== 2. HF 数值对拍(短文本 + 超长截断) ===="
./scripts/gen_hf_ref.sh
echo "-- 短文本 --"
./build/test_bert_model "北京的天气不错" /tmp/ref_short.txt 2>/dev/null | grep -E "对拍|PASS|FAIL"
LONG=$(python3 -c "print('北京是中国的首都,有着悠久的历史和灿烂的文化,故宫长城颐和园闻名天下,吸引着全世界的游客前来参观游览。'*80)")
echo "-- 超长文本(截断) --"
./build/test_bert_model "$LONG" /tmp/ref_long.txt 2>/dev/null | grep -E "对拍|PASS|FAIL"
echo
echo "==== 校验完成 ===="
