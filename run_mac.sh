#!/bin/bash
# DeepResearch Mac 启动脚本 - OpenRouter 版本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ 环境变量已加载"
else
    echo "✗ .env 文件不存在"
    exit 1
fi

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ 虚拟环境已激活"
else
    echo "✗ 虚拟环境不存在"
    echo "  请运行: python3 -m venv venv && pip install -r requirements_mac.txt"
    exit 1
fi

# 验证 API Key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "✗ OPENROUTER_API_KEY 未设置"
    exit 1
fi
echo "✓ OPENROUTER_API_KEY 已设置"

# 配置
MODEL="${MODEL:-alibaba/tongyi-deepresearch-30b-a3b}"
DATASET="${DATASET:-inference/eval_data/example.jsonl}"
OUTPUT="${OUTPUT_PATH:-./outputs}"
MAX_ITEMS="${MAX_ITEMS:-1}"

echo ""
echo "========================================"
echo "DeepResearch - OpenRouter Edition"
echo "========================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo "Max Items: $MAX_ITEMS"
echo "========================================"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT"

# 运行
cd inference
python run_openrouter.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --max_items $MAX_ITEMS \
    --temperature 0.6

echo ""
echo "Done!"
