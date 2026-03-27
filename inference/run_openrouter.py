#!/usr/bin/env python3
"""
DeepResearch OpenRouter Runner
"""
import argparse
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm

# 加载 .env 文件
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from react_agent_openrouter import OpenRouterReactAgent


def load_data(filepath: str) -> list:
    """加载评估数据"""
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError("Input JSON must be a list")
    elif filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError("Use .json or .jsonl")
    return items


def main():
    parser = argparse.ArgumentParser(description="DeepResearch OpenRouter Runner")
    parser.add_argument("--model", type=str, default="alibaba/tongyi-deepresearch-30b-a3b")
    parser.add_argument("--dataset", type=str, default="eval_data/example.jsonl")
    parser.add_argument("--output", type=str, default="../outputs")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    # 验证 API Key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"results_{timestamp}.jsonl")
    artifact_dir = os.path.join(args.output, f"artifacts_{timestamp}")
    os.makedirs(artifact_dir, exist_ok=True)
    os.environ.setdefault("DEEP_RESEARCH_KEEP_MESSAGES", "0")
    os.environ["DEEP_RESEARCH_RESULT_ARTIFACT_DIR"] = artifact_dir

    # 加载数据
    print(f"Loading: {args.dataset}")
    items = load_data(args.dataset)
    if args.max_items:
        items = items[:args.max_items]
    print(f"Total items: {len(items)}")

    # 初始化 Agent
    llm_cfg = {
        "model": args.model,
        "generate_cfg": {"temperature": args.temperature}
    }
    agent = OpenRouterReactAgent(llm_cfg=llm_cfg)

    # 处理
    for i, item in enumerate(tqdm(items, desc="Processing")):
        print(f"\n{'='*50}")
        print(f"Item {i+1}/{len(items)}: {item.get('question', '')[:100]}...")
        
        task = {"item": item}
        try:
            result = agent._run(task)
            result["item_idx"] = i
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Result: {result.get('prediction', '')[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"item_idx": i, "error": str(e)}, ensure_ascii=False) + "\n")

    print(f"\nDone! Results saved to: {output_file}")


if __name__ == "__main__":
    main()
