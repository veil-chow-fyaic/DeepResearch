"""
完整 Agent 测试 - 验证工具调用
"""
import os
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('../.env')

from openai import OpenAI
from tool_search import Search
from tool_visit import Visit
from tool_scholar import Scholar
import json

# 初始化
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    timeout=120.0,
)
search = Search()
visit = Visit()
scholar = Scholar()

SYSTEM_PROMPT = """You are a deep research assistant. You can use tools to search and visit webpages.

When you need to search, use this format:
Action: search
Action Input: {"query": ["your search query"]}
Observation: [search results will appear here]

When you need to visit a webpage:
Action: visit
Action Input: {"url": "https://example.com", "goal": "what you want to find"}
Observation: [page content will appear here]

When you have the answer, respond with:
<answer>your final answer</answer>

Today's date: 2026-03-19
"""

def run_research(question: str, max_rounds: int = 5):
    """运行深度研究"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}")
        print(f"{'='*50}")
        
        # 调用 API
        response = client.chat.completions.create(
            model="alibaba/tongyi-deepresearch-30b-a3b",
            messages=messages,
            temperature=0.6,
            max_tokens=4096,
        )
        
        content = response.choices[0].message.content
        if not content:
            print("Empty response!")
            continue
            
        print(f"Model: {content[:500]}..." if len(content) > 500 else f"Model: {content}")
        messages.append({"role": "assistant", "content": content})
        
        # 检查答案
        if '<answer>' in content and '</answer>' in content:
            answer = content.split('<answer>')[1].split('</answer>')[0]
            print(f"\n{'='*50}")
            print(f"FINAL ANSWER: {answer}")
            return answer
        
        # 检查工具调用
        if 'Action:' in content and 'Action Input:' in content:
            try:
                action = content.split('Action:')[1].split('Action Input:')[0].strip()
                input_str = content.split('Action Input:')[1].split('Observation:')[0].strip()
                
                # 解析参数
                try:
                    args = json.loads(input_str)
                except:
                    args = {"query": [input_str]}
                
                print(f"\n调用工具: {action}")
                print(f"参数: {args}")
                
                # 执行工具
                if action == "search":
                    result = search.call(args)
                elif action == "visit":
                    result = visit.call(args)
                elif action == "scholar" or action == "google_scholar":
                    result = scholar.call(args)
                else:
                    result = f"Unknown tool: {action}"
                
                obs = f"Observation: {result[:3000]}..." if len(str(result)) > 3000 else f"Observation: {result}"
                print(f"\n工具返回: {obs[:500]}...")
                messages.append({"role": "user", "content": obs})
                
            except Exception as e:
                print(f"工具调用错误: {e}")
                messages.append({"role": "user", "content": f"Observation: Error - {e}"})
        else:
            # 没有工具调用，提示继续
            messages.append({"role": "user", "content": "Please continue your research. Use tools if needed."})
    
    return "Max rounds reached"

if __name__ == "__main__":
    # 测试问题
    question = "What is the latest version of Claude AI model in 2025?"
    print(f"Question: {question}")
    result = run_research(question, max_rounds=5)
