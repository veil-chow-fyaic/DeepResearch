"""
DeepResearch Agent - 简化版 (不依赖 sandbox_fusion)
用于快速测试 OpenRouter API 连接
"""
import json
import json5
import os
from typing import Dict, List, Optional
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from datetime import datetime
import time
import random

from prompt import SYSTEM_PROMPT
from tool_search import Search
from tool_visit import Visit
from tool_scholar import Scholar

# 初始化工具
search_tool = Search()
visit_tool = Visit()
scholar_tool = Scholar()

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))


class SimpleDeepResearchAgent:
    """简化版 DeepResearch Agent"""

    def __init__(self, model: str = "alibaba/tongyi-deepresearch-30b-a3b"):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=600.0,
        )
        print(f"✓ Agent initialized: {self.model}")

    def call_server(self, msgs, max_tries=5):
        """调用 OpenRouter API"""
        for attempt in range(max_tries):
            try:
                print(f"--- API Call {attempt + 1}/{max_tries} ---")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=0.6,
                    max_tokens=10000,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
            except Exception as e:
                print(f"Error: {e}")
                if attempt < max_tries - 1:
                    time.sleep(min(2 ** attempt, 10))
        return "API Error"

    def call_tool(self, tool_name: str, args: dict) -> str:
        """调用工具"""
        try:
            if tool_name == "search":
                return search_tool.call(args)
            elif tool_name == "visit":
                return visit_tool.call(args)
            elif tool_name == "scholar" or tool_name == "google_scholar":
                return scholar_tool.call(args)
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Tool error: {e}"

    def run(self, question: str) -> dict:
        """执行深度研究"""
        start_time = time.time()
        system_prompt = SYSTEM_PROMPT + datetime.now().strftime("%Y-%m-%d")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        for round_num in range(1, MAX_LLM_CALL_PER_RUN + 1):
            if time.time() - start_time > 300:  # 5分钟超时
                return {"question": question, "prediction": "Timeout", "termination": "timeout"}
            
            content = self.call_server(messages)
            print(f"\n=== Round {round_num} ===")
            print(content[:500] + "..." if len(content) > 500 else content)
            
            messages.append({"role": "assistant", "content": content})
            
            # 检查答案
            if '<answer>' in content and '</answer>' in content:
                answer = content.split('<answer>')[1].split('</answer>')[0]
                return {
                    "question": question,
                    "prediction": answer,
                    "termination": "answer",
                    "rounds": round_num
                }
            
            # 检查工具调用 (ReAct 格式)
            if 'Action:' in content and 'Action Input:' in content:
                try:
                    action = content.split('Action:')[1].split('Action Input:')[0].strip()
                    input_str = content.split('Action Input:')[1].split('Observation:')[0].strip()
                    try:
                        args = json5.loads(input_str)
                    except:
                        args = {"query": input_str}
                    
                    result = self.call_tool(action, args)
                    obs = f"Observation: {result[:5000]}..." if len(str(result)) > 5000 else f"Observation: {result}"
                    messages.append({"role": "user", "content": obs})
                except Exception as e:
                    messages.append({"role": "user", "content": f"Observation: Error - {e}"})
        
        return {"question": question, "prediction": "Max rounds reached", "termination": "max_rounds"}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('../.env')
    
    agent = SimpleDeepResearchAgent()
    result = agent.run("What is the capital of France?")
    print("\n=== Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
