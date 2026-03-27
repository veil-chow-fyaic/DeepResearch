"""
DeepResearch Agent - OpenRouter API Version
根据 README 修改，支持通过 OpenRouter 调用 Tongyi-DeepResearch-30B-A3B
"""
import json
import json5
import os
import hashlib
from typing import Dict, List, Optional, Union
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from datetime import datetime
import time
import asyncio
import random

from prompt import *
from tool_file import FileParser
from tool_scholar import Scholar
from tool_search import Search
from tool_visit import Visit

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

ENABLE_PYTHON_INTERPRETER = os.getenv("ENABLE_PYTHON_INTERPRETER", "").strip().lower() in {"1", "true", "yes"}
KEEP_FULL_MESSAGES = os.getenv("DEEP_RESEARCH_KEEP_MESSAGES", "").strip().lower() in {"1", "true", "yes"}
RESULT_ARTIFACT_DIR = os.getenv("DEEP_RESEARCH_RESULT_ARTIFACT_DIR", "").strip()


def build_tool_class():
    tools = [
        FileParser(),
        Scholar(),
        Visit(),
        Search(),
    ]
    if ENABLE_PYTHON_INTERPRETER:
        try:
            from tool_python import PythonInterpreter, SANDBOX_FUSION_IMPORT_ERROR
            if SANDBOX_FUSION_IMPORT_ERROR is None:
                tools.append(PythonInterpreter())
        except Exception:
            pass
    return tools


TOOL_CLASS = build_tool_class()
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


def today_date():
    return datetime.now().strftime("%Y-%m-%d")


class OpenRouterReactAgent:
    """使用 OpenRouter API 的 DeepResearch Agent"""

    def __init__(self, llm_cfg: dict):
        self.llm_generate_cfg = llm_cfg.get("generate_cfg", {})
        self.model = llm_cfg.get("model", "alibaba/tongyi-deepresearch-30b-a3b")

        # OpenRouter 配置
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=600.0,
        )
        print(f"Initialized OpenRouter client with model: {self.model}")

    def call_server(self, msgs, max_tries=10):
        """调用 OpenRouter API"""
        base_sleep_time = 1
        for attempt in range(max_tries):
            try:
                print(f"--- Attempt {attempt + 1}/{max_tries} ---")

                chat_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    max_tokens=10000,
                )

                content = chat_response.choices[0].message.content

                # OpenRouter 可能返回 reasoning 字段
                try:
                    reasoning = getattr(chat_response.choices[0].message, 'reasoning', None)
                    if reasoning:
                        content = "<think\\>\n" + reasoning.strip() + "\n</think\\>\n" + content
                except:
                    pass

                if content and content.strip():
                    print("--- Service call successful ---")
                    return content.strip()
                else:
                    print(f"Warning: Empty response on attempt {attempt + 1}")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"API Error on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")

            if attempt < max_tries - 1:
                sleep_time = min(base_sleep_time * (2 ** attempt) + random.uniform(0, 1), 30)
                print(f"Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)

        return "Server error: All retries exhausted"

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        """调用工具"""
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                result = raw_result
                if not isinstance(raw_result, str):
                    result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result
        else:
            return f"Error: Tool {tool_name} not found"

    def _persist_message_artifact(self, question: str, messages: list[dict]) -> str:
        if not RESULT_ARTIFACT_DIR:
            return ""
        try:
            os.makedirs(RESULT_ARTIFACT_DIR, exist_ok=True)
            question_hash = hashlib.sha1((question or "").encode("utf-8")).hexdigest()[:12]
            artifact_path = os.path.join(RESULT_ARTIFACT_DIR, f"{question_hash}.messages.json")
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            return artifact_path
        except Exception:
            return ""

    def _build_result(self, question: str, answer: str, messages: list[dict], prediction: str, termination: str) -> dict:
        artifact_path = self._persist_message_artifact(question, messages)
        result = {
            "question": question,
            "answer": answer,
            "prediction": prediction,
            "termination": termination,
            "research_state": {
                "messages_count": len(messages or []),
                **({"messages_artifact": artifact_path} if artifact_path else {}),
            },
        }
        if KEEP_FULL_MESSAGES:
            result["messages"] = messages
        return result

    def _run(self, data: dict, **kwargs) -> dict:
        """执行深度研究任务"""
        try:
            question = data['item']['question']
        except:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        start_time = time.time()
        answer = data['item'].get('answer', '')

        system_prompt = SYSTEM_PROMPT + today_date()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0

        while num_llm_calls_available > 0:
            # 超时检查
            if time.time() - start_time > 150 * 60:
                return self._build_result(question, answer, messages, "Timeout after 2h30m", "timeout")

            round_num += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages)

            print(f'Round {round_num}: {content[:500]}...' if len(content) > 500 else f'Round {round_num}: {content}')

            messages.append({"role": "assistant", "content": content.strip()})

            # 检查工具调用
            if 'Action:' in content and 'Action Input:' in content:
                try:
                    action_start = content.find("Action:") + len("Action:")
                    action_end = content.find("Action Input:")
                    tool_name = content[action_start:action_end].strip()

                    input_start = content.find("Action Input:") + len("Action Input:")
                    input_end = content.find("Observation:") if "Observation:" in content else len(content)
                    tool_input_str = content[input_start:input_end].strip()

                    try:
                        tool_args = json5.loads(tool_input_str)
                    except:
                        tool_args = {"query": tool_input_str}

                    result = self.custom_call_tool(tool_name, tool_args)
                    observation = f"Observation: {result}"
                    messages.append({"role": "user", "content": observation})

                except Exception as e:
                    messages.append({"role": "user", "content": f"Observation: Tool call error: {e}"})

            # 检查答案
            if '<answer>' in content and '</answer>' in content:
                prediction = content.split('<answer>')[1].split('</answer>')[0]
                return self._build_result(question, answer, messages, prediction, "answer")

            if num_llm_calls_available <= 0:
                messages[-1]['content'] = 'LLM call limit reached. Please provide your best answer.'

        prediction = "No answer found"
        if "<answer>" in messages[-1]["content"]:
            prediction = messages[-1]["content"].split("<answer>")[1].split("</answer>")[0]

        return self._build_result(question, answer, messages, prediction, "max_calls")
