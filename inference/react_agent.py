import json
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt import *
import time
import asyncio
import tiktoken

from tool_file import FileParser
from tool_scholar import Scholar
from tool_search import Search
from tool_visit import Visit

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 100))

USE_OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY", "").strip() and os.getenv("OPENROUTER_BASE_URL", "").strip())
ENABLE_PYTHON_INTERPRETER = os.getenv("ENABLE_PYTHON_INTERPRETER", "").strip().lower() in {"1", "true", "yes"}


def build_tool_class() -> List[BaseTool]:
    tools: List[BaseTool] = [
        FileParser(),
        Scholar(),
        Visit(),
        Search(),
    ]

    # OpenRouter mode should not eagerly depend on local sandbox/runtime tooling.
    if not USE_OPENROUTER or ENABLE_PYTHON_INTERPRETER:
        try:
            from tool_python import PythonInterpreter, SANDBOX_FUSION_IMPORT_ERROR
            if SANDBOX_FUSION_IMPORT_ERROR is None:
                tools.append(PythonInterpreter())
        except Exception:
            pass

    return tools


TOOL_CLASS = build_tool_class()
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


def build_research_config(generate_cfg: Dict) -> Dict:
    return {
        "min_rounds": int(generate_cfg.get("min_rounds", os.getenv("DEEP_RESEARCH_MIN_ROUNDS", 8))),
        "min_tool_calls": int(generate_cfg.get("min_tool_calls", os.getenv("DEEP_RESEARCH_MIN_TOOL_CALLS", 8))),
        "min_search_calls": int(generate_cfg.get("min_search_calls", os.getenv("DEEP_RESEARCH_MIN_SEARCH_CALLS", 3))),
        "min_visit_calls": int(generate_cfg.get("min_visit_calls", os.getenv("DEEP_RESEARCH_MIN_VISIT_CALLS", 3))),
        "min_scholar_calls": int(generate_cfg.get("min_scholar_calls", os.getenv("DEEP_RESEARCH_MIN_SCHOLAR_CALLS", 0))),
        "reflection_interval": int(generate_cfg.get("reflection_interval", os.getenv("DEEP_RESEARCH_REFLECTION_INTERVAL", 3))),
        "max_minutes": int(generate_cfg.get("max_minutes", os.getenv("DEEP_RESEARCH_MAX_MINUTES", 150))),
    }

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self._tokenizer = None
        self._encoding = None
        self.research_config = build_research_config(self.llm_generate_cfg)

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port, max_tries=10):
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        openrouter_api_base = os.getenv("OPENROUTER_BASE_URL", "").strip()
        use_openrouter = bool(openrouter_api_key and openrouter_api_base)

        openai_api_key = openrouter_api_key if use_openrouter else "EMPTY"
        openai_api_base = openrouter_api_base if use_openrouter else f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1 
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    max_tokens=10000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = chat_response.choices[0].message.content

                if use_openrouter:
                    reasoning = getattr(chat_response.choices[0].message, "reasoning", None)
                    if reasoning and reasoning.strip():
                        reasoning_content = "<think>\n" + reasoning.strip() + "\n</think>\n"
                        content = reasoning_content + (content or "")
                
                if content and content.strip():
                    print("--- Service call successful, received a valid response ---")
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return f"vllm server error!!!"

    def count_tokens(self, messages):
        if self.llm_local_path and os.path.exists(self.llm_local_path):
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
            full_prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = self._tokenizer(full_prompt, return_tensors="pt")
            return len(tokens["input_ids"][0])

        if self._encoding is None:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        serialized = json.dumps(messages, ensure_ascii=False)
        return len(self._encoding.encode(serialized))

    def build_research_state(self) -> Dict:
        return {
            "rounds": 0,
            "tool_calls": 0,
            "search_calls": 0,
            "visit_calls": 0,
            "scholar_calls": 0,
        }

    def update_research_state(self, state: Dict, tool_name: str):
        state["tool_calls"] += 1
        if tool_name == "search":
            state["search_calls"] += 1
        elif tool_name == "visit":
            state["visit_calls"] += 1
        elif tool_name == "google_scholar":
            state["scholar_calls"] += 1

    def research_requirements_met(self, state: Dict) -> Tuple[bool, List[str]]:
        gaps = []
        cfg = self.research_config
        if state["rounds"] < cfg["min_rounds"]:
            gaps.append(f"current round count is {state['rounds']} but minimum is {cfg['min_rounds']}")
        if state["tool_calls"] < cfg["min_tool_calls"]:
            gaps.append(f"current tool call count is {state['tool_calls']} but minimum is {cfg['min_tool_calls']}")
        if state["search_calls"] < cfg["min_search_calls"]:
            gaps.append(f"current search call count is {state['search_calls']} but minimum is {cfg['min_search_calls']}")
        if state["visit_calls"] < cfg["min_visit_calls"]:
            gaps.append(f"current visit call count is {state['visit_calls']} but minimum is {cfg['min_visit_calls']}")
        if state["scholar_calls"] < cfg["min_scholar_calls"]:
            gaps.append(f"current Google Scholar call count is {state['scholar_calls']} but minimum is {cfg['min_scholar_calls']}")
        return len(gaps) == 0, gaps

    def build_continue_research_message(self, question: str, state: Dict, gaps: List[str]) -> str:
        gap_text = "\n".join([f"- {gap}" for gap in gaps])
        return (
            "Your previous response attempted to conclude too early.\n"
            "Do not provide <answer> yet. Continue researching.\n\n"
            f"Original research question:\n{question}\n\n"
            "Research gaps that must be addressed before final synthesis:\n"
            f"{gap_text}\n\n"
            "Next step requirements:\n"
            "- Continue with additional search and visit tool calls.\n"
            "- Expand source coverage across missing subtopics, disagreements, comparisons, and recent evidence.\n"
            "- Prefer filling evidence gaps over rewriting a summary.\n"
            "- Only produce the final <answer> after the research constraints are satisfied."
        )

    def build_reflection_message(self, question: str, state: Dict) -> str:
        return (
            "Research checkpoint.\n"
            f"Question: {question}\n"
            f"Current progress: rounds={state['rounds']}, tool_calls={state['tool_calls']}, "
            f"search_calls={state['search_calls']}, visit_calls={state['visit_calls']}, scholar_calls={state['scholar_calls']}.\n"
            "Before the next tool call, think about:\n"
            "- What key subtopics still lack evidence?\n"
            "- Which claims need stronger primary sources?\n"
            "- What comparisons, failure modes, costs, risks, or implementation details are still missing?\n"
            "- Which targeted search or visit should be executed next to close the biggest gap?\n"
            "Do not finalize yet unless the research constraints are fully satisfied."
        )

    def build_force_answer_message(self) -> str:
        return (
            "You must now stop calling tools and produce the final report.\n"
            "Write a comprehensive report in this format:\n"
            "<think>final synthesis reasoning</think>\n"
            "<answer>\n"
            "## Executive Summary\n"
            "## Research Scope and Method\n"
            "## Key Findings\n"
            "## Detailed Analysis by Theme\n"
            "## Risks, Caveats, and Uncertainty\n"
            "## Conclusion\n"
            "## Sources\n"
            "</answer>"
        )

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        system_prompt = build_system_prompt(self.research_config, today_date())
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        research_state = self.build_research_state()
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > self.research_config["max_minutes"] * 60:
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "research_state": research_state,
                    "research_config": self.research_config,
                }
                return result
            round += 1
            research_state["rounds"] = round
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    if "python" in tool_call.lower():
                        try:
                            code_raw=content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            self.update_research_state(research_state, "PythonInterpreter")
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                        except:
                            result = "[Python Interpreter Error]: Formatting error."

                    else:
                        tool_call = json5.loads(tool_call)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        self.update_research_state(research_state, tool_name)
                        result = self.custom_call_tool(tool_name, tool_args)

                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                # print(result)
                messages.append({"role": "user", "content": result})
            if '<answer>' in content and '</answer>' in content:
                requirements_met, gaps = self.research_requirements_met(research_state)
                if requirements_met:
                    termination = 'answer'
                    break
                messages.append({"role": "user", "content": self.build_continue_research_message(question, research_state, gaps)})
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            if (
                self.research_config["reflection_interval"] > 0
                and research_state["rounds"] % self.research_config["reflection_interval"] == 0
                and '<answer>' not in content
                and num_llm_calls_available > 0
            ):
                messages.append({"role": "user", "content": self.build_reflection_message(question, research_state)})

            requirements_met, _ = self.research_requirements_met(research_state)
            if (
                requirements_met
                and '<answer>' not in content
                and num_llm_calls_available <= 1
            ):
                messages.append({"role": "user", "content": self.build_force_answer_message()})

            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = self.build_force_answer_message()
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "research_state": research_state,
                    "research_config": self.research_config,
                }
                return result

        requirements_met, _ = self.research_requirements_met(research_state)
        if '<answer>' not in messages[-1]['content'] and requirements_met:
            content = self.call_server(messages + [{"role": "user", "content": self.build_force_answer_message()}], planning_port)
            messages.append({"role": "user", "content": self.build_force_answer_message()})
            messages.append({"role": "assistant", "content": content.strip()})

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "research_state": research_state,
            "research_config": self.research_config,
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
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
