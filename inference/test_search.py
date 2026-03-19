"""测试搜索功能"""
import os
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('../.env')

from tool_search import Search
from tool_visit import Visit

print("=== 测试搜索工具 ===")
search = Search()
result = search.call({"query": ["OpenAI GPT-4 latest news 2025"]})
print(result[:2000])
