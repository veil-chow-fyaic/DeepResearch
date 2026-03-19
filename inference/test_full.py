"""
完整功能测试 - 验证 DeepResearch 工具链
"""
import os
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('../.env')

print("="*60)
print("DeepResearch 功能测试")
print("="*60)

# 1. 测试环境变量
print("\n[1] 环境变量检查")
required_keys = ['OPENROUTER_API_KEY', 'SERPER_KEY_ID', 'JINA_API_KEYS']
for key in required_keys:
    val = os.getenv(key, '')
    status = "✓" if val and val != 'your_key' else "✗"
    print(f"  {status} {key}: {val[:15]}..." if val else f"  {status} {key}: 未设置")

# 2. 测试工具导入
print("\n[2] 工具导入测试")
try:
    from tool_search import Search
    print("  ✓ Search 工具")
except Exception as e:
    print(f"  ✗ Search 工具: {e}")

try:
    from tool_visit import Visit
    print("  ✓ Visit 工具")
except Exception as e:
    print(f"  ✗ Visit 工具: {e}")

try:
    from tool_scholar import Scholar
    print("  ✓ Scholar 工具")
except Exception as e:
    print(f"  ✗ Scholar 工具: {e}")

# 3. 测试文件解析（USE_IDP=False）
print("\n[3] 文件解析测试 (USE_IDP=False)")
os.environ['USE_IDP'] = 'False'
try:
    from file_tools.file_parser import SingleFileParser
    print("  ✓ 文件解析器导入成功")
except Exception as e:
    print(f"  ✗ 文件解析器: {e}")

# 4. 测试 OpenRouter API
print("\n[4] OpenRouter API 测试")
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    response = client.chat.completions.create(
        model="alibaba/tongyi-deepresearch-30b-a3b",
        messages=[{"role": "user", "content": "Say 'Hello' in one word"}],
        max_tokens=10
    )
    print(f"  ✓ API 响应: {response.choices[0].message.content}")
except Exception as e:
    print(f"  ✗ API 错误: {e}")

# 5. 测试搜索工具
print("\n[5] 搜索工具测试")
try:
    search = Search()
    result = search.call({"query": ["Python programming language"]})
    print(f"  ✓ 搜索成功，返回 {len(result)} 字符")
except Exception as e:
    print(f"  ✗ 搜索失败: {e}")

print("\n" + "="*60)
print("测试完成!")
print("="*60)
