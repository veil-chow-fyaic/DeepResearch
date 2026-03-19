# DeepResearch 项目架构文档

---

## 一、项目结构

```
DeepResearch/
├── inference/                    # 核心推理代码
│   ├── react_agent.py           # 原始 Agent (需要本地 vLLM)
│   ├── run_multi_react.py       # 批量运行脚本
│   ├── run_react_infer.sh       # 原始启动脚本
│   ├── prompt.py                # 系统提示词
│   ├── tool_search.py           # 搜索工具
│   ├── tool_visit.py            # 网页访问工具
│   ├── tool_scholar.py          # 学术搜索工具
│   ├── tool_python.py           # Python 解释器工具
│   ├── tool_file.py             # 文件解析工具
│   ├── file_tools/              # 文件解析实现
│   │   ├── file_parser.py       # 主解析器
│   │   ├── idp.py               # 阿里云 IDP 接口
│   │   └── video_agent.py       # 视频解析
│   └── eval_data/               # 测试数据
│       ├── example.jsonl
│       └── file_corpus/
├── Agent/                       # Agent 研究论文
├── WebAgent/                    # Web Agent 实现
├── evaluation/                  # 评估脚本
├── docs/                        # 文档
├── .env                         # 环境配置
├── .env.example                 # 环境配置模板
└── requirements.txt             # 依赖
```

---

## 二、架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户问题                                │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   DeepResearch Agent                         │
│              (alibaba/tongyi-deepresearch-30b-a3b)          │
│                      via OpenRouter                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    Search     │ │    Visit      │ │   Scholar     │
│   (Serper)    │ │ (Jina + LLM)  │ │   (Serper)    │
└───────────────┘ └───────┬───────┘ └───────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  OpenAI 兼容   │
                  │   API (总结)   │
                  └───────────────┘
```

---

## 三、环境配置 (.env)

### 3.1 官方要求的 API Keys

根据官方 README 和 .env.example，需要以下配置：

| 变量名 | 必需性 | 用途 | 获取地址 |
|--------|--------|------|----------|
| `SERPER_KEY_ID` | **必需** | Google 搜索和学术搜索 | https://serper.dev/ |
| `JINA_API_KEYS` | **必需** | 网页内容读取 | https://jina.ai/ |
| `API_KEY` | **必需** | 网页内容总结 | 任何 OpenAI 兼容 API |
| `API_BASE` | **必需** | 总结模型 API 地址 | - |
| `SUMMARY_MODEL_NAME` | **必需** | 总结模型名称 | - |
| `MODEL_PATH` | 或OpenRouter | 本地模型路径 | - |
| `DASHSCOPE_API_KEY` | 可选 | 阿里云文件解析 | https://dashscope.aliyun.com/ |
| `SANDBOX_FUSION_ENDPOINT` | 可选 | Python 执行沙箱 | https://github.com/bytedance/SandboxFusion |
| `DATASET` | 必需 | 评估数据集路径 | - |
| `OUTPUT_PATH` | 必需 | 输出目录 | - |

### 3.2 完整 .env 示例

```bash
# =============================================================================
# 主模型 - 两种方案二选一
# =============================================================================

# 方案一: 本地 vLLM (需要 GPU)
MODEL_PATH=/path/to/model
DATASET=eval_data/example.jsonl
OUTPUT_PATH=./outputs

# 方案二: OpenRouter API (无需 GPU) - 我们采用的方案
OPENROUTER_API_KEY=sk-or-v1-xxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL=alibaba/tongyi-deepresearch-30b-a3b

# =============================================================================
# 搜索和网页读取 (必需)
# =============================================================================
# Serper API for web search and Google Scholar
SERPER_KEY_ID=your_serper_key

# Jina API for web page reading
JINA_API_KEYS=your_jina_key

# =============================================================================
# 网页总结 - OpenAI-compatible API (必需)
# 可以用 OpenAI、阶跃星辰、或其他兼容 API
# =============================================================================
API_KEY=your_api_key
API_BASE=https://api.openai.com/v1  # 或其他兼容 API
SUMMARY_MODEL_NAME=gpt-4o-mini      # 或其他模型

# =============================================================================
# 文件解析 (可选)
# =============================================================================
# 使用阿里云 IDP 高级文件解析
USE_IDP=False
DASHSCOPE_API_KEY=your_dashscope_key

# =============================================================================
# Python 代码执行沙箱 (可选)
# =============================================================================
SANDBOX_FUSION_ENDPOINT=http://localhost:8080

# =============================================================================
# 运行配置
# =============================================================================
TEMPERATURE=0.6
MAX_WORKERS=1
ROLLOUT_COUNT=1
```

---

## 四、API Keys 详解

### 4.1 SERPER_KEY_ID (搜索) - 必需

**官方说明**: "Get your key from Serper.dev for web search and Google Scholar"

**用途**:
- 执行 Google 搜索
- 执行 Google Scholar 学术搜索

**获取方式**:
1. 访问 https://serper.dev/
2. 注册账号
3. 在 API Key 页面获取

**费用**: 免费额度 2500 次/月

---

### 4.2 JINA_API_KEYS (网页读取) - 必需

**官方说明**: "Get your key from Jina.ai for web page reading"

**用途**:
- 读取网页原始内容
- 返回干净的文本（去除广告、导航等）

**获取方式**:
1. 访问 https://jina.ai/reader
2. 注册获取 API Key

**费用**: 免费额度充足

---

### 4.3 API_KEY / API_BASE / SUMMARY_MODEL_NAME (网页总结) - 必需

**官方说明**: "OpenAI-compatible API for page summarization"

**用途**:
- 总结网页内容
- 提取关键信息

**可用选项**:
| 提供商 | API_BASE | 模型示例 |
|--------|----------|----------|
| OpenAI | https://api.openai.com/v1 | gpt-4o-mini |
| 阶跃星辰 | https://api.stepfun.com/v1 | step-3.5-flash |
| 智谱AI | https://open.bigmodel.cn/api/paas/v4 | glm-4-flash |
| 阿里云 | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-turbo |

**获取方式**: 各平台官网注册

---

### 4.4 MODEL_PATH 或 OpenRouter (主模型) - 必需

**官方说明**: "Path to your model weights" 或使用 OpenRouter

**两种方案**:

| 方案 | 要求 | 说明 |
|------|------|------|
| 本地 vLLM | 8x GPU | 原始方案，性能最佳 |
| OpenRouter | API Key | 无需 GPU，官方支持 |

**OpenRouter 配置**:
```bash
OPENROUTER_API_KEY=sk-or-v1-xxx
MODEL=alibaba/tongyi-deepresearch-30b-a3b
```

**获取方式**:
1. 访问 https://openrouter.ai/
2. 注册账号
3. 在 Keys 页面创建

---

### 4.5 DASHSCOPE_API_KEY (文件解析) - 可选

**官方说明**: "Get your key from Dashscope for file parsing"

**用途**:
- 高级 PDF 解析（保留布局、表格）
- Word/Excel/PPT 解析
- 图片 OCR
- 视频/音频转录

**替代方案**: 设置 `USE_IDP=False`，使用本地解析器

**本地解析器支持**:
- PDF (pdfplumber)
- Word (python-docx)
- PPT (python-pptx)
- Excel (openpyxl)
- CSV/TXT/HTML

---

### 4.6 SANDBOX_FUSION_ENDPOINT (Python 沙箱) - 可选

**官方说明**: "Python interpreter sandbox endpoints"

**用途**:
- 安全执行 Python 代码
- 数据分析和计算

**限制**:
- 需要部署 SandboxFusion 服务
- Python 3.14 不兼容

---

## 五、启动方法

### 5.1 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/Alibaba-NLP/DeepResearch.git
cd DeepResearch

# 2. 创建虚拟环境 (官方推荐 Python 3.10.0)
conda create -n deepresearch python=3.10.0
conda activate deepresearch

# 3. 安装依赖
pip install -r requirements.txt
```

### 5.2 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Keys
vim .env
```

### 5.3 运行

**官方方式 (本地 vLLM)**:
```bash
# 1. 启动 vLLM 服务 (需要 GPU)
bash inference/run_react_infer.sh
```

**OpenRouter 方式 (无需 GPU)**:
```bash
# 使用我们创建的简化脚本
cd inference
python react_agent_simple.py
```

---

## 六、数据格式

### 6.1 JSONL 格式 (推荐)

```json
{"question": "What is the capital of France?", "answer": "Paris"}
{"question": "Explain quantum computing", "answer": ""}
```

### 6.2 JSON 格式

```json
[
  {"question": "What is the capital of France?", "answer": "Paris"},
  {"question": "Explain quantum computing", "answer": ""}
]
```

### 6.3 文件引用

如果问题涉及文件解析：
```
project_root/
├── eval_data/
│   ├── my_questions.jsonl
│   └── file_corpus/
│       ├── report.pdf
│       └── data.xlsx
```

问题格式：
```json
{"question": "(Uploaded 1 file: ['report.pdf'])\n\nWhat are the key findings?", "answer": "..."}
```

---

## 七、常见问题

### Q1: Mac 没有 GPU 怎么办？

**A**: 使用 OpenRouter API 方案，无需本地 GPU。

### Q2: Python 3.14 兼容性问题？

**A**: `sandbox_fusion` 与 Python 3.14 不兼容。解决方案：
- 使用 Python 3.10.0 (官方推荐)
- 或暂时禁用 Python 解释器工具

### Q3: 如何选择网页总结模型？

**A**: 任何 OpenAI 兼容的便宜模型都可以：
- OpenAI gpt-4o-mini
- 阶跃星辰 step-3.5-flash
- 智谱 glm-4-flash
- 阿里 qwen-turbo

---

## 八、参考资源

- **官方 GitHub**: https://github.com/Alibaba-NLP/DeepResearch
- **HuggingFace 模型**: https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B
- **OpenRouter**: https://openrouter.ai/alibaba/tongyi-deepresearch-30b-a3b
- **技术论文**: https://arxiv.org/pdf/2510.24701
- **技术博客**: https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
