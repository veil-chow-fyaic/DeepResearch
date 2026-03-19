# DeepResearch 项目部署 TODO List

> 创建时间: 2026-03-18
> 项目路径: `~/projects/DeepResearch`
> 部署状态: ✅ 基础部署完成

---

## 项目概述

**DeepResearch** 是阿里通义实验室的深度研究 Agent 模型，30B 参数，专为长周期深度信息搜索任务设计。

### 项目结构
```
DeepResearch/
├── inference/                 # 推理代码
│   ├── react_agent.py         # 原始 vLLM 版本 (需要 GPU)
│   ├── react_agent_openrouter.py  # Mac 适配版 (OpenRouter API)
│   ├── run_openrouter.py      # Mac 运行脚本
│   ├── tool_*.py              # 工具模块
│   └── eval_data/             # 测试数据
├── Agent/                     # Agent 相关研究
├── WebAgent/                  # Web Agent 实现
├── evaluation/                # 评估脚本
├── .env                       # 环境配置
├── requirements_mac.txt       # Mac 依赖
├── run_mac.sh                 # Mac 启动脚本
└── venv/                      # Python 虚拟环境
```

---

## ✅ 已完成任务

### 1. 项目克隆
- [x] 克隆仓库到 `~/projects/DeepResearch`
- [x] 确认文件完整性

### 2. 环境配置
- [x] 创建 Python 虚拟环境 (venv)
- [x] 安装 Mac 兼容依赖 (`requirements_mac.txt`)
- [x] 安装额外依赖 (pandas, scipy, soundfile)

### 3. Mac 适配
- [x] 创建 `react_agent_openrouter.py` (OpenRouter API 版本)
- [x] 创建 `run_openrouter.py` (Mac 运行脚本)
- [x] 创建 `run_mac.sh` (启动脚本)
- [x] 更新 `.env` 配置文件

---

## 🔲 待完成任务

### 优先级 P0 - 立即执行

#### 1. 配置 API Keys
- [ ] 获取 OpenRouter API Key: https://openrouter.ai/
- [ ] 更新 `.env` 文件中的 `OPENROUTER_API_KEY`
- [ ] (可选) 配置 Serper API 用于搜索
- [ ] (可选) 配置 Jina API 用于网页读取

#### 2. 首次运行测试
```bash
cd ~/projects/DeepResearch
source venv/bin/activate

# 设置 API Key (临时)
export OPENROUTER_API_KEY="sk-or-..."

# 运行测试
bash run_mac.sh
```

### 优先级 P1 - 功能验证

#### 3. 验证工具链
- [ ] 测试 Search 工具 (需要 Serper API)
- [ ] 测试 Visit 工具 (需要 Jina API)
- [ ] 测试文件解析工具
- [ ] 测试 Python 解释器 (需要 SandboxFusion)

#### 4. 自定义测试数据
- [ ] 创建 `eval_data/my_questions.jsonl`
- [ ] 运行自定义问题测试

### 优先级 P2 - 深度集成

#### 5. 与 OpenClaw 集成 (可选)
- [ ] 评估是否可作为 OpenClaw 的深度研究后端
- [ ] 考虑 API 成本 vs 自建 vLLM 服务

#### 6. 服务器部署 (可选)
- [ ] 准备 GPU 服务器 (8x A100/H100)
- [ ] 下载模型权重 (30B-A3B)
- [ ] 部署 vLLM 服务
- [ ] 运行原始 `run_react_infer.sh`

---

## 运行指南

### Mac 快速启动
```bash
# 1. 进入项目目录
cd ~/projects/DeepResearch

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 设置 API Key
export OPENROUTER_API_KEY="your_key_here"

# 4. 运行
bash run_mac.sh
```

### 手动运行
```bash
cd ~/projects/DeepResearch/inference
source ../venv/bin/activate

python run_openrouter.py \
    --model "alibaba/tongyi-deepresearch-30b-a3b" \
    --dataset "eval_data/example.jsonl" \
    --output "../outputs" \
    --max_items 1
```

---

## 技术说明

### Mac 适配策略
由于 Mac 没有 NVIDIA GPU，无法运行 vLLM。采用以下方案：
1. 使用 OpenRouter API 调用云端模型
2. 保留所有工具链功能
3. 单线程运行 (避免 API 限流)

### API 成本估算
- OpenRouter: ~$0.60/1M tokens (输入) + ~$2.40/1M tokens (输出)
- 预计每个深度研究任务: $0.10 - $1.00

### 已知限制
1. Python 解释器工具需要 SandboxFusion 服务
2. 视频分析需要 Dashscope API
3. 部分工具可能需要额外配置

---

## 参考资源

- [GitHub 仓库](https://github.com/Alibaba-NLP/DeepResearch)
- [HuggingFace 模型](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
- [OpenRouter API](https://openrouter.ai/alibaba/tongyi-deepresearch-30b-a3b)
- [技术博客](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)
- [论文](https://arxiv.org/pdf/2510.24701)
