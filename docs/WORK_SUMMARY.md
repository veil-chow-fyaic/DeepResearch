# DeepResearch 部署工作总结

**部署日期**: 2026-03-19
**部署环境**: Mac M4, Python 3.14
**项目来源**: https://github.com/veil-chow-fyaic/DeepResearch.git

---

## 一、项目概述

**Tongyi DeepResearch** 是阿里通义实验室开源的深度研究 AI Agent 模型：
- **模型**: Tongyi-DeepResearch-30B-A3B (30B 参数, 3B 激活)
- **特点**: 专为长周期深度信息搜索设计
- **能力**: 网页搜索、学术搜索、网页内容提取、文件解析

---

## 二、部署方案

由于 Mac 没有 NVIDIA GPU，无法运行官方的本地 vLLM 方案，因此采用 **OpenRouter API** 方案。

### 2.1 架构对比

| 组件 | 官方方案 | 我们的方案 |
|------|----------|------------|
| 主模型 | 本地 vLLM (8x GPU) | OpenRouter API |
| 网页总结 | OpenAI API | 阶跃星辰 API |
| 文件解析 | 阿里云 IDP | 本地解析器 |
| Python 沙箱 | SandboxFusion | 暂不可用 (Python 3.14 不兼容) |

### 2.2 依赖服务

| 服务 | 用途 | 费用 |
|------|------|------|
| OpenRouter | 调用 DeepResearch 模型 | ~$0.6/1M tokens |
| 阶跃星辰 | 网页内容总结 | 免费额度 |
| Serper | Google 搜索 | 免费额度 |
| Jina | 网页内容读取 | 免费额度 |

---

## 三、新增文件清单

所有新增文件均为独立文件，**未修改任何原始代码**：

```
~/projects/DeepResearch/
├── .env                              # 环境配置
├── requirements_mac.txt              # Mac 兼容依赖
├── run_mac.sh                        # 启动脚本
├── docs/
│   ├── WORK_SUMMARY.md               # 本文档
│   └── ARCHITECTURE.md               # 架构文档
└── inference/
    ├── react_agent_simple.py         # 简化版 Agent (支持 OpenRouter)
    ├── test_agent.py                 # 测试脚本
    └── test_full.py                  # 功能测试
```

---

## 四、测试结果

| 功能 | 状态 | 说明 |
|------|------|------|
| OpenRouter API | ✅ 通过 | DeepResearch 模型响应正常 |
| Serper 搜索 | ✅ 通过 | 返回搜索结果 |
| Jina 网页读取 | ✅ 通过 | 返回网页内容 |
| 阶跃星辰 API | ✅ 通过 | 网页总结正常 |
| ReAct 工具调用 | ✅ 通过 | 模型正确使用 Action/Action Input |
| 文件解析 (本地) | ✅ 通过 | USE_IDP=False 时可用 |
| Python 沙箱 | ❌ 不可用 | sandbox_fusion 与 Python 3.14 不兼容 |

---

## 五、已知问题

### 5.1 Python 3.14 兼容性
- **问题**: `sandbox_fusion` 依赖 `pydantic v1`，与 Python 3.14 不兼容
- **影响**: Python 解释器工具不可用
- **解决**: 暂时禁用，或降级到 Python 3.11

### 5.2 JSON 格式解析
- **问题**: 模型有时生成带 trailing comma 的 JSON
- **影响**: 工具调用可能解析失败
- **解决**: 使用 `json5` 库解析

### 5.3 OpenRouter 速率限制
- **问题**: 上游提供商可能有速率限制
- **影响**: 高频调用时可能返回 429 错误
- **解决**: 添加重试逻辑，或在 OpenRouter 添加自己的 API Key

---

## 六、后续建议

1. **稳定测试** - 运行更多问题验证稳定性
2. **集成到 OpenClaw** - 封装为可调用的服务
3. **GPU 部署** - 如需更高性能，可部署 vLLM 服务
4. **Python 3.11** - 如需 Python 沙箱，可降级 Python 版本

---

## 七、参考资源

- [GitHub 仓库](https://github.com/Alibaba-NLP/DeepResearch)
- [HuggingFace 模型](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
- [OpenRouter API](https://openrouter.ai/)
- [技术论文](https://arxiv.org/pdf/2510.24701)
