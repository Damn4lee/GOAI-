# GOAI 科研助手

面向材料/学术文献调研的 AI Agent：检索真实文献 → 提取证据 → 可证伪性审查 → 输出结构化结论。

## 快速开始

1. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

2. 配置密钥（**不要把真实密钥提交到仓库**）

   ```bash
   cp .env.example .env
   ```

   然后在 `.env` 中填入 `SCIVERSE_TOKEN` 与 `DEEPSEEK_API_KEY`。
   `config.py` 会自动加载 `.env`（python-dotenv）并读取这些值；未设置时会给出明确报错。

3. 运行

   ```bash
   # 阶段化搜索 Agent（把研究问题作为参数传入）
   python research_agent.py "你的研究问题"

   # LangGraph 状态图流程
   python research_state_graph.py
   ```

   也可以打开 `科研助手三.ipynb` 按 Cell 顺序运行。

## 项目结构

| 文件 | 作用 |
|---|---|
| `config.py` | 配置（密钥从环境变量读取）与阶段 Prompt |
| `common.py` | 工具注册装饰器、JSON 解析 |
| `literature_searcher.py` | Sciverse 多端点检索客户端与检索工具 |
| `keyword_generator.py` | 关键词/实体提取 |
| `gap_analyzer.py` | 可证伪性检查、DOI 验证、断言提取、机制验证 |
| `research_agent.py` | 6 阶段状态机 Agent 主循环 |
| `research_state_graph.py` | LangGraph 状态图编排 |

## 安全说明

- 历史版本曾把真实 API 密钥提交到公开仓库，**已轮换的密钥请勿再次提交**；
- 密钥一律走环境变量（`.env`），运行时产物（缓存/日志/状态）已在 `.gitignore` 中排除。

## 待办（已知问题）

- `paper_relations` 端点参数已修正为 `relation`，但建议实测一次引用追踪；
- 模型名默认 `deepseek-chat`（DeepSeek 官方模型），若换供应商需同步处理 `extra_body` 等专有参数；
- 尚未建立评测集，改版质量无法量化对比。
