# -*- coding: utf-8 -*-

import json
import os
from openai import OpenAI


def _env(name: str, default: str = "") -> str:
    """从环境变量读取配置，未设置时返回默认值。"""
    return os.environ.get(name, "").strip() or default


CONFIG = {
    # ── Sciverse 学术搜索引擎 API Token ──
    # 用途：认证 Sciverse meta-search API，用于文献检索
    # 已迁移到环境变量读取（SCIVERSE_TOKEN）
    # 如果要换：到 https://sciverse.space 申请新的 Token，粘贴替换此值
    # 修改此值时需要同步检查：SCIVERSE_DATA_DIR（数据目录）、sciverse_client 实例化
    "sciverse_token": _env("SCIVERSE_TOKEN"),

    # ── DeepSeek LLM API Key ──
    # 用途：调用大语言模型（用于 Agent 推理、GA 评估、可证伪性检查等）
    # 当前值：
    # 如果要换：到 https://platform.deepseek.com/api_keys 生成新 Key，粘贴替换
    # 如果换提供商（如 OpenAI、智谱）：同时修改 llm_base_url 和 llm_model
    # 修改此值时需要同步检查：llm_base_url、llm_model
    # 由于github政策问题，需测试员自己寻找api并填写上。
    "deepseek_key": _env("DEEPSEEK_API_KEY"),

    # ── LLM API 基础 URL ──
    # 用途：指定 LLM API 的端点地址
    # 当前值：https://api.deepseek.com/v1 （DeepSeek 官方端点）
    # 如果换提供商：
    #   - OpenAI:     "https://api.openai.com/v1"
    #   - 智谱 GLM:   "https://open.bigmodel.cn/api/paas/v4"
    #   - 本地 Ollama: "http://localhost:11434/v1"
    #   - 硅基流动:    "https://api.siliconflow.cn/v1"
    # 修改此值时需要同步检查：deepseek_key、llm_model
    "llm_base_url": _env("LLM_BASE_URL", "https://api.deepseek.com/v1"),

    # ── LLM 模型名称 ──
    # 用途：指定调用的具体模型，影响推理质量和速度
    # 当前值：deepseek-v4-pro（关闭思考模式，适配结构化JSON和工具调用）
    # 如果要换模型：
    #   - deepseek-v4-pro     （DeepSeek V4 高性能模型）
    #   - deepseek-v4-flash   （DeepSeek V4 快速模型）
    #   - gpt-4o              （OpenAI 模型）
    #   - glm-4-flash         （智谱模型）
    #   - qwen/qwen2.5-72b-instruct （硅基流动上的通义千问）
    # ⚠️ 注意：本项目的结构化解析和工具调用统一关闭思考模式
    # 修改此值时需要同步检查：deepseek_key、llm_base_url
    "llm_model": _env("LLM_MODEL", "deepseek-chat"),

    # ── 搜索阶段轮次配置 [v3新增] ──
# 双循环阶段化搜索：每个阶段的最大轮次
# 总计 12 轮（3+2+1+3+2+1）
"search_phases": {
    "diverge_1": 3,      # 发散搜索1（强检索+弱检索+关联发散）
    "converge_1": 2,     # 收敛搜索1（聚焦假设+补充证据）
    "falsify": 1,        # 证伪检查（判断方向可行性）
    "diverge_2": 3,      # 拓展发散（变体搜索+作者追踪+跨领域类比）
    "converge_2": 2,     # 收敛整合（整合发现+构建证据链）
    "final_verify": 1,   # 最终验证（输出JSON结果）
},
}


state = {
"phase": "research", # "research" = 文献调研阶段, "discovery" = 任务A（GA材料搜索）阶段
"context": {
"gaps": [], # Phase 1 发现的 Research Gap 列表，Phase 2 会读取
"cached_literature": [], # 核心文献缓存
"target_property": "" # 用户指定的性能指标
},
"api_stats": {
"total_calls": 0, # Sciverse API 总调用次数（不含缓存命中）
"cache_hits": 0, # 缓存命中次数
"errors": 0 # [迁移改动] API 调用失败次数
}
}


SCIVERSE_DATA_DIR = r"D:\AI科研助手专用目录"
try:
    os.makedirs(SCIVERSE_DATA_DIR, exist_ok=True)
except OSError as e:
    # D盘不存在或无权限时，回退到脚本所在目录
    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AI科研助手专用目录")
    print(f"⚠️ 无法创建 {SCIVERSE_DATA_DIR}（{e}），已回退至: {fallback}")
    SCIVERSE_DATA_DIR = fallback
    os.makedirs(SCIVERSE_DATA_DIR, exist_ok=True)

CACHE_FILE_PATH = os.path.join(SCIVERSE_DATA_DIR, "cache.json")
LOG_FILE_PATH = os.path.join(SCIVERSE_DATA_DIR, "log.txt")

# [迁移改动] 不再单独定义 SCIVERSE_API_TOKEN 变量，直接从 CONFIG 读取

# 原版: SCIVERSE_API_TOKEN = "sci_xxx".strip()

SCIVERSE_API_TOKEN = CONFIG["sciverse_token"]  # [迁移改动] 从 CONFIG 集中读取


if "CONFIG" not in globals():
    raise RuntimeError("请先运行 Cell 0，创建 CONFIG 配置。")

required_config_keys = [
    "deepseek_key",
    "llm_base_url",
    "llm_model",
]

missing_config_keys = [
    key for key in required_config_keys
    if not CONFIG.get(key)
]

if missing_config_keys:
    raise RuntimeError(
        f"CONFIG 缺少配置项：{missing_config_keys}"
    )


# ============================================================
# 2. 加载 System Prompt
# ============================================================

try:
    with open(
        "materials_agent_prompt.txt",
        "r",
        encoding="utf-8",
    ) as prompt_file:
        SYSTEM_PROMPT = prompt_file.read()

    print(
        f"✅ 已加载 System Prompt："
        f"{len(SYSTEM_PROMPT)} 个字符"
    )

except FileNotFoundError:
    SYSTEM_PROMPT = (
        "你是一个材料科学研究助手。"
        "你的任务是基于学术文献进行分析，"
        "识别 Research Gap，构建证据链，"
        "并输出结构化 JSON 结果。"
        "请始终使用搜索工具获取文献证据，"
        "不要编造不存在的文献、DOI 或实验数据。"
    )

    print(
        "⚠️ 没有找到 materials_agent_prompt.txt，"
        "暂时使用内置 System Prompt。"
    )


# ============================================================
# 3. 创建 DeepSeek 客户端
# ============================================================

llm_client = OpenAI(
    api_key=CONFIG["deepseek_key"].strip(),
    base_url=CONFIG["llm_base_url"],
)

# 保留旧代码可能使用的 model 变量
model = CONFIG["llm_model"]

print("✅ DeepSeek 客户端创建完成")


PHASE_PROMPTS = {
    "diverge_1": (
       "【阶段1/6：发散搜索】你的任务是广泛搜索相关文献，建立领域全景。\n"
        "本阶段采用'摘要优先'策略：所有搜索工具默认返回 title + abstract + DOI + year + venue。\n"
       "不要获取全文，先低成本获取大量题录+摘要进行筛选。\n"
        "请按以下顺序执行搜索：\n"
        "  1. 使用 search_literature 进行宽泛关键词搜索（page_size=10），覆盖主要方向\n"
        "  2. 使用 search_weak 搜索近义词、上位概念、跨领域关键词，补充盲区\n"
        "  3. 使用 search_related 从轮次1-2的结果中提取关键实体（论文名、作者名、算法名），继续追踪相关工作\n"
        "目标：建立对该研究领域的全面认识，识别潜在 Research Gap，不要过早聚焦。"
    ),
    "converge_1": (
        "【阶段2/6：收敛搜索】基于上一阶段的摘要和题录，聚焦最有可能的2-3个假设。\n"
        "继续使用 search_literature / search_weak / search_related 收集支持/反驳证据，\n"
        "重点关注：哪些假设有摘要级证据支持？证据缺口在哪里？\n"
        "本阶段仍不要获取全文，只通过题录和摘要进行筛选。"
    ),
    "falsify": (
        "【阶段3/6：证伪检查】现在请对最有证据支持的假设调用 falsification_check。\n"
        "将你的假设和目前收集到的证据摘要传入，让审稿人视角来检查漏洞。\n"
        "如果某篇文献对证伪至关重要且 abstract 明显不够用，可以调用 fetch_paper_content(level='abstract') 再确认一下。"
    ),
    "diverge_2": (
        "【阶段4/6：拓展发散】证伪检查通过后，继续围绕已确认的方向进行拓展搜索：\n"
        "  1. 使用 search_related 搜索相关算法/方法的变体（relation_type='similar'）\n"
        "  2. 使用 search_related 搜索关键作者的其他工作（relation_type='author'）\n"
        "  3. 使用 search_weak 搜索其他领域的类似问题解法，寻找跨领域灵感\n"
        "本阶段仍以摘要和题录为主，仅对极少数核心文献使用 fetch_paper_content。"
    ),
    "converge_2": (
        "【阶段5/6：收敛整合】整合所有搜索发现，构建完整的证据链。\n"
        "使用 search_literature 和 search_related 针对最终假设补充最后的证据。\n"
        "重点：搜索证据链中缺失的环节。\n"
        "对于证据链上的关键文献（最多2-3篇），可以调用 fetch_paper_content(level='abstract' 或 'fulltext') 获取更完整内容。"
    ),
    "final_verify": (
        "【阶段6/6：最终验证】所有搜索已完成。请基于全部搜索结果，输出最终的结构化 JSON 结果。\n"
        "注意：你当前能访问的文献数据包括标题、摘要、DOI、年份、期刊。\n"
        "如果某些文献没有拿到摘要或全文，请在输出中明确标注，不要编造。\n"
        "请严格使用以下 JSON 格式，不要包含任何解释文字或 markdown 代码块标记：\n"
        "{\n"
        "  \"research_gaps\": [\"gap1\", \"gap2\", ...],  // Research Gap 列表，每个条目为字符串\n"
        "  \"key_findings\": [\"发现1\", \"发现2\", ...],  // 关键文献发现\n"
        "  \"evidence_chain\": [\"证据1（含文献来源/DOI，注明是否来自摘要/全文）\", \"证据2\", ...],\n"
        "  \"hypotheses\": [\"假设1\", \"假设2\", ...],  // 基于证据的候选构效关系假设\n"
        "  \"future_directions\": [\"方向1\", \"方向2\"],  // 未来研究方向\n"
        "  \"confidence\": \"high|medium|low\",  // 对结论整体置信度\n"
        "  \"data_limitations\": \"说明哪些证据来自摘要，哪些来自全文，哪些缺失\"\n"
        "}\n"
        "注意：\n"
        "  - research_gaps 必须是字符串数组，这是下游 GA 发现阶段的输入。\n"
        "  - 每条 gap 请写明具体缺失的知识连接或待验证假设。\n"
        "  - 不要再调用搜索工具，直接输出最终结果。"
    ),
}


PHASE_ORDER = [
    "diverge_1",     # 阶段1：发散搜索（外循环起点）
    "converge_1",    # 阶段2：收敛搜索（内循环起点）
    "falsify",       # 阶段3：证伪检查（关键检查点）
    "diverge_2",     # 阶段4：拓展发散（外循环第二轮）
    "converge_2",    # 阶段5：收敛整合（内循环第二轮）
    "final_verify",  # 阶段6：最终验证（出口）
]
