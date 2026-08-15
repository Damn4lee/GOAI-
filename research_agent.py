# -*- coding: utf-8 -*-

import json
from openai import OpenAI

import gap_analyzer
import keyword_generator
import literature_searcher
from common import TOOL_REGISTRY, _safe_parse_llm_json, get_tools_list
from config import CONFIG, PHASE_ORDER, PHASE_PROMPTS, SYSTEM_PROMPT, llm_client, state
from literature_searcher import sciverse_client

TOOLS = get_tools_list()


def run_agent(user_message: str, max_tool_rounds: int = None):
    """
    [v3重构] 双循环阶段化搜索 Agent

    核心改进：将原来的"自由循环"改为"阶段状态机"。
    - 6个阶段，代码控制阶段转换（不是 LLM 自己决定）
    - 每个阶段有最大轮次限制（从 CONFIG["search_phases"] 读取）
    - 每个阶段开始时注入对应的 system prompt
    - 保留原有所有防护机制（查询去重、异常捕获、JSON解析重试等）

    参数:
    user_message: 用户的研究问题
    max_tool_rounds: 全局最大轮次（可选，默认从 CONFIG 自动计算）

    返回:
    解析后的 JSON 结果字典
    """
    # [v3新增] 自动计算全局最大轮次
    # 如果调用者没有指定 max_tool_rounds，则从 CONFIG 的 search_phases 求和
    if max_tool_rounds is None:
        phases_config = CONFIG.get("search_phases", {})
        max_tool_rounds = sum(phases_config.values())
        # 加 2 轮余量，用于 JSON 解析重试等意外情况
        max_tool_rounds += 2

    # ========== 初始化 messages ==========
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # ========== 防护机制初始化（保留原版所有防护） ==========
    executed_queries: set = set()          # 防护1: 查询去重
    consecutive_empty = 0                   # 防护5: 连续空内容计数
    MAX_CONSECUTIVE_EMPTY = 2              # 防护5: 连续空内容上限

    # ========== [v3新增] 阶段状态机变量 ==========
    current_phase_idx = 0                   # 当前阶段在 PHASE_ORDER 中的索引
    current_phase_name = PHASE_ORDER[0]     # 当前阶段名称
    phase_round_count = 0                   # 当前阶段已消耗的轮次数
    total_round = 0                         # 全局轮次计数

    # 从 CONFIG 读取各阶段最大轮次
    phases_config = CONFIG.get("search_phases", {})

    # [v3新增] 注入第一个阶段的 system prompt
    # 这样 LLM 从一开始就知道自己处于哪个阶段、该做什么
    _first_phase_prompt = PHASE_PROMPTS.get(current_phase_name, "")
    if _first_phase_prompt:
        messages.append({"role": "system", "content": _first_phase_prompt})
        print(f"📍 进入阶段: {current_phase_name} (最大 {phases_config.get(current_phase_name, '?')} 轮)")

    # ========== 主循环 ==========
    for round_num in range(max_tool_rounds):
        # ---------- [v3新增] 阶段转换检查 ----------
        # 在每个 LLM 调用前，检查当前阶段是否已用完轮次
        # 如果是，自动推进到下一阶段
        _max_rounds_for_phase = phases_config.get(current_phase_name, 1)
        if phase_round_count >= _max_rounds_for_phase and current_phase_idx < len(PHASE_ORDER) - 1:
            # 当前阶段轮次已用完，推进到下一阶段
            current_phase_idx += 1
            current_phase_name = PHASE_ORDER[current_phase_idx]
            phase_round_count = 0

            # 注入新阶段的 system prompt
            _phase_prompt = PHASE_PROMPTS.get(current_phase_name, "")
            if _phase_prompt:
                messages.append({"role": "system", "content": _phase_prompt})
                print(f"📍 进入阶段: {current_phase_name} (最大 {phases_config.get(current_phase_name, '?')} 轮)")

                # 如果是最终验证阶段，提示 LLM 不要再调用工具
                if current_phase_name == "final_verify":
                    messages.append({
                        "role": "user",
                        "content": "你已进入最终验证阶段。请不要再调用任何搜索工具，"
                                   "直接基于之前所有搜索结果，输出最终的结构化 JSON 结果。"
                    })

        # ---------- API 调用 ----------
        try:
            response = llm_client.chat.completions.create(
                # [迁移改动] 模型名从 CONFIG 读取
                model=CONFIG["llm_model"],
                messages=messages,
                tools=TOOLS,
                temperature=0.5,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as e:
            print(f"❌ [Round {round_num}] API错误: {type(e).__name__}: {e}")
            raise

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        print(f"[Round {round_num}] ✅ finish_reason={finish_reason}, "
              f"tool_calls={len(msg.tool_calls) if msg.tool_calls else 0} | "
              f"阶段={current_phase_name} ({phase_round_count}/{phases_config.get(current_phase_name, '?')})")

        # ========== 分支A: 模型请求调用工具 ==========
        if msg.tool_calls:
            messages.append(msg)

            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                # --- 防护1: 查询去重 ---
                query_key = f"{func_name}:{json.dumps(func_args, sort_keys=True)}"
                if query_key in executed_queries:
                    tool_result = {
                        "warning": "此查询已在之前的轮次中执行过且结果相同。"
                                   "请基于已有搜索结果进行分析并直接输出最终JSON，不要重复搜索。"
                    }
                    print(f"  ⚠️ 重复查询已拦截: {func_name}({func_args})")
                else:
                    executed_queries.add(query_key)
                    # --- 防护2: 工具执行异常捕获 ---
                    try:
                        if func_name in TOOL_REGISTRY:
                            tool_result = TOOL_REGISTRY[func_name]["func"](**func_args)
                        else:
                            tool_result = {"error": f"未注册的工具: {func_name}"}
                        print(f"  🔧 {func_name}({func_args}) → 成功")
                    except Exception as tool_err:
                        tool_result = {"error": f"{type(tool_err).__name__}: {tool_err}"}
                        print(f"  ❌ {func_name}({func_args}) → 失败: {tool_err}")

                # --- 防护3: 工具结果不能为空字符串 ---
                result_str = json.dumps(tool_result, ensure_ascii=False)
                if not result_str or result_str == '""':
                    result_str = json.dumps({"info": "工具返回了空结果"}, ensure_ascii=False)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str
                })

            # [v3改动] 工具调用后，当前阶段的轮次计数 +1
            phase_round_count += 1
            total_round += 1
            consecutive_empty = 0
            continue

        # ========== 分支B: 模型返回文本（尝试解析JSON） ==========
        content = (msg.content or "").strip()

        # --- 防护4: 空内容处理 ---
        if not content:
            consecutive_empty += 1
            print(f"  ⚠️ 空内容 (连续第{consecutive_empty}次)")
            messages.append({"role": "assistant", "content": ""})
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                messages.append({
                    "role": "user",
                    "content": "你已连续多次返回空内容。请立即基于已有信息输出最终JSON结果，不要再调用工具。"
                })
            else:
                messages.append({
                    "role": "user",
                    "content": "你的回复为空。请直接输出JSON格式的结果。"
                })
            continue

        consecutive_empty = 0

        # --- 去除 markdown 代码块包裹 ---
        if content.startswith("```"):
            lines = content.split("\n")
            start = 1
            end = len(lines)
            if lines[-1].strip() == "```":
                end = -1
            content = "\n".join(lines[start:end]).strip()

        # --- 防护5: JSON 解析 + 阶段守卫 ---
        # [v5修复] 非 final_verify 阶段不允许直接输出 JSON 结束
        # 防止 LLM "偷懒" 跳过搜索直接给结果
        if current_phase_name != "final_verify":
            print(f"  ⚠️ 阶段守卫: {current_phase_name} 阶段不允许直接输出结果，强制继续搜索")
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"你当前处于【{current_phase_name}】阶段，尚未完成本阶段的搜索任务。"
                    "请不要直接输出最终结果。"
                    "请继续使用搜索工具（search_literature / search_weak / search_related / fetch_paper_content / verify_doi_reference / extract_claims / check_mechanism_consistency）"
                    "完成本阶段的搜索目标，只有在 final_verify 阶段才能输出最终 JSON。"
                )
            })
            phase_round_count += 1
            continue

        # 只有 final_verify 阶段才允许解析 JSON 并结束
        try:
            parsed = _safe_parse_llm_json(content)
            if "parse_error" in parsed:
                raise json.JSONDecodeError("parse_error", content, 0)
            print(f"[Round {round_num}] 🎉 JSON解析成功，Agent完成")
            print(f"📊 搜索统计: 总计 {total_round} 轮工具调用, {len(executed_queries)} 次不重复查询")
            return parsed
        except json.JSONDecodeError as e:
            print(f"  ⚠️ JSON解析失败: {e}")
            print(f"  📄 原始内容前200字: {content[:200]}")
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"你上次输出的JSON无效（错误: {e}）。"
                    "请仅输出合法的JSON对象，不要包含任何解释文字、markdown标记或代码块符号。"
                )
            })
            phase_round_count += 1
            continue
