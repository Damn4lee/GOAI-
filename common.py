# -*- coding: utf-8 -*-

import json
import re


TOOL_REGISTRY = {}


def register_tool(description: str, parameters: dict):
    """
    工具注册装饰器 - 唯一注册入口

    用法：在 def 函数上方加 @register_tool(description="...", parameters={...})
    """
    def decorator(func):
        """注册工具函数。"""
        func._tool_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters
            }
        }
        TOOL_REGISTRY[func.__name__] = {
            "func": func,
            "schema": func._tool_schema,
            "description": description
        }
        return func
    return decorator


def get_tools_list() -> list:
    """生成给 LLM 的 tools 参数列表"""
    return [tool["schema"] for tool in TOOL_REGISTRY.values()]


def _safe_parse_llm_json(text: str) -> dict:
    """从 LLM 输出中安全解析 JSON"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        start = 1
        end = len(lines)
        if lines[-1].strip() == "```":
            end = -1
        cleaned = "\n".join(lines[start:end]).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
            match = re.search(pattern, cleaned)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    continue
    return {"parse_error": True, "raw": text[:500]}
