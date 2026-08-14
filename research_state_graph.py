# -*- coding: utf-8 -*-

import json
from typing import Literal

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

import config
from gap_analyzer import (
    check_mechanism_consistency,
    extract_claims,
    extract_gaps_from_research_result,
    falsification_check,
)
from keyword_generator import extract_keywords
from literature_searcher import search_literature


class ResearchContext(TypedDict):
    gaps: list[str]
    cached_literature: list[dict]
    target_property: str


class ApiStats(TypedDict):
    total_calls: int
    cache_hits: int
    errors: int


class ResearchState(TypedDict):
    phase: str
    context: ResearchContext
    api_stats: ApiStats
    query: str
    year_min: int | None
    page_size: int
    status: str
    error: str
    api_error: object
    total_returned: int
    papers: list[dict]
    raw: dict
    research_result: dict
    gaps: list[str]
    keywords: list[dict]
    executed_queries: set[str]
    total_round: int
    max_tool_rounds: int
    hypothesis: str
    evidence_summary: str
    domain: str
    claims: list[dict]
    vulnerabilities: list[str]
    rebuttals: dict
    required_experiments: str
    verdict: str
    score: float
    is_consistent: bool


# TODO: 需人工确认。
# 原文献循环没有 evidence_score 终止阈值；0.8 来自
# check_mechanism_consistency 原 Prompt 中“0.8-1.0：机制清晰”的评分区间。
SCORE_THRESHOLD = 0.8


def search_node(state: ResearchState) -> dict:
    """从 State 映射 search_literature 的输入，并回填其真实返回字段。"""
    func_args = {
        "query": state["query"],
        "year_min": state["year_min"],
        "page_size": state["page_size"],
    }
    query_key = f"search_literature:{json.dumps(func_args, sort_keys=True)}"
    executed_queries = set(state["executed_queries"])

    if query_key in executed_queries:
        tool_result = {
            "warning": "此查询已在之前的轮次中执行过且结果相同。"
                       "请基于已有搜索结果进行分析并直接输出最终JSON，不要重复搜索。"
        }
        return {
            "research_result": tool_result,
            "total_round": state["total_round"] + 1,
        }

    tool_result = search_literature(**func_args)
    executed_queries.add(query_key)

    papers = tool_result.get("papers", state["papers"])
    context = dict(state["context"])
    context["cached_literature"] = context["cached_literature"] + papers

    return {
        "phase": "research",
        "context": context,
        "api_stats": dict(config.state["api_stats"]),
        "status": tool_result.get("status", "error"),
        "error": tool_result.get("error", ""),
        "api_error": tool_result.get("api_error"),
        "total_returned": tool_result.get("total_returned", 0),
        "papers": papers,
        "raw": tool_result.get("raw", {}),
        "executed_queries": executed_queries,
        "total_round": state["total_round"] + 1,
    }


def analyze_node(state: ResearchState) -> dict:
    """调用现有Gap分析函数，并把真实返回字段映射回 State。"""
    if state["status"] != "success":
        return {
            "status": state["status"],
            "research_result": {
                "status": state["status"],
                "error": state["error"],
                "api_error": state["api_error"],
            },
        }

    if not state["papers"]:
        return {
            "status": "error",
            "error": "Sciverse 检索成功，但没有返回论文。",
            "research_result": {
                "status": "error",
                "error": "Sciverse 检索成功，但没有返回论文。",
            },
        }

    paper_text = json.dumps(state["papers"], ensure_ascii=False)
    hypothesis = state["hypothesis"]

    claims_result = extract_claims(
        paper_text=paper_text,
        hypothesis=hypothesis,
    )
    claims = claims_result.get("claims", [])
    evidence_summary = json.dumps(claims, ensure_ascii=False)[:300]

    falsification_result = falsification_check(
        hypothesis=hypothesis,
        evidence_summary=evidence_summary,
    )
    mechanism_result = check_mechanism_consistency(
        hypothesis=hypothesis,
        domain=state["domain"],
        evidence_summary=evidence_summary,
    )

    analysis_status = mechanism_result.get(
        "status",
        falsification_result.get("status", claims_result.get("status", "error")),
    )
    if (
        claims_result.get("status") != "success"
        or falsification_result.get("status") != "success"
        or mechanism_result.get("status") != "success"
    ):
        return {
            "status": analysis_status,
            "research_result": state["research_result"],
            "gaps": state["gaps"],
            "hypothesis": hypothesis,
            "evidence_summary": state["evidence_summary"],
            "claims": state["claims"],
            "vulnerabilities": state["vulnerabilities"],
            "rebuttals": state["rebuttals"],
            "required_experiments": state["required_experiments"],
            "verdict": state["verdict"],
            "score": state["score"],
            "is_consistent": state["is_consistent"],
        }

    research_result = {
        "status": "success",
        "gaps": falsification_result.get("vulnerabilities", []),
        "claims": claims,
        "vulnerabilities": falsification_result.get("vulnerabilities", []),
        "rebuttals": falsification_result.get("rebuttals", {}),
        "required_experiments": falsification_result.get("required_experiments", ""),
        "verdict": falsification_result.get("verdict", "未知"),
        "score": mechanism_result.get("score", 0.0),
        "is_consistent": mechanism_result.get("is_consistent", False),
    }
    gaps = extract_gaps_from_research_result(research_result)

    return {
        "status": mechanism_result.get("status", falsification_result.get("status", "error")),
        "research_result": research_result,
        "gaps": gaps,
        "hypothesis": hypothesis,
        "evidence_summary": evidence_summary,
        "claims": claims,
        "vulnerabilities": falsification_result.get("vulnerabilities", []),
        "rebuttals": falsification_result.get("rebuttals", {}),
        "required_experiments": falsification_result.get("required_experiments", ""),
        "verdict": falsification_result.get("verdict", "未知"),
        "score": mechanism_result.get("score", 0.0),
        "is_consistent": mechanism_result.get("is_consistent", False),
    }


def gen_keywords_node(state: ResearchState) -> dict:
    """调用现有extract_keywords，并更新Gap和查询历史对应的State字段。"""
    search_results_text = json.dumps(state["papers"], ensure_ascii=False)
    keywords = [
        keyword
        for keyword in extract_keywords(search_results_text)
        if str(keyword.get("value", "")).strip().lower()
        not in {"gaps", "papers"}
    ]

    context = dict(state["context"])
    gaps = list(context["gaps"])
    history = {
        gap.strip().lower()
        for gap in gaps
        if gap
    }
    new_gaps = [
        gap
        for gap in state["gaps"]
        if gap and gap.strip().lower() not in history
    ]
    for gap in state["gaps"]:
        if gap not in gaps:
            gaps.append(gap)
    context["gaps"] = gaps

    topic_prefix = next(
        (
            line.strip()
            for line in context["target_property"].splitlines()
            if line.strip()
        ),
        state["hypothesis"],
    )
    query = state["query"]
    for gap in new_gaps:
        gap_short = gap[:80]
        candidate_query = f"{topic_prefix} {gap_short}"
        func_args = {
            "query": candidate_query,
            "year_min": state["year_min"],
            "page_size": state["page_size"],
        }
        query_key = f"search_literature:{json.dumps(func_args, sort_keys=True)}"
        if query_key not in state["executed_queries"]:
            query = candidate_query
            break

    return {
        "context": context,
        "keywords": keywords,
        "query": query,
    }


def should_continue(state: ResearchState) -> Literal["gen_keywords_node", "__end__"]:
    """按原轮次上限及现有Gap、score字段决定继续或结束。"""
    if state["total_round"] >= state["max_tool_rounds"]:
        return END

    if state["score"] >= SCORE_THRESHOLD:
        return END

    gaps = state["gaps"]
    if not gaps:
        return END

    history = {
        gap.strip().lower()
        for gap in state["context"]["gaps"]
        if gap
    }
    if all(gap.strip().lower() in history for gap in gaps if gap):
        return END

    return "gen_keywords_node"


graph_builder = StateGraph(ResearchState)
graph_builder.add_node("search_node", search_node)
graph_builder.add_node("analyze_node", analyze_node)
graph_builder.add_node("gen_keywords_node", gen_keywords_node)

graph_builder.add_edge(START, "search_node")
graph_builder.add_edge("search_node", "analyze_node")
graph_builder.add_conditional_edges("analyze_node", should_continue)
graph_builder.add_edge("gen_keywords_node", "search_node")

graph = graph_builder.compile()


if __name__ == "__main__":
    phases_config = config.CONFIG.get("search_phases", {})
    max_tool_rounds = sum(phases_config.values()) + 2
    initial_state: ResearchState = {
        "phase": "research",
        "context": {
            "gaps": [],
            "cached_literature": [],
            "target_property": "LiFePO4 magnesium doping rate performance",
        },
        "api_stats": {
            "total_calls": 0,
            "cache_hits": 0,
            "errors": 0,
        },
        "query": "LiFePO4 magnesium doping rate performance",
        "year_min": None,
        "page_size": 2,
        "status": "",
        "error": "",
        "api_error": None,
        "total_returned": 0,
        "papers": [],
        "raw": {},
        "research_result": {},
        "gaps": [],
        "keywords": [],
        "executed_queries": set(),
        "total_round": 0,
        "max_tool_rounds": max_tool_rounds,
        "hypothesis": "LiFePO4 magnesium doping rate performance",
        "evidence_summary": "",
        "domain": "materials_science",
        "claims": [],
        "vulnerabilities": [],
        "rebuttals": {},
        "required_experiments": "",
        "verdict": "未知",
        "score": 0.0,
        "is_consistent": False,
    }

    result = graph.invoke(
        initial_state,
        {"recursion_limit": max_tool_rounds * 3 + 1},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=list))
