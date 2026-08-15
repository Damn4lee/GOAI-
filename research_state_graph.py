# -*- coding: utf-8 -*-

import json
from typing import Literal

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
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
    """通过CrewAI协作调用现有Gap分析函数，并把真实返回字段映射回 State。"""
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

    def build_state_update(claims_result, falsification_result, mechanism_result):
        claims = claims_result.get("claims", [])
        evidence_summary = json.dumps(claims, ensure_ascii=False)[:300]
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
            "status": mechanism_result.get(
                "status",
                falsification_result.get("status", "error"),
            ),
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

    def run_original_analysis():
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
        return build_state_update(
            claims_result,
            falsification_result,
            mechanism_result,
        )

    crew_results = {}

    @tool("extract_claims", result_as_answer=True)
    def extract_claims_tool() -> str:
        """调用现有extract_claims处理State中的papers和hypothesis。"""
        claims_result = extract_claims(
            paper_text=paper_text,
            hypothesis=hypothesis,
        )
        claims = claims_result.get("claims", [])
        evidence_summary = json.dumps(claims, ensure_ascii=False)[:300]
        crew_results["claims_result"] = claims_result
        crew_results["evidence_summary"] = evidence_summary
        return json.dumps(
            {
                "status": claims_result.get("status", "error"),
                "hypothesis": hypothesis,
                "claims": claims,
                "evidence_summary": evidence_summary,
            },
            ensure_ascii=False,
        )

    @tool("falsification_check", result_as_answer=True)
    def falsification_check_tool() -> str:
        """调用现有falsification_check处理hypothesis和evidence_summary。"""
        falsification_result = falsification_check(
            hypothesis=hypothesis,
            evidence_summary=crew_results.get("evidence_summary", ""),
        )
        crew_results["falsification_result"] = falsification_result
        return json.dumps(falsification_result, ensure_ascii=False)

    @tool("check_mechanism_consistency", result_as_answer=True)
    def check_mechanism_consistency_tool() -> str:
        """调用现有check_mechanism_consistency并生成兼容的State更新。"""
        mechanism_result = check_mechanism_consistency(
            hypothesis=hypothesis,
            domain=state["domain"],
            evidence_summary=crew_results.get("evidence_summary", ""),
        )
        state_update = build_state_update(
            crew_results["claims_result"],
            crew_results["falsification_result"],
            mechanism_result,
        )
        crew_results["mechanism_result"] = mechanism_result
        crew_results["state_update"] = state_update
        return json.dumps(state_update, ensure_ascii=False)

    try:
        crew_llm = LLM(
            model=config.CONFIG["llm_model"],
            custom_openai=True,
            base_url=config.CONFIG["llm_base_url"],
            api_key=config.CONFIG["deepseek_key"].strip(),
            temperature=0.3,
            max_tokens=1024,
            additional_params=(
                {"extra_body": {"thinking": {"type": "disabled"}}}
                if config.CONFIG.get("disable_thinking", True)
                else {}
            ),
        )

        collector_agent = Agent(
            role="文献事实整理专家",
            goal=(
                "从State的papers中提取与hypothesis相关、由论文明确表达的事实断言，"
                "并形成evidence_summary。"
            ),
            backstory="你只接受论文明确表达的事实，不把自己的推断写成论文结论。",
            tools=[extract_claims_tool],
            allow_delegation=False,
            llm=crew_llm,
        )
        reviewer_agent = Agent(
            role="严格的学术审稿人",
            goal=(
                "基于hypothesis和evidence_summary执行falsification_check，"
                "识别vulnerabilities、rebuttals、required_experiments和verdict。"
            ),
            backstory="你是一位严格的学术审稿人，负责对假设进行可证伪性分析。",
            tools=[falsification_check_tool],
            allow_delegation=False,
            llm=crew_llm,
        )
        analyst_agent = Agent(
            role=f"{state['domain']}领域专家",
            goal=(
                "基于hypothesis、domain和evidence_summary判断机制一致性，"
                "并生成与原analyze_node完全兼容的research_result和State更新。"
            ),
            backstory=(
                f"你是{state['domain']}领域的专家，"
                "基于已知科学原理判断假设的机制是否自洽。"
            ),
            tools=[check_mechanism_consistency_tool],
            allow_delegation=False,
            llm=crew_llm,
        )

        collect_task = Task(
            description=(
                "读取State字段papers和hypothesis。必须调用extract_claims工具。"
                "只提取论文中明确陈述的事实，不能加入自己的推断；"
                "每条claims包含claim_text、entity、property、value、unit、confidence；"
                "数值必须带单位；背景介绍的confidence设为low。"
                "将claims序列化并截取前300个字符形成evidence_summary。"
                f"当前papers共{len(state['papers'])}条，hypothesis为：{hypothesis}"
            ),
            expected_output=(
                "extract_claims工具返回的JSON对象，字段为status、hypothesis、"
                "claims、evidence_summary；不得改写工具结果。"
            ),
            agent=collector_agent,
        )
        falsification_task = Task(
            description=(
                "读取State字段hypothesis，并使用上一任务产生的evidence_summary。"
                "必须调用falsification_check工具。按原逻辑给出3个具体的潜在漏洞，"
                "逐项说明现有证据能在多大程度上反驳漏洞，列出所需验证实验，"
                "最终verdict只能是可证伪、部分可证伪或不可证伪。"
                f"当前hypothesis为：{hypothesis}"
            ),
            expected_output=(
                "falsification_check工具返回的JSON对象，包含status、vulnerabilities、"
                "rebuttals、required_experiments、verdict；不得改写工具结果。"
            ),
            agent=reviewer_agent,
            context=[collect_task],
        )
        synthesize_task = Task(
            description=(
                "读取State字段hypothesis、domain、evidence_summary、claims、"
                "vulnerabilities、rebuttals、required_experiments和verdict。"
                "必须调用check_mechanism_consistency工具。判断假设是否与已知原理一致、"
                "支持和反驳原理以及潜在机制问题。score严格使用原评分标准："
                "0.8-1.0机制清晰且因果链完整；0.5-0.79基本合理但有未解决细节；"
                "0.2-0.49机制牵强且部分冲突；0.0-0.19机制明显错误或矛盾。"
                "工具会返回与原analyze_node相同的State更新，必须原样作为最终输出。"
                f"当前domain为：{state['domain']}。"
            ),
            expected_output=(
                "严格JSON对象，字段为status、research_result、gaps、hypothesis、"
                "evidence_summary、claims、vulnerabilities、rebuttals、"
                "required_experiments、verdict、score、is_consistent；不得添加包装字段。"
            ),
            agent=analyst_agent,
            context=[collect_task, falsification_task],
        )

        crew = Crew(
            agents=[collector_agent, reviewer_agent, analyst_agent],
            tasks=[collect_task, falsification_task, synthesize_task],
            process=Process.sequential,
            verbose=False,
        )
        crew_output = crew.kickoff()
        raw_output = crew_output.raw.strip()
        if raw_output.startswith("```"):
            output_lines = raw_output.splitlines()
            raw_output = "\n".join(output_lines[1:-1]).strip()
        parsed_output = json.loads(raw_output)
        expected_fields = (
            "status",
            "research_result",
            "gaps",
            "hypothesis",
            "evidence_summary",
            "claims",
            "vulnerabilities",
            "rebuttals",
            "required_experiments",
            "verdict",
            "score",
            "is_consistent",
        )
        if not isinstance(parsed_output, dict) or any(
            field not in parsed_output for field in expected_fields
        ):
            raise ValueError("CrewAI输出与原analyze_node返回结构不兼容。")
        return {field: parsed_output[field] for field in expected_fields}
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ CrewAI 编排失败({type(e).__name__}: {e})，已回退到原有分析逻辑")
        return run_original_analysis()


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
