# -*- coding: utf-8 -*-

import json
import re
import urllib.parse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import _safe_parse_llm_json, register_tool
from config import CONFIG, llm_client, llm_extra_params
from literature_searcher import (
    _extract_papers,
    _normalize_paper,
    fetch_paper_content,
    sciverse_client,
)


@register_tool(
    description="对当前最有可能的假设/结论进行可证伪性检查。"
    "通过模拟审稿人视角，识别潜在漏洞、分析证据反驳力、建议验证实验，"
    "最终判定该假设是否可证伪。"
    "仅在搜索阶段充分后、准备输出最终结论前调用此工具。",
    parameters={
        "type": "object",
        "properties": {
            "hypothesis": {
                "type": "string",
                "description": "需要检验的假设或结论，应基于前期搜索证据提出"
            },
            "evidence_summary": {
                "type": "string",
                "description": "支持该假设的证据摘要（不超过300字），来自前期搜索"
            }
        },
        "required": ["hypothesis", "evidence_summary"]
    }
)
def falsification_check(hypothesis: str, evidence_summary: str):
    """可证伪性检查工具：优先要求 JSON 输出，正则解析仅作兜底。"""
    try:
        prompt = f"""
你是一位严格的学术审稿人。请对以下假设进行可证伪性分析。

【待检验的假设】
{hypothesis}

【支持证据摘要】
{evidence_summary}

请严格输出 JSON（不要任何其他文字）：
{{
  "vulnerabilities": ["漏洞1的具体描述", "漏洞2的具体描述", "漏洞3的具体描述"],
  "rebuttals": [{{"vulnerability": "漏洞1", "rebuttal": "现有证据能在多大程度上反驳"}}],
  "required_experiments": ["验证实验1", "验证实验2"],
  "verdict": "可证伪 或 部分可证伪 或 不可证伪"
}}
"""
        response = llm_client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            extra_body=llm_extra_params(),
        )
        raw_output = response.choices[0].message.content.strip()
        parsed = _safe_parse_llm_json(raw_output)
        if isinstance(parsed, dict) and "parse_error" not in parsed:
            return {
                "status": "success",
                "vulnerabilities": (parsed.get("vulnerabilities") or [])[:3],
                "rebuttals": parsed.get("rebuttals", {}),
                "required_experiments": parsed.get("required_experiments", ""),
                "verdict": parsed.get("verdict", "未知"),
            }
        return _parse_falsification_output(raw_output)
    except Exception as e:
        return {
            "status": "parse_failed",
            "raw": f"Error during LLM call or parsing: {str(e)}"
        }


def _parse_falsification_output(text: str) -> dict:
    """解析证伪检查输出。"""
    try:
        text = re.sub(r'(?m)^\s*#{1,6}\s*', '', text)
        text = text.replace('**', '')
        vulnerabilities_match = re.search(r'潜在漏洞：\s*\n((?:\d+\.\s*.+\n?)+)', text)
        rebuttal_match = re.search(r'证据反驳分析：\s*\n((?:-\s*漏洞\d+：.+\n?)+)', text)
        experiment_match = re.search(r'所需验证实验：\s*\n(.+?)(?=\n最终判定：)', text, re.DOTALL)
        verdict_match = re.search(r'最终判定：\s*\n\s*(可证伪|部分可证伪|不可证伪)', text)

        vulnerabilities = []
        if vulnerabilities_match:
            vuln_text = vulnerabilities_match.group(1)
            vulnerabilities = [line.split('.', 1)[1].strip()
                             for line in vuln_text.strip().split('\n')
                             if line.strip()]

        rebuttals = {}
        if rebuttal_match:
            reb_text = rebuttal_match.group(1)
            for line in reb_text.strip().split('\n'):
                if '：' in line:
                    key, value = line.split('：', 1)
                    vuln_num = re.search(r'\d+', key)
                    if vuln_num:
                        rebuttals[f"漏洞{vuln_num.group()}"] = value.strip()

        return {
            "status": "success",
            "vulnerabilities": vulnerabilities[:3],
            "rebuttals": rebuttals,
            "required_experiments": experiment_match.group(1).strip() if experiment_match else "",
            "verdict": verdict_match.group(1) if verdict_match else "未知"
        }
    except Exception:
        return {"status": "parse_failed", "raw": text}


def _doi_exists(doi: str) -> dict:
    """用 CrossRef 核验 DOI 真实性；查不到时再用 DataCite 兜底（覆盖 arXiv/Zenodo）。"""
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto=research.assistant@example.com"
    try:
        req = Request(url, headers={"User-Agent": "GOAI-ResearchAssistant/1.0 (mailto:research.assistant@example.com)"})
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        m = data.get("message", {})
        year = None
        for k in ("published-print", "published-online", "issued"):
            parts = m.get(k, {}).get("date-parts") if m.get(k) else None
            if parts and parts[0]:
                year = parts[0][0]
                break
        return {
            "valid": True,
            "title": (m.get("title") or [""])[0],
            "year": year,
        }
    except Exception:
        pass

    try:
        url2 = f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}"
        req = Request(url2, headers={"User-Agent": "GOAI-ResearchAssistant/1.0"})
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        attrs = data.get("data", {}).get("attributes", {})
        titles = attrs.get("titles") or [{}]
        title = titles[0].get("title", "") if isinstance(titles, list) else ""
        return {
            "valid": True,
            "title": title,
            "year": attrs.get("publicationYear"),
        }
    except Exception:
        return {"valid": False, "title": "", "year": None}


@register_tool(
    description="【DOI验证】验证一个 DOI 是否真实存在，并判断给定断言是否被该论文支持。"
    "当某条证据声称来自某 DOI 时，调用本工具确认：1) DOI 是否真实存在；2) 该断言是否被论文内容支持。"
    "返回状态包括 verified / partial / unsupported / not_found。",
    parameters={
        "type": "object",
        "properties": {
            "doi": {
                "type": "string",
                "description": "待验证的 DOI，例如 '10.1038/nmat1718'"
            },
            "claim": {
                "type": "string",
                "description": "要验证的断言。"
            }
        },
        "required": ["doi", "claim"]
    }
)
def verify_doi_reference(doi: str, claim: str):
    """
    1. 用 DOI 调 meta_search 确认存在
    2. 用 fetch_paper_content(level='paragraphs') 获取段落
    3. 让 LLM 判断 claim 是否被支持
    """
    if not doi or not doi.startswith("10."):
        return {"status": "invalid", "doi": doi, "message": "DOI 格式异常，应以 10. 开头"}

    # 1. 用 CrossRef/DataCite 确认 DOI 真实存在并获取元数据
    try:
        info = _doi_exists(doi)
        if not info.get("valid"):
            return {"status": "not_found", "doi": doi, "message": "CrossRef/DataCite 未找到该 DOI"}
        paper = {
            "unique_id": None,
            "doc_id": None,
            "title": info.get("title", ""),
            "abstract": "",
            "doi": doi,
            "publication_published_year": info.get("year"),
            "publication_venue_name_unified": "",
        }
    except Exception as e:
        return {"status": "error", "doi": doi, "message": f"DOI 核验异常: {str(e)}"}

    # 2. 获取段落内容
    paragraphs_result = fetch_paper_content(
        doi=doi,
        level="paragraphs"
    )
    paragraphs = ""
    if paragraphs_result.get("status") == "success":
        paras = paragraphs_result.get("paragraphs", [])
        if paras:
            paragraphs = "\n\n".join([
                p.get("body", p.get("text", str(p))) for p in paras[:5]
            ])

    # 3. LLM 判断
    try:
        prompt = f"""
你是一位严格的文献核对专家。请根据以下论文信息，判断给定断言是否被该论文支持。

论文标题：{paper.get('title', '')}
论文摘要：{paper.get('abstract', '')}
论文段落：{paragraphs}

待验证断言：{claim}

请按以下格式输出：
- verdict: [supported / partially_supported / unsupported / not_found]
- 理由：...
- 引用的原文片段：...
- 注意：如果论文中完全没有提到断言中的关键概念，必须返回 not_found
"""
        resp = llm_client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            extra_body=llm_extra_params(),
        )
        llm_text = resp.choices[0].message.content.strip()

        # 解析 verdict：优先匹配 "verdict:" 行，其次整词匹配，避免 "not supported" 被 "supported" 误命中
        verdict = "not_found"
        llm_lower = llm_text.lower()
        m = re.search(r"verdict\s*[:：]\s*[\[（(]?\s*([a-z_]+)", llm_lower)
        if m:
            word = m.group(1)
            if word.startswith("partial"):
                verdict = "partial"
            elif word == "supported":
                verdict = "verified"
            elif word == "unsupported":
                verdict = "unsupported"
        else:
            if re.search(r"\bpartially[_\s-]*supported\b", llm_lower):
                verdict = "partial"
            elif re.search(r"\bunsupported\b|\bnot[_\s-]+supported\b", llm_lower):
                verdict = "unsupported"
            elif re.search(r"\bsupported\b", llm_lower):
                verdict = "verified"

        # 状态映射
        status_map = {
            "verified": "verified",
            "partial": "partial",
            "unsupported": "unsupported",
            "not_found": "not_found"
        }

        # 尝试提取原文片段
        evidence_quote = ""
        quote_match = re.search(r'引用的原文片段[：:]\s*(.+?)(?=\n|$)', llm_text, re.DOTALL)
        if quote_match:
            evidence_quote = quote_match.group(1).strip()

        return {
            "status": status_map.get(verdict, "not_found"),
            "doi": doi,
            "paper_title": paper.get("title", ""),
            "verdict": "断言被摘要/正文支持" if verdict == "verified" else
                       "断言被部分支持" if verdict == "partial" else
                       "断言与论文内容不符" if verdict == "unsupported" else
                       "无法确认",
            "evidence_quote": evidence_quote,
            "llm_analysis": llm_text[:800]
        }
    except Exception as e:
        return {
            "status": "error",
            "doi": doi,
            "paper_title": paper.get("title", ""),
            "message": f"LLM 验证异常: {str(e)}"
        }


@register_tool(
    description="【断言提取】从论文摘要或段落文本中，提取与某个假设相关的结构化断言。"
    "用于把非结构化的论文文本变成可验证的 claim 列表，每个 claim 标注置信度和相关实体。",
    parameters={
        "type": "object",
        "properties": {
            "paper_text": {
                "type": "string",
                "description": "论文摘要或段落文本"
            },
            "hypothesis": {
                "type": "string",
                "description": "可选，用于筛选相关 claim 的假设或研究问题"
            }
        },
        "required": ["paper_text"]
    }
)
def extract_claims(paper_text: str, hypothesis: str = ""):
    """用 LLM 从文本中提取与假设相关的结构化断言。"""
    try:
        prompt = f"""
请从以下论文内容中提取具体的事实断言。每个断言必须是论文明确表达的内容，不能是你自己的推断。

论文内容：
{paper_text[:4000]}

{"相关假设：" + hypothesis if hypothesis else ""}

输出格式：JSON 数组
[
  {{
    "claim_text": "论文中的断言",
    "entity": "涉及的材料/对象",
    "property": "性能/性质",
    "value": "数值（如有）",
    "unit": "单位（如有）",
    "confidence": "high/medium/low"
  }}
]

注意：
- 只提取论文中明确陈述的事实
- 数值必须带单位
- 如果某个断言只是背景介绍，不是研究发现，confidence 设为 low
"""
        resp = llm_client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            extra_body=llm_extra_params(),
        )
        content = resp.choices[0].message.content
        parsed = _safe_parse_llm_json(content)
        if isinstance(parsed, list):
            return {"status": "success", "hypothesis": hypothesis, "claims": parsed}
        elif "parse_error" in parsed:
            return {"status": "parse_failed", "hypothesis": hypothesis, "raw": parsed.get("raw", "")}
        else:
            return {"status": "success", "hypothesis": hypothesis, "claims": [parsed]}
    except Exception as e:
        return {"status": "error", "hypothesis": hypothesis, "error": str(e)}


@register_tool(
    description="【机制验证】验证给定假设在指定学科领域内的机制合理性。"
    "不依赖外部数据库，基于 LLM 的领域知识判断假设的物理/化学/学科机制是否自洽。"
    "适用于快速判断一个构效关系假设是否值得进一步验证。",
    parameters={
        "type": "object",
        "properties": {
            "hypothesis": {
                "type": "string",
                "description": "待验证的假设或构效关系"
            },
            "domain": {
                "type": "string",
                "description": "学科领域，例如 materials_science, chemistry, biology, physics"
            },
            "evidence_summary": {
                "type": "string",
                "description": "支持或反驳该假设的证据摘要（不超过500字）"
            }
        },
        "required": ["hypothesis", "domain"]
    }
)
def check_mechanism_consistency(hypothesis: str, domain: str, evidence_summary: str = ""):
    """基于 LLM 和证据摘要，验证假设的机制合理性。"""
    try:
        prompt = f"""
你是 {domain} 领域的专家。请判断以下假设在科学机制上是否合理。

假设：
{hypothesis}

{f"证据摘要：{evidence_summary}" if evidence_summary else ""}

请基于该领域已知的科学原理，回答：
1. 该假设是否与已知原理一致？
2. 有哪些已知的原理支持或反驳它？
3. 可能存在什么机制上的问题？

输出格式（严格 JSON）：
{{
  "is_consistent": true/false,
  "score": 0.0-1.0,
  "supporting_principles": ["..."],
  "conflicting_principles": ["..."],
  "potential_issues": ["..."],
  "reason": "..."
}}

评分标准：
- 0.8-1.0：机制清晰，与已知原理一致，因果链完整
- 0.5-0.79：机制基本合理，但存在未解决细节
- 0.2-0.49：机制牵强，与已知原理部分冲突
- 0.0-0.19：机制明显错误或与已知原理矛盾
"""
        resp = llm_client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            extra_body=llm_extra_params(),
        )
        content = resp.choices[0].message.content
        parsed = _safe_parse_llm_json(content)
        if "parse_error" in parsed:
            return {"status": "parse_failed", "raw": parsed.get("raw", "")}
        return {
            "status": "success",
            "hypothesis": hypothesis,
            "domain": domain,
            "is_consistent": parsed.get("is_consistent", False),
            "score": parsed.get("score", 0.0),
            "supporting_principles": parsed.get("supporting_principles", []),
            "conflicting_principles": parsed.get("conflicting_principles", []),
            "potential_issues": parsed.get("potential_issues", []),
            "reason": parsed.get("reason", "")
        }
    except Exception as e:
        return {"status": "error", "hypothesis": hypothesis, "error": str(e)}


def extract_gaps_from_research_result(result: dict) -> list:
    """
    从 run_agent 返回的 JSON 中提取 Research Gaps。
    兼容多种可能的字段名和嵌套格式。
    """
    if not isinstance(result, dict):
        return []

    # 优先读取标准字段
    for key in ["research_gaps", "gaps", "research_gaps_list", "identified_gaps"]:
        gaps = result.get(key)
        if isinstance(gaps, list):
            return [str(g).strip() for g in gaps if g]

    # 兜底：从文本字段中按行提取包含 "gap" 或 "不足" / "缺失" 的句子
    for key in ["summary", "conclusion", "findings", "output", "analysis"]:
        text = result.get(key, "")
        if not isinstance(text, str):
            continue
        candidates = re.split(r'[。\n]', text)
        gaps = [c.strip() for c in candidates
                if any(kw in c.lower() for kw in ["gap", "不足", "缺失", "空白", "未解决", "不明确"])]
        if gaps:
            return gaps

    return []
