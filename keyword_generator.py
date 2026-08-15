# -*- coding: utf-8 -*-

import json
import re


def extract_keywords(search_results_text: str) -> list:
    """
    从搜索结果文本中提取关键词/实体。
    优先解析 JSON 结构提取标题/年份/DOI/关键词字段，
    解析失败时退化为正则兜底。
    """
    papers = _parse_papers(search_results_text)
    if papers:
        keywords = []
        seen_titles = set()
        for p in papers:
            if not isinstance(p, dict):
                continue
            title = p.get("title")
            if title and len(str(title)) > 10 and str(title) not in seen_titles:
                seen_titles.add(str(title))
                keywords.append({"type": "title", "value": str(title)})
            year = p.get("publication_published_year") or p.get("year")
            if year:
                keywords.append({"type": "year", "value": str(year)})
            doi = p.get("doi")
            if doi:
                keywords.append({"type": "doi", "value": str(doi)})
            raw_keywords = p.get("keywords")
            if isinstance(raw_keywords, list):
                for kw in raw_keywords[:3]:
                    if isinstance(kw, str) and kw.strip():
                        keywords.append({"type": "concept", "value": kw.strip()})
        return _dedupe(keywords)[:20]

    return _regex_extract(search_results_text)


def _parse_papers(text):
    """尝试把文本解析为论文列表。"""
    try:
        data = json.loads(text)
    except Exception:
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "papers", "data", "hits", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return None


def _dedupe(keywords):
    seen = set()
    out = []
    for kw in keywords:
        key = (kw["type"], str(kw["value"]).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(kw)
    return out


def _regex_extract(search_results_text: str) -> list:
    """正则兜底：仅在结构化解析失败时使用。"""
    keywords = []

    title_pattern = r'"title"\s*:\s*"([^"]+)"'
    titles = re.findall(title_pattern, search_results_text)
    for t in titles[:5]:
        if len(t) > 10:
            keywords.append({"type": "title", "value": t})

    year_pattern = r'"publication_published_year"\s*:\s*(\d{4})'
    years = re.findall(year_pattern, search_results_text)
    unique_years = list(set(years))[:3]
    for y in unique_years:
        keywords.append({"type": "year", "value": y})

    doi_pattern = r'"doi"\s*:\s*"([^"]+)"'
    dois = re.findall(doi_pattern, search_results_text)
    for doi in dois[:3]:
        keywords.append({"type": "doi", "value": doi})

    concept_pattern = r'"([^"]{4,50})"'
    concepts = re.findall(concept_pattern, search_results_text)
    skip_words = {
        "title", "doi", "status", "query", "results", "error",
        "success", "publication_published_year", "type", "value",
        "publication_venue_name_unified", "search_type", "entity",
        "relation_type", "year_min", "page_size", "fields",
    }
    for c in concepts[:10]:
        if c.lower() not in skip_words and not c.startswith("sci_") and not c.startswith("sk-"):
            keywords.append({"type": "concept", "value": c})

    return _dedupe(keywords)[:20]
