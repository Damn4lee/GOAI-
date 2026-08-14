# -*- coding: utf-8 -*-

import re


def extract_keywords(search_results_text: str) -> list:
    """
    从搜索结果文本中提取关键词/实体 - 内部辅助函数
    """
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
    skip_words = {"title", "doi", "status", "query", "results", "error",
                  "success", "publication_published_year", "type", "value",
                  "publication_venue_name_unified", "search_type", "entity",
                  "relation_type", "year_min", "page_size", "fields"}
    for c in concepts[:10]:
        if c.lower() not in skip_words and not c.startswith("sci_") and not c.startswith("sk-"):
            keywords.append({"type": "concept", "value": c})

    return keywords[:20]
