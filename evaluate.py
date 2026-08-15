# -*- coding: utf-8 -*-
"""评测脚手架：用少量带预期答案的问题量化 Agent 改版效果。

指标：
- 是否产出合法 JSON；
- 关键字段是否齐全（summary / evidence_chain / confidence）；
- 证据链中 DOI 的核验通过率（CrossRef/DataCite）；
- 预期关键文献是否被检索到（recall 占位，需人工补充答案）。

用法：
    python evaluate.py
"""

import json
import sys

from gap_analyzer import _doi_exists
from research_agent import run_agent


SAMPLE_QUESTIONS = [
    {
        "question": "Mg掺杂LiFePO4正极材料中，掺杂比例和烧结温度对5C倍率容量保持率有什么影响？",
        "expected_dois": [],  # 占位：填入人工确认的关键文献 DOI 后即可计算召回率
    },
    {
        "question": "钙钛矿太阳能电池中哪些添加剂能提升长期稳定性？",
        "expected_dois": [],
    },
]


def evaluate_one(item):
    q = item["question"]
    try:
        result = run_agent(q)
        valid_json = isinstance(result, dict)
        keys = result.keys() if valid_json else []
        has_summary = "summary" in keys
        has_chain = "evidence_chain" in keys and isinstance(result.get("evidence_chain"), list)
        dois = []
        for c in result.get("evidence_chain") or []:
            if c.get("doi"):
                dois.append(c["doi"])
        doi_results = {d: _doi_exists(d) for d in dict.fromkeys(dois)}
        doi_valid = sum(1 for r in doi_results.values() if r.get("valid"))
        expected = set(item.get("expected_dois") or [])
        found = set(dois) & expected
        recall = len(found) / len(expected) if expected else None
        return {
            "question": q[:50],
            "valid_json": valid_json,
            "has_summary": has_summary,
            "has_evidence_chain": has_chain,
            "doi_total": len(doi_results),
            "doi_valid": doi_valid,
            "expected_recall": recall,
            "confidence": result.get("confidence") if valid_json else None,
        }
    except Exception as e:  # noqa: BLE001
        return {"question": q[:50], "error": f"{type(e).__name__}: {e}"}


def main():
    questions = SAMPLE_QUESTIONS
    if len(sys.argv) > 1:
        questions = [{"question": " ".join(sys.argv[1:]), "expected_dois": []}]
    results = [evaluate_one(item) for item in questions]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
