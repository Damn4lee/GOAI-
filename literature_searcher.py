# -*- coding: utf-8 -*-

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import _safe_parse_llm_json, register_tool
from config import (
    CACHE_FILE_PATH,
    CONFIG,
    LOG_FILE_PATH,
    SCIVERSE_API_TOKEN,
    llm_client,
    state,
)


def check_sciverse_connection(token: str) -> bool:
    """
    检测 Sciverse API 是否接入成功
    发送一个最小化的 meta-search 请求，根据响应判断连通性
    :param token: Bearer API Token
    :return: True=接入成功, False=接入失败
    """
    url = "https://api.sciverse.space/meta-search"
    payload = json.dumps({
        "query": "test",
        "page": 1,
        "page_size": 1,
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    print("=" * 50)
    print("🔍 正在检测 Sciverse API 接入状态...")
    print("=" * 50)

    # 检查Token是否为空或占位符
    if not token or token in ("YOUR_API_TOKEN", "在这里粘贴你的真实Token"):
        print("❌ 接入失败：尚未填写有效的 API Token")
        print("   请在 CONFIG['sciverse_token'] 中填入你的真实Token")
        print("=" * 50)
        return False

    try:
        req = Request(url, data=payload, headers=headers, method="POST")
        with urlopen(req, timeout=15) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))

            has_results = "results" in body or "data" in body or "hits" in body
            if status == 200 and has_results:
                print(f"✅ 接入成功！HTTP {status}")
                print(f"   返回数据键: {list(body.keys())}")
                result_count = len(body.get("results", body.get("data", body.get("hits", []))))
                print(f"   测试查询返回 {result_count} 条结果")
                print("=" * 50)
                return True
            else:
                print(f"⚠️  响应异常：HTTP {status}，但返回结构不符合预期")
                print(f"   返回内容: {json.dumps(body, ensure_ascii=False)[:200]}")
                print("=" * 50)
                return False

    except HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8")[:200]
        except Exception:
            pass
        if e.code == 401:
            print(f"❌ 接入失败：认证错误 (HTTP 401)")
            print(f"   Token 无效或已过期，请检查后重试")
        elif e.code == 403:
            print(f"❌ 接入失败：权限不足 (HTTP 403)")
            print(f"   当前账号可能未开通 meta-search 权限")
        elif e.code == 429:
            print(f"⚠️  触发限流 (HTTP 429)，但说明Token有效、接入正常")
            print(f"   请稍后再运行检测")
            print("=" * 50)
            return True  # 429说明认证通过了
        else:
            print(f"❌ 接入失败：HTTP {e.code}")
        if error_body:
            print(f"   服务端响应: {error_body}")
        print("=" * 50)
        return False

    except URLError as e:
        print(f"❌ 接入失败：网络错误")
        print(f"   原因: {e.reason}")
        print(f"   请检查网络连接及代理设置")
        print("=" * 50)
        return False

    except Exception as e:
        print(f"❌ 接入失败：未知错误 - {type(e).__name__}: {e}")
        print("=" * 50)
        return False


class SciverseClient:
    """
    Sciverse API 多端点调用封装
    - 内置 JSON 文件缓存（key 包含 endpoint + payload MD5）
    - 令牌桶限速器（所有端点共享）
    - 429/5xx 自动指数退避重试
    - 400 错误记录详细请求/响应并返回结构化错误
    - 仅使用 Python 标准库
    """

    BASE_URLS = {
        "meta-search": "https://api.sciverse.space/meta-search",
        "agentic-search": "https://api.sciverse.space/agentic-search",
        "content": "https://api.sciverse.space/content",
        "meta-paper-relations": "https://api.sciverse.space/meta-paper-relations",
        "meta-catalog": "https://api.sciverse.space/meta-catalog",
    }

    CACHE_FILE = "sciverse_cache.json"
    LOG_FILE = "log.txt"
    MAX_REQUESTS_PER_MINUTE = 25
    MAX_RETRIES = 3

    def __init__(self, api_token: str, state: dict = None):
        """初始化 Sciverse API 客户端。"""
        self.api_token = api_token
        self.state = state
        self.cache = self._load_cache()
        self.cache_lock = threading.Lock()

        self.token_bucket_lock = threading.Lock()
        self.tokens = self.MAX_REQUESTS_PER_MINUTE
        self.last_refill_time = time.monotonic()
        self.refill_rate = self.MAX_REQUESTS_PER_MINUTE / 60.0

    # ==================== 缓存 ====================

    def _load_cache(self) -> dict:
        """加载本地缓存。"""
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        """保存本地缓存。"""
        try:
            with open(self.CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except IOError as e:
            self._write_log("SYSTEM", "CACHE_SAVE_ERROR", False, 0, str(e))

    @staticmethod
    def _make_cache_key(endpoint: str, payload: dict) -> str:
        """生成缓存键。"""
        raw = f"{endpoint}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    # ==================== 限速 ====================

    def _acquire_token(self):
        """从令牌桶获取请求令牌。"""
        while True:
            with self.token_bucket_lock:
                now = time.monotonic()
                elapsed = now - self.last_refill_time
                new_tokens = elapsed * self.refill_rate
                self.tokens = min(self.MAX_REQUESTS_PER_MINUTE, self.tokens + new_tokens)
                self.last_refill_time = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return

            wait_time = (1.0 - self.tokens) / self.refill_rate
            time.sleep(wait_time)

    # ==================== 日志 ====================

    def _write_log(self, timestamp: str, query: str, cache_hit: bool,
                   duration_ms: float, extra: str = ""):
        """写入请求日志。"""
        cache_str = "HIT" if cache_hit else "MISS"
        line = f"[{timestamp}] | {query} | {cache_str} | {duration_ms:.1f}ms"
        if extra:
            line += f" | {extra}"
        line += "\n"
        try:
            with open(self.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
        except IOError:
            pass

    # ==================== HTTP 请求 ====================

    def _do_request(self, endpoint: str, payload: dict) -> dict:
        """
        通用请求方法。
        成功：返回 API 响应 dict。
        失败：返回 {"error": True, "status_code": ..., "message": ..., "endpoint": ..., "payload": ...}
        """
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        url = self.BASE_URLS.get(endpoint, endpoint)
        last_exception = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                req = Request(url, data=data, headers=headers, method="POST")
                with urlopen(req, timeout=30) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)

            except HTTPError as e:
                last_exception = e
                status_code = e.code
                try:
                    err_body = e.read().decode("utf-8")
                except Exception:
                    err_body = ""
                try:
                    err_json = json.loads(err_body) if err_body else {}
                except Exception:
                    err_json = {"raw": err_body}

                # 400 记录详细错误并直接返回结构化错误
                if status_code == 400:
                    return {
                        "error": True,
                        "status_code": 400,
                        "message": f"HTTP 400 Bad Request: {err_body[:500]}",
                        "endpoint": endpoint,
                        "payload": payload,
                        "api_error": err_json,
                    }

                if status_code == 429 or 500 <= status_code < 600:
                    if attempt < self.MAX_RETRIES:
                        backoff = 2 ** (attempt + 1)
                        time.sleep(backoff)
                        continue

                return {
                    "error": True,
                    "status_code": status_code,
                    "message": f"HTTP {status_code}: {err_body[:500]}",
                    "endpoint": endpoint,
                    "payload": payload,
                    "api_error": err_json,
                }

            except URLError as e:
                last_exception = e
                if attempt < self.MAX_RETRIES:
                    backoff = 2 ** (attempt + 1)
                    time.sleep(backoff)
                    continue
                return {
                    "error": True,
                    "status_code": None,
                    "message": f"URLError: {str(e)}",
                    "endpoint": endpoint,
                    "payload": payload,
                }

        return {
            "error": True,
            "status_code": None,
            "message": f"最终失败: {str(last_exception)}",
            "endpoint": endpoint,
            "payload": payload,
        }

    # ==================== 端点封装 ====================

    def _call_endpoint(self, endpoint: str, payload: dict, log_label: str = "") -> dict:
        """
        带缓存、限速、日志的通用端点调用。
        """
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cache_key = self._make_cache_key(endpoint, payload)

        with self.cache_lock:
            if cache_key in self.cache:
                elapsed_ms = (time.time() - start_time) * 1000
                self._write_log(timestamp, log_label or endpoint, True, elapsed_ms)
                if self.state is not None:
                    self.state["api_stats"]["cache_hits"] += 1
                return self.cache[cache_key]

        self._acquire_token()

        result = self._do_request(endpoint, payload)

        elapsed_ms = (time.time() - start_time) * 1000

        # 更新统计
        if self.state is not None:
            if result.get("error"):
                self.state["api_stats"]["errors"] += 1
            else:
                self.state["api_stats"]["total_calls"] += 1

        # 记录日志
        extra = ""
        if result.get("error"):
            extra = f"ERROR: {result.get('message', '')[:200]}"
        self._write_log(timestamp, log_label or endpoint, False, elapsed_ms, extra)

        # 缓存成功响应
        if not result.get("error"):
            with self.cache_lock:
                self.cache[cache_key] = result
                self._save_cache()

        return result

    def meta_search(self, query: str, filters: list = None, fields: list = None,
                    page: int = 1, page_size: int = 5) -> dict:
        """调用 meta-search 端点。"""
        payload = {
            "query": query,
            "page": page,
            "page_size": page_size,
        }
        if filters is not None:
            payload["filters"] = filters
        if fields is not None:
            payload["fields"] = fields
        return self._call_endpoint("meta-search", payload, log_label=f"meta-search:{query[:40]}")

    # 保留 search 别名，兼容旧代码（如 Cell 6 GA）
    def search(self, query: str, filters: list = None, fields: list = None,
               page: int = 1, page_size: int = 5) -> dict:
        """兼容旧代码的搜索入口。"""
        return self.meta_search(query, filters, fields, page, page_size)

    def agentic_search(self, query: str, page_size: int = 5) -> dict:
        """调用 agentic-search 端点。"""
        payload = {
            "query": query,
            "page": 1,
            "page_size": page_size,
        }
        return self._call_endpoint("agentic-search", payload, log_label=f"agentic:{query[:40]}")

    def content(self, doc_id: str) -> dict:
        """调用 content 端点。"""
        payload = {"doc_id": doc_id}
        return self._call_endpoint("content", payload, log_label=f"content:{doc_id[:40]}")

    def paper_relations(self, unique_id: str, relation_type: str, page_size: int = 10) -> dict:
        """调用论文关系端点。"""
        payload = {
            "unique_id": unique_id,
            "relation_type": relation_type,
            "page_size": page_size,
        }
        return self._call_endpoint(
            "meta-paper-relations",
            payload,
            log_label=f"relations:{relation_type}:{unique_id[:40]}"
        )

    def test_connection(self, endpoint: str = "meta-search") -> dict:
        """测试指定端点的连接。"""
        if endpoint not in self.BASE_URLS:
            return {
                "error": True,
                "message": f"未知端点: {endpoint}，可用: {list(self.BASE_URLS.keys())}"
            }
        # 用最小请求测试连通性
        if endpoint in ("meta-search", "agentic-search"):
            payload = {"query": "test", "page": 1, "page_size": 1}
        elif endpoint == "content":
            payload = {"doc_id": "test"}
        elif endpoint == "meta-paper-relations":
            payload = {"unique_id": "test", "relation_type": "related", "page_size": 1}
        else:
            payload = {}
        return self._call_endpoint(endpoint, payload, log_label=f"test:{endpoint}")


def test_rate_limit_and_cache():
    """测试限速器和缓存机制。"""
    client = SciverseClient(api_token=CONFIG["sciverse_token"], state=state)
    queries = [
        "graphene battery cycle stability",
        "lithium sulfur battery cathode",
        "solid state electrolyte interface",
        "sodium ion battery anode material",
        "perovskite solar cell degradation",
    ]
    print("=" * 60)
    print("开始测试：连续调用30次，验证限速与缓存")
    print("=" * 60)

    call_count = 0
    overall_start = time.time()

    for query in queries:
        for i in range(6):
            call_count += 1
            t0 = time.time()
            try:
                result = client.meta_search(
                    query=query,
                    filters=[{
                        "field": "publication_published_year",
                        "operator": "FILTER_OP_GTE",
                        "value": 2022
                    }],
                    page=1,
                    page_size=5,
                )
                elapsed = time.time() - t0
                likely_cached = elapsed < 0.05
                print(f"  [{call_count:02d}] query='{query[:30]}...' "
                      f"耗时={elapsed:.3f}s {'(缓存)' if likely_cached else '(API)'}")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{call_count:02d}] query='{query[:30]}...' "
                      f"耗时={elapsed:.3f}s ERROR: {e}")

    total_elapsed = time.time() - overall_start
    print("=" * 60)
    print(f"测试完成！总调用: {call_count}次, 总耗时: {total_elapsed:.2f}秒")
    print(f"详细日志请查看: {SciverseClient.LOG_FILE}")
    print(f"缓存文件请查看: {SciverseClient.CACHE_FILE}")
    print("=" * 60)


sciverse_client = SciverseClient(api_token=CONFIG["sciverse_token"], state=state)


def _extract_papers(result: dict) -> list:
    """
    统一从 Sciverse 各端点响应中提取论文/段落列表。
    兼容 results / data / hits / paragraphs / chunks 等常见 key。
    """
    if not isinstance(result, dict):
        return []
    for key in ("results", "data", "hits", "papers", "paragraphs", "chunks", "items"):
        items = result.get(key)
        if isinstance(items, list):
            return items
    # 如果响应本身就是列表
    if isinstance(result, list):
        return result
    return []


def _normalize_paper(paper: dict) -> dict:
    """统一论文字段名，方便上层工具使用。"""
    if not isinstance(paper, dict):
        return paper
    return {
        "unique_id": paper.get("unique_id") or paper.get("id") or paper.get("_id"),
        "doc_id": paper.get("doc_id"),
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "doi": paper.get("doi", ""),
        "publication_published_year": paper.get("publication_published_year") or paper.get("year"),
        "publication_venue_name_unified": paper.get("publication_venue_name_unified", ""),
        "author": paper.get("author") or paper.get("authors", []),
        "citation_count": paper.get("citation_count") or paper.get("citations_count"),
        "source_payload": paper,
    }


def _meta_search_lookup(identifier: str, page_size: int = 5) -> list:
    """用 DOI/title 等 identifier 反查 unique_id，返回论文列表。"""
    try:
        res = sciverse_client.meta_search(query=identifier, page=1, page_size=page_size)
        if res.get("error"):
            return []
        papers = _extract_papers(res)
        return [_normalize_paper(p) for p in papers]
    except Exception:
        return []


@register_tool(
    description="【强检索】在 Sciverse 学术数据库中搜索研究文献。"
    "适用于需要查找特定领域最新论文、已知关键词的精确搜索、验证假设。"
    "query 应使用专业术语和精确关键词，page_size 默认 5 适合聚焦搜索。"
    "如果需要扩大搜索范围、寻找跨领域关联，请配合 search_weak 使用。"
    "返回文献列表，包含 unique_id、title、abstract、doi、year、venue 等字段。默认获取摘要，不获取全文。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "研究主题或关键词，如 'perovskite solar cell stability'"
            },
            "year_min": {
                "type": "integer",
                "description": "筛选发表年份下限（含），如 2022。不填则不限"
            },
            "page_size": {
                "type": "integer",
                "description": "返回文献条数，默认 5。聚焦搜索建议 5，发散搜索建议 10"
            }
        },
        "required": ["query"]
    }
)
def search_literature(query: str, year_min: int = None, page_size: int = 5):
    """
    在 Sciverse 学术数据库中搜索研究文献（强检索工具）
    - meta-search 默认返回字段已包含 abstract，无需传自定义 fields
    """
    try:
        filters = None
        if year_min:
            filters = [{
                "field": "publication_published_year",
                "operator": "FILTER_OP_GTE",
                "value": year_min
            }]

        result = sciverse_client.meta_search(
            query=query,
            filters=filters,
            page=1,
            page_size=page_size
        )

        if result.get("error"):
            return {
                "status": "error",
                "query": query,
                "error": result.get("message", "未知错误"),
                "api_error": result.get("api_error")
            }

        papers = [_normalize_paper(p) for p in _extract_papers(result)]
        return {
            "status": "success",
            "query": query,
            "year_min": year_min,
            "total_returned": len(papers),
            "papers": papers,
            "raw": result
        }

    except Exception as e:
        return {"status": "error", "query": query, "error": str(e)}


@register_tool(
    description="【弱检索】扩大搜索范围的学术文献检索工具。"
    "当需要扩大搜索范围、寻找跨领域关联、发现被遗漏的相关工作时使用。"
    "与 search_literature 的区别：使用近义词、上位概念、跨领域关键词，返回更多结果。"
    "适用场景：(1) 强检索结果太少 (<3条)；(2) 需要发现跨领域灵感；"
    "(3) 需要找到被不同术语描述的相同概念。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "宽泛关键词，建议使用上位概念或近义词。"
            },
            "year_min": {
                "type": "integer",
                "description": "筛选发表年份下限（含），如 2020。不填则不限"
            },
            "page_size": {
                "type": "integer",
                "description": "返回文献条数，默认 10（弱检索需要更大范围）。最大 20"
            }
        },
        "required": ["query"]
    }
)
def search_weak(query: str, year_min: int = None, page_size: int = 10):
    """弱检索工具 - 使用宽泛/跨领域/近义词关键词扩大搜索范围"""
    page_size = min(page_size, 20)
    try:
        filters = None
        if year_min:
            filters = [{
                "field": "publication_published_year",
                "operator": "FILTER_OP_GTE",
                "value": year_min
            }]

        result = sciverse_client.meta_search(
            query=query,
            filters=filters,
            page=1,
            page_size=page_size
        )

        if result.get("error"):
            return {
                "status": "error",
                "query": query,
                "error": result.get("message", "未知错误"),
                "api_error": result.get("api_error")
            }

        papers = [_normalize_paper(p) for p in _extract_papers(result)]
        return {
            "status": "success",
            "query": query,
            "search_type": "weak",
            "year_min": year_min,
            "total_returned": len(papers),
            "papers": papers,
            "raw": result
        }

    except Exception as e:
        return {"status": "error", "query": query, "search_type": "weak", "error": str(e)}


@register_tool(
    description="【关联搜索】基于 unique_id 追踪论文的引用/被引/相关工作。"
    "当你已知某篇关键论文的 unique_id 时使用；如果只有 DOI 或 title，本工具会先反查 unique_id。"
    "这是真正的引用关系检索，不是关键词模拟。",
    parameters={
        "type": "object",
        "properties": {
            "unique_id": {
                "type": "string",
                "description": "来自 meta-search 的论文 unique_id。优先使用此参数。"
            },
            "doi": {
                "type": "string",
                "description": "论文 DOI（当 unique_id 未知时用于反查）。"
            },
            "title": {
                "type": "string",
                "description": "论文标题（当 unique_id 和 DOI 都未知时用于反查）。"
            },
            "relation_type": {
                "type": "string",
                "enum": ["citations", "references", "related"],
                "description": "关系类型：'citations'=谁引用了这篇论文，'references'=这篇论文引用了谁，'related'=相关工作。"
            },
            "page_size": {
                "type": "integer",
                "description": "返回结果数量，默认 10"
            }
        },
        "required": ["relation_type"]
    }
)
def search_related(unique_id: str = None, doi: str = None, title: str = None,
                   relation_type: str = "related", page_size: int = 10):
    """
    关联搜索工具 - 基于 meta-paper-relations 追踪引用/被引/相关工作
    - 如果未提供 unique_id，先用 DOI 或 title 反查
    """
    # 1. 确定 unique_id
    target_id = unique_id
    if not target_id:
        lookup = doi or title
        if not lookup:
            return {"status": "error", "message": "必须提供 unique_id、doi 或 title 至少一个参数"}
        papers = _meta_search_lookup(lookup, page_size=3)
        if not papers:
            return {"status": "error", "message": f"无法通过 '{lookup}' 找到 unique_id"}
        target_id = papers[0].get("unique_id")
        if not target_id:
            return {"status": "error", "message": "反查成功但结果中缺少 unique_id"}

    # 2. 标准化 relation_type
    relation_type = relation_type.lower().strip()
    allowed = {"citations", "references", "related"}
    if relation_type not in allowed:
        relation_type = "related"

    try:
        result = sciverse_client.paper_relations(
            unique_id=target_id,
            relation_type=relation_type,
            page_size=page_size
        )

        if result.get("error"):
            return {
                "status": "error",
                "unique_id": target_id,
                "relation_type": relation_type,
                "error": result.get("message", "未知错误"),
                "api_error": result.get("api_error")
            }

        papers = [_normalize_paper(p) for p in _extract_papers(result)]
        return {
            "status": "success",
            "unique_id": target_id,
            "relation_type": relation_type,
            "total_returned": len(papers),
            "papers": papers,
            "raw": result
        }

    except Exception as e:
        return {
            "status": "error",
            "unique_id": target_id,
            "relation_type": relation_type,
            "error": str(e)
        }


@register_tool(
    description="【按需内容获取】针对已筛选出的高价值单篇文献，获取其摘要、相关段落或全文。"
    "在文献调研的前几轮（发散/收敛阶段）应该先通过 search_literature/search_weak/search_related 获取批量题录+摘要；"
    "只有当某篇文献对验证假设、填补证据链至关重要时，才调用本工具获取更完整内容。"
    "参数 level 控制获取深度：abstract 仅摘要；paragraphs 返回相关段落（agentic-search）；"
    "fulltext 尝试返回全文（需要该论文有 doc_id）。",
    parameters={
        "type": "object",
        "properties": {
            "unique_id": {
                "type": "string",
                "description": "来自 meta-search 的论文 unique_id。优先使用。"
            },
            "doi": {
                "type": "string",
                "description": "文献 DOI，用于反查 unique_id。"
            },
            "title": {
                "type": "string",
                "description": "文献标题，当 unique_id/DOI 不可用时使用。"
            },
            "level": {
                "type": "string",
                "enum": ["abstract", "paragraphs", "fulltext"],
                "description": "获取深度。'abstract'=摘要（成本低）；'paragraphs'=相关段落；'fulltext'=全文（需要 doc_id，否则降级为 paragraphs）。"
            }
        },
        "required": ["level"]
    }
)
def fetch_paper_content(unique_id: str = None, doi: str = None, title: str = None,
                        level: str = "abstract"):
    """
    按需获取单篇文献的摘要、段落或全文。
    - abstract: 直接返回 meta_search 的 abstract
    - paragraphs: 调用 agentic_search
    - fulltext: 先拿 doc_id，再调 content；无 doc_id 则降级为 paragraphs
    """
    if not unique_id and not doi and not title:
        return {"status": "error", "message": "必须提供 unique_id、doi 或 title 至少一个参数"}

    # 1. 获取论文元数据
    paper = None
    if unique_id:
        papers = _meta_search_lookup(unique_id, page_size=3)
        if papers:
            paper = papers[0]
    if not paper and (doi or title):
        papers = _meta_search_lookup(doi or title, page_size=3)
        if papers:
            paper = papers[0]

    if not paper:
        return {
            "status": "no_results",
            "unique_id": unique_id,
            "doi": doi,
            "title": title,
            "level": level,
            "message": "无法通过给定标识定位到论文。"
        }

    # 2. abstract 级别
    if level == "abstract":
        return {
            "status": "success",
            "level": "abstract",
            "unique_id": paper.get("unique_id"),
            "doc_id": paper.get("doc_id"),
            "title": paper.get("title"),
            "doi": paper.get("doi"),
            "year": paper.get("publication_published_year"),
            "venue": paper.get("publication_venue_name_unified"),
            "abstract": paper.get("abstract", ""),
            "note": "仅返回摘要。"
        }

    # 3. paragraphs 级别
    if level == "paragraphs":
        try:
            res = sciverse_client.agentic_search(
                query=paper.get("title", doi or title),
                page_size=5
            )
            if res.get("error"):
                return {
                    "status": "error",
                    "level": "paragraphs",
                    "unique_id": paper.get("unique_id"),
                    "error": res.get("message", "未知错误"),
                    "api_error": res.get("api_error")
                }
            paragraphs = _extract_papers(res)
            return {
                "status": "success",
                "level": "paragraphs",
                "unique_id": paper.get("unique_id"),
                "doc_id": paper.get("doc_id"),
                "title": paper.get("title"),
                "doi": paper.get("doi"),
                "paragraphs": paragraphs,
                "raw": res
            }
        except Exception as e:
            return {"status": "error", "level": "paragraphs", "error": str(e)}

    # 4. fulltext 级别
    if level == "fulltext":
        doc_id = paper.get("doc_id")
        if doc_id:
            try:
                res = sciverse_client.content(doc_id=doc_id)
                if res.get("error"):
                    # 出错则降级为 paragraphs
                    fallback = fetch_paper_content(
                        unique_id=paper.get("unique_id"),
                        level="paragraphs"
                    )
                    fallback["note"] = f"content 端点失败({res.get('message', '')})，已降级为 paragraphs。"
                    return fallback
                return {
                    "status": "success",
                    "level": "fulltext",
                    "unique_id": paper.get("unique_id"),
                    "doc_id": doc_id,
                    "title": paper.get("title"),
                    "doi": paper.get("doi"),
                    "content": res,
                    "note": "返回全文内容。"
                }
            except Exception as e:
                fallback = fetch_paper_content(
                    unique_id=paper.get("unique_id"),
                    level="paragraphs"
                )
                fallback["note"] = f"content 端点异常({str(e)})，已降级为 paragraphs。"
                return fallback
        else:
            # 无 doc_id，降级为 paragraphs
            fallback = fetch_paper_content(
                unique_id=paper.get("unique_id"),
                level="paragraphs"
            )
            fallback["note"] = "该论文没有 doc_id，已降级为 paragraphs。"
            return fallback

    return {"status": "error", "message": f"未知 level: {level}"}


@register_tool(
    description="【相关性过滤】让 LLM 判断单篇论文与研究问题的相关程度。"
    "返回 topic/method/conclusion 三个维度的评分和综合判断。",
    parameters={
        "type": "object",
        "properties": {
            "paper": {
                "type": "object",
                "description": "来自 meta_search 的单篇论文（包含 title、abstract 等字段）。"
            },
            "research_question": {
                "type": "string",
                "description": "研究问题或假设"
            },
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": '评估维度，如 ["topic", "method", "conclusion"]'
            }
        },
        "required": ["paper", "research_question"]
    }
)
def filter_paper_relevance(paper: dict, research_question: str, criteria: list = None):
    """让 LLM 判断论文与研究问题的相关性。"""
    if not isinstance(paper, dict):
        return {"status": "error", "message": "paper 必须是 dict"}
    criteria = criteria or ["topic", "method", "conclusion"]
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    try:
        prompt = f"""
请判断以下论文与给定研究问题的相关性。

研究问题：{research_question}
论文标题：{title}
论文摘要：{abstract}

请从以下维度打分（0-1）：
1. topic_relevance：论文主题是否与研究问题相关
2. method_relevance：论文方法是否可用于回答研究问题
3. conclusion_relevance：论文结论是否直接支持/反驳研究问题

输出格式（严格 JSON）：
{{
  "is_relevant": true/false,
  "overall_score": 0.0-1.0,
  "topic_score": 0.0-1.0,
  "method_score": 0.0-1.0,
  "conclusion_score": 0.0-1.0,
  "reason": "简要说明"
}}
"""
        resp = llm_client.chat.completions.create(
            model=CONFIG["llm_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = resp.choices[0].message.content
        parsed = _safe_parse_llm_json(content)
        if "parse_error" in parsed:
            return {"status": "parse_failed", "raw": parsed.get("raw", "")}
        return {
            "status": "success",
            "criteria": criteria,
            "is_relevant": parsed.get("is_relevant", False),
            "score": parsed.get("overall_score", 0.0),
            "dimensions": {
                "topic": parsed.get("topic_score", 0.0),
                "method": parsed.get("method_score", 0.0),
                "conclusion": parsed.get("conclusion_score", 0.0)
            },
            "reason": parsed.get("reason", "")
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
