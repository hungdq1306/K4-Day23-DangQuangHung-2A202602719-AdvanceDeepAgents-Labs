"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_LAST_ARXIV_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _redact_exa_key(text: str) -> str:
    key = os.getenv("EXA_API_KEY")
    if key and key in text:
        text = text.replace(key, "[REDACTED]")
    return text


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.HTTPStatusError, httpx.TransportError) as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code not in (429, 500, 502, 503, 504):
                    raise
            if attempt == attempts - 1:
                raise

            retry_after = None
            if isinstance(exc, RetryableError) and exc.retry_after is not None:
                retry_after = exc.retry_after
            elif isinstance(exc, httpx.HTTPStatusError):
                header = exc.response.headers.get("Retry-After")
                if header:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        pass

            if retry_after is not None:
                delay = min(float(retry_after), cap)
            else:
                raw_delay = base * (2**attempt)
                jitter = random.uniform(0.0, 0.5 * raw_delay)
                delay = min(raw_delay + jitter, cap)

            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        terms = re.findall(r"[\w\-]+", query)
        if not terms:
            return "NO RESULTS"

        elapsed = time.time() - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        _LAST_ARXIV_CALL = time.time()

        clamped_results = max(1, min(max_results, 8))
        search_query = " AND ".join(f"all:{t}" for t in terms)
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_results,
            "start": 0,
        }

        def _call():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(ARXIV_URL, params=params)
                resp.raise_for_status()
                return resp.text

        text = with_retry(_call, attempts=5, base=2.0, cap=60.0)
        root = xml.etree.ElementTree.fromstring(text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", ns):
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or not id_elem.text:
                continue
            raw_id = id_elem.text.strip().split("/abs/")[-1]
            paper_id = re.sub(r"v\d+$", "", raw_id)
            url = f"https://arxiv.org/abs/{paper_id}"

            pub_elem = entry.find("atom:published", ns)
            published = pub_elem.text.strip()[:10] if pub_elem is not None and pub_elem.text else ""

            title_elem = entry.find("atom:title", ns)
            title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else ""

            sum_elem = entry.find("atom:summary", ns)
            summary = " ".join(sum_elem.text.split())[:400] if sum_elem is not None and sum_elem.text else ""

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {str(exc)}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(limit, 10))
        params = {"limit": clamped_limit}
        if date:
            params["date"] = date

        def _call():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_DAILY_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(_call, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            paper = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("id")
            if not paper_id:
                continue
            title = paper.get("title") or item.get("title", "")
            summary = (paper.get("summary") or item.get("summary", ""))[:400]
            published = (paper.get("publishedAt") or item.get("publishedAt", ""))[:10]
            upvotes = paper.get("upvotes", 0) or 0
            github = paper.get("githubRepo") or ""
            stars = paper.get("githubStars", 0) or 0
            url = f"https://huggingface.co/papers/{paper_id}"

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if keyword:
            kw = keyword.lower()
            records = [r for r in records if kw in r["title"].lower() or kw in r["summary"].lower()]

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records[:8], ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {str(exc)}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        clamped_limit = max(1, min(limit, 8))
        params = {"q": query, "limit": clamped_limit}

        def _call():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(_call, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            paper = item.get("paper") if isinstance(item, dict) and isinstance(item.get("paper"), dict) else item
            if not isinstance(paper, dict):
                continue
            paper_id = paper.get("id")
            if not paper_id:
                continue
            title = paper.get("title") or item.get("title", "")
            summary = (paper.get("ai_summary") or paper.get("summary") or item.get("summary", ""))[:400]
            published = (paper.get("publishedAt") or item.get("publishedAt", ""))[:10]
            upvotes = paper.get("upvotes", 0) or 0
            github = paper.get("githubRepo") or ""
            stars = paper.get("githubStars", 0) or 0
            url = f"https://huggingface.co/papers/{paper_id}"

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {str(exc)}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    key = os.getenv("EXA_API_KEY")
    endpoint = EXA_URL + (f"?exaApiKey={key}" if key else "")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _call():
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            text = resp.text

            data = None
            for line in text.splitlines():
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    if raw:
                        try:
                            data = json.loads(raw)
                            break
                        except json.JSONDecodeError:
                            pass
            if data is None:
                try:
                    data = resp.json()
                except Exception:
                    raise RetryableError("Invalid response format from Exa MCP")

            if "error" in data:
                err_msg = str(data["error"])
                if "rate limit" in err_msg.lower() or "too many requests" in err_msg.lower():
                    raise RetryableError(f"Exa rate limit: {err_msg}")
                raise RuntimeError(err_msg)

            result = data.get("result", {})
            meta = result.get("_meta", {})
            meta_str = json.dumps(meta).lower()
            if "ratelimit" in meta_str or "rate_limit" in meta_str:
                raise RetryableError("Exa rate limited per _meta")

            content = result.get("content", [])
            output_texts = []
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    t = c.get("text", "")
                    if "rate limit" in t.lower() and len(t) < 300:
                        raise RetryableError(f"Exa rate limited: {t}")
                    output_texts.append(t)

            return "\n\n".join(output_texts).strip()

    return with_retry(_call, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        obj = objective or f"Find high-quality research papers, survey articles, and project documentation about: {query}"
        results_text = _call_exa_mcp("web_search_exa", {
            "query": query,
            "objective": obj,
            "numResults": max(1, min(num_results, 4)),
        })
        if not results_text:
            return "NO RESULTS"
        return results_text
    except Exception as exc:
        return _redact_exa_key(f"ERROR: {type(exc).__name__}: {str(exc)}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        results_text = _call_exa_mcp("web_fetch_exa", {"urls": [url]})
        if not results_text:
            return "NO RESULTS"
        return results_text[:5000]
    except Exception as exc:
        return _redact_exa_key(f"ERROR: {type(exc).__name__}: {str(exc)}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
