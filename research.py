"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator

# Intercept Google rate limits so 1-minute quota windows are waited out gracefully
try:
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAI, GoogleRateLimitError

    _orig_generate = ChatGoogleGenerativeAI._generate

    def _robust_generate(self, *args, **kwargs):
        for attempt in range(8):
            try:
                return _orig_generate(self, *args, **kwargs)
            except GoogleRateLimitError as exc:
                if attempt == 7:
                    raise
                delay = 50.0
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
                if match:
                    delay = float(match.group(1)) + 3.0
                print(f"[RateLimit] Quota limit hit, waiting {delay:.1f}s for quota window to reset...", file=sys.stderr)
                time.sleep(delay)
        return _orig_generate(self, *args, **kwargs)

    ChatGoogleGenerativeAI._generate = _robust_generate
except ImportError:
    pass


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    t = (topic or "").strip().lower()
    slug = re.sub(r"[^\w]+", "-", t).strip("-")
    if not slug:
        slug = "topic"
    return slug[:60].rstrip("-")


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return f"""Please conduct a rigorous, comprehensive deep research survey on the following topic:
"{topic}"

Follow your mandatory workflow step by step:
1. Call `write_todos` to plan and break down the topic into 3-4 distinct sub-questions (foundations, methods/architectures, recent advances/benchmarks).
2. Use the `task` tool to delegate each sub-question to the `researcher` subagent (at least 3 subagent calls total).
   MANDATORY: You must assign:
   - Task 1 to use `arxiv_search` to find arXiv papers (source: arxiv)
   - Task 2 to use `hf_search_papers` / `hf_daily_papers` to find Hugging Face papers (source: hf-search or hf-daily)
   - Task 3 to use `web_search` / `web_fetch` to find web surveys/articles (source: web)
   This guarantees that the combined sources span at least 3 distinct source families.
3. Inspect notes in /tmp/work/research/notes/ and ensure that at least 3 source families (arxiv, hf-search/hf-daily, web) are represented in sources.json.
4. Merge all sources into /tmp/work/research/sources.json following the schema [{{ "n": 1, "id": "...", "url": "...", "title": "...", "date": "YYYY-MM-DD", "source": "..." }}].
5. Write the comprehensive survey body to /tmp/work/report/report.md following REPORT_TEMPLATE.md. Do NOT include a `## References` section in your writeup.
6. Execute `python3 /tmp/work/research/finalize_citations.py` using `execute` to automatically generate `## References` and sync sources.json.
7. Execute `python3 /tmp/work/research/check_citations.py` using `execute` until it prints "OK".
8. Have `citation-checker` verify key claims.
"""


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    tool_counts = Counter()
    total_input_tokens = 0
    total_output_tokens = 0

    for msg in messages:
        t_calls = getattr(msg, "tool_calls", None)
        if not t_calls and isinstance(msg, dict):
            t_calls = msg.get("tool_calls")
        if t_calls:
            for tc in t_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", str(tc))
                if name:
                    tool_counts[name] += 1

        usage = getattr(msg, "usage_metadata", None)
        if not usage and isinstance(msg, dict):
            usage = msg.get("usage_metadata")
        if usage and isinstance(usage, dict):
            total_input_tokens += int(usage.get("input_tokens") or 0)
            total_output_tokens += int(usage.get("output_tokens") or 0)

    subagent_calls = tool_counts.get("task", 0)
    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("Report file is missing or empty in sandbox")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("sources.json file is missing or empty in sandbox")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
        if not isinstance(sources_data, list):
            raise ValueError("sources.json must be a list")
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}")

    slug = slugify(topic)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    source_families = sorted(list({
        str(s.get("source")) for s in sources_data if isinstance(s, dict) and s.get("source")
    }))

    meta = summarize(messages, elapsed, model_name)
    meta["topic"] = topic
    meta["n_sources"] = len(sources_data)
    meta["source_families"] = source_families

    out_md = reports_path / f"{slug}.md"
    out_sources = reports_path / f"{slug}.sources.json"
    out_meta = reports_path / f"{slug}.meta.json"

    out_md.write_bytes(report_bytes)
    out_sources.write_bytes(sources_bytes)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_md


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic = (topic or "").strip()
    if not topic:
        print("Usage: python research.py '<topic>'", file=sys.stderr)
        return 2

    model = make_model()
    model_name = os.getenv("LAB_MODEL") or getattr(model, "model", None) or getattr(model, "model_name", "unknown")
    start = time.monotonic()

    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })

        agent = build_lead_agent(backend, model)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": build_prompt(topic)}]},
            config={"recursion_limit": 1000},
        )
        elapsed = time.monotonic() - start
        messages = result.get("messages", []) if isinstance(result, dict) else []

        try:
            saved_report = save_outputs(backend, topic, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {saved_report}")
            return 0
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
