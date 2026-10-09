"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_URL = re.compile(r"https?://\S+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    source_numbers = set()
    source_by_n = {}
    seen_urls = set()

    for idx, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"source entry {idx} is not an object")
            continue
        n = entry.get("n")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source entry {idx} n={n!r} must be an integer")
        else:
            if n in source_numbers:
                problems.append(f"duplicate source number [{n}] in sources.json")
            source_numbers.add(n)
            source_by_n[n] = entry

        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] has invalid or missing url: {url!r}")
        else:
            if url in seen_urls:
                problems.append(f"duplicate URL across sources: {url}")
            seen_urls.add(url)

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' section")
        body = report_text
        ref_text = ""
    else:
        last_match = matches[-1]
        body = report_text[:last_match.start()]
        ref_text = report_text[last_match.end():]

    segments = _CODE.split(body)
    cited = set()
    for i, segment in enumerate(segments):
        if i % 2:
            continue
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    for num in sorted(cited):
        if num not in source_numbers:
            problems.append(f"[{num}] cited in body but missing from sources.json")

    for n in sorted(source_numbers):
        if n not in cited:
            problems.append(f"source [{n}] never cited in body")

    ref_lines_by_n = {}
    for line in ref_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _REF_LINE.match(line)
        if m:
            num = int(m.group(1))
            if num in ref_lines_by_n:
                problems.append(f"duplicate reference entry for [{num}] in ## References")
            ref_lines_by_n[num] = line

    for n in sorted(source_numbers):
        if n not in ref_lines_by_n:
            problems.append(f"source [{n}] missing reference line in ## References")
        else:
            line = ref_lines_by_n[n]
            raw_urls = _URL.findall(line)
            cleaned_urls = [u.rstrip(").,;") for u in raw_urls]
            source_url = source_by_n[n].get("url", "")
            if len(cleaned_urls) != 1:
                problems.append(f"reference line [{n}] must contain exactly one URL (found {len(cleaned_urls)})")
            elif cleaned_urls[0] != source_url and raw_urls[0] != source_url:
                problems.append(f"reference line [{n}] URL does not match source URL '{source_url}'")

    for num in sorted(ref_lines_by_n.keys()):
        if num not in source_numbers:
            problems.append(f"reference [{num}] in ## References does not exist in sources.json")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
