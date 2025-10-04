"""Shim CLI that can either query `/recall` or post `REMEMBER:` lines to `/remember`.

Usage:
  # recall example
  python scripts/shim_cli.py "dark mode"

  # remember example (single-line)
  python scripts/shim_cli.py "REMEMBER: I like dark mode in VSCode"

  # remember with priority
  python scripts/shim_cli.py "REMEMBER[high]: My passport is in the drawer"

It also accepts stdin for multi-line input. Lines beginning with REMEMBER will be POSTed.
"""
import sys
import json
import re
import urllib.parse
import urllib.request

DEFAULT_HOST = "http://127.0.0.1:5001"

REMEMBER_RE = re.compile(r"^REMEMBER(?:\[(?P<prio>[^\]]+)\])?\s*:\s*(?P<text>.+)$", flags=re.IGNORECASE)


def post_remember(text: str, priority: str | None = None):
    url = f"{DEFAULT_HOST}/remember"
    payload = {"text": text}
    tags = []
    if priority:
        tags.append(f"priority:{priority.lower()}")
    if tags:
        payload["tags"] = tags
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.load(resp)


def recall_query(q: str, limit: int = 3):
    params = urllib.parse.urlencode({"q": q, "limit": limit})
    url = f"{DEFAULT_HOST}/recall?{params}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.load(resp)


def main():
    # Read input from arg or stdin
    raw = None
    if len(sys.argv) >= 2:
        raw = " ".join(sys.argv[1:]).strip()
    else:
        raw = sys.stdin.read().strip()

    # If the input contains REMEMBER: lines, parse and post them
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    remember_lines = []
    for line in lines:
        m = REMEMBER_RE.match(line)
        if m:
            remember_lines.append((m.group("text"), m.group("prio")))

    if remember_lines:
        results = []
        for text, prio in remember_lines:
            try:
                res = post_remember(text, prio)
            except Exception as e:
                res = {"error": str(e), "text": text}
            results.append(res)
        print(json.dumps({"remembered": results}, indent=2))
        return

    # Otherwise treat the input as a recall query
    if not raw:
        print(json.dumps({"error": "no input provided"}))
        sys.exit(2)
    try:
        out = recall_query(raw)
        print(json.dumps(out, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == '__main__':
    main()
