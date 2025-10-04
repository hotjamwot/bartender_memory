"""Simple CLI shim that queries the running bartender /recall endpoint.

Usage:
    python scripts/shim_cli.py "what's my preference"

It prints JSON results to stdout for easy piping into other tools.
"""
import sys
import json
import urllib.parse
import urllib.request

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/shim_cli.py 'query'")
        sys.exit(2)
    q = sys.argv[1]
    params = urllib.parse.urlencode({"q": q, "limit": 3})
    url = f"http://127.0.0.1:5001/recall?{params}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read().decode("utf-8")
            print(data)
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == '__main__':
    main()
