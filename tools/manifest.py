"""Read assets/manifest.js (the curriculum) from Python.

The manifest is a JS file (so pages work from file://), so we evaluate it with node and
return plain Python data. Used by tools/qa.py and tools/build_site.py.
"""
from __future__ import annotations

import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_JS = os.path.join(ROOT, "assets", "manifest.js")


def load() -> dict:
    code = ("global.window = {}; require(process.argv[1]); "
            "process.stdout.write(JSON.stringify(window.LLM_MANIFEST));")
    out = subprocess.run(["node", "-e", code, MANIFEST_JS], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def pages(status: str | None = None) -> list[dict]:
    items = [p for part in load()["parts"] for p in part["pages"]]
    return [p for p in items if status is None or p.get("status") == status]


def written_page_paths() -> list[str]:
    return [os.path.join(ROOT, "pages", p["slug"] + ".html") for p in pages("written")]


def written_test_paths() -> list[str]:
    tests = []
    for p in pages("written"):
        for m in p.get("drills", []):
            t = os.path.join(ROOT, "exercises", "tests", f"test_{m}.py")
            if os.path.exists(t):
                tests.append(t)
    return tests
