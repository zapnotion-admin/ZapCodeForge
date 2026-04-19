"""
engine/apply_changes.py
Parses AI output and writes files to disk safely.

Two-tier parsing strategy:
  1. Structured  — FILE: path\n```\ncode\n```  (preferred, explicit filename)
  2. Fallback    — any ```lang\ncode\n``` block, filename inferred from task

The fallback exists because local models don't reliably follow output format
instructions even when explicitly asked.  Rather than silently doing nothing,
we infer a sensible filename from the language tag and task description.

Rules enforced:
  - All writes sandboxed inside base_dir (path escape blocked)
  - Directories created automatically
  - Returns list of written paths for UI confirmation
  - Never raises on parse failure — returns empty list
"""

import os
import re
from engine.logger import log

_LANG_EXT = {
    "python": ".py",  "py":    ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "html":  ".html", "css":  ".css",
    "bash":  ".sh",   "sh":   ".sh",
    "json":  ".json", "yaml": ".yaml", "yml": ".yml",
    "sql":   ".sql",  "rust": ".rs",
    "cpp":   ".cpp",  "c":    ".c",
    "java":  ".java", "go":   ".go",
    "ruby":  ".rb",   "php":  ".php",
}


def _infer_filename(task: str, lang: str) -> str:
    ext = _LANG_EXT.get(lang.lower(), f".{lang.lower()}" if lang else ".txt")

    # First priority: explicit filename in the task ("save it as calculator.py",
    # "save as foo.py", "call it bar.py", "name it baz.js")
    explicit = re.search(
        r"(?:save (?:it )?as|save to|call it|name it|file(?:name)? (?:is )?)\s+([\w\-]+\.\w+)",
        task, re.IGNORECASE
    )
    if explicit:
        filename = explicit.group(1).strip()
        log(f"[apply_changes] Explicit filename from task: {filename}")
        return filename

    # Also catch bare "calculator.py" style mentions anywhere in the task
    named = re.search(r"([\w\-]+\.(?:py|js|ts|html|css|sh|json|yaml|sql|rs|cpp|c|java|go|rb|php))", task)
    if named:
        filename = named.group(1).strip()
        log(f"[apply_changes] Named file found in task: {filename}")
        return filename

    # Fallback: derive stem from meaningful words in the task
    stop = {"a", "an", "the", "simple", "basic", "write", "create", "make",
            "build", "generate", "with", "using", "for", "in", "and", "or",
            "script", "save", "as", "it", "me", "slightly", "more", "complex",
            "gui", "app", "tool", "program", "code", "file", "python", "js"}
    words = re.findall(r"[a-zA-Z]+", task.lower())
    meaningful = [w for w in words if w not in stop]
    stem = meaningful[0] if meaningful else "output"
    stem = re.sub(r"[^a-z0-9_]", "_", stem)
    filename = f"{stem}{ext}"
    log(f"[apply_changes] Inferred filename: {filename}")
    return filename


def extract_files(text: str, task: str = "") -> list:
    files = []

    # Tier 1: explicit FILE: blocks
    structured = re.findall(r"FILE:\s*([^\n]+)\n```[^\n]*\n(.*?)```", text, re.DOTALL)
    for raw_path, code in structured:
        path = raw_path.strip().replace("\\", "/")
        if path:
            files.append({"path": path, "code": code.rstrip()})
            log(f"[apply_changes] Structured: {path}")

    if files:
        return files

    # Tier 2: fallback — any fenced code block
    log("[apply_changes] No FILE: blocks — trying fallback")
    blocks = re.findall(r"```([^\n]*)\n(.*?)```", text, re.DOTALL)
    seen = set()
    for lang, code in blocks:
        lang = lang.strip().lower()
        code = code.rstrip()
        if code.count("\n") < 2:  # skip tiny inline snippets
            continue
        path = _infer_filename(task, lang)
        if path in seen:
            base, ext = os.path.splitext(path)
            i = 2
            while f"{base}_{i}{ext}" in seen:
                i += 1
            path = f"{base}_{i}{ext}"
        seen.add(path)
        files.append({"path": path, "code": code})
        log(f"[apply_changes] Fallback: {path}")

    if not files:
        log("[apply_changes] No code blocks found")
    return files


def write_files(files: list, base_dir: str) -> list:
    base_abs = os.path.abspath(base_dir)
    written = []
    for f in files:
        rel_path = f["path"]
        code     = f["code"]
        full_path = os.path.abspath(os.path.join(base_abs, rel_path))
        if not full_path.startswith(base_abs + os.sep) and full_path != base_abs:
            log(f"[apply_changes] BLOCKED escape: {rel_path}")
            continue
        try:
            parent = os.path.dirname(full_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as fh:
                fh.write(code)
            written.append(full_path)
            log(f"[apply_changes] Wrote: {full_path}")
        except Exception as e:
            log(f"[apply_changes] Failed {full_path}: {e}")
    return written
