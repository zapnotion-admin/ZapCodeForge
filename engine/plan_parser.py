"""
engine/plan_parser.py
Parses structured step plans from the REASON stage output.

Expected format from the model:
    STEP 1: Create calculator.py with window and entry widget
    FILES: calculator.py
    DEPENDS_ON: none

    STEP 2: Add button grid
    FILES: calculator.py
    DEPENDS_ON: STEP 1

Falls back gracefully if the model doesn't follow the format exactly.
"""

import re
from engine.logger import log


def parse_steps(plan_text: str) -> list:
    """
    Parses REASON output into a list of step dicts.

    Returns:
        List of dicts with keys: number, description, files, depends_on
        Returns [] if no structured steps found (caller should fall back).
    """
    steps = []

    # Try structured STEP N: format first
    # Matches: STEP 1: description (then optional FILES: and DEPENDS_ON: lines)
    step_pattern = re.compile(
        r"STEP\s+(\d+)\s*:\s*(.+?)(?=STEP\s+\d+\s*:|$)",
        re.IGNORECASE | re.DOTALL
    )

    for match in step_pattern.finditer(plan_text):
        number = int(match.group(1))
        block  = match.group(2).strip()

        # Extract description (first line of block)
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        description = lines[0] if lines else f"Step {number}"

        # Extract FILES:
        files = []
        files_match = re.search(r"FILES?\s*:\s*(.+)", block, re.IGNORECASE)
        if files_match:
            raw = files_match.group(1).strip()
            files = [f.strip() for f in re.split(r"[,\s]+", raw) if f.strip() and f.lower() != "none"]

        # Extract DEPENDS_ON:
        depends_on = []
        dep_match = re.search(r"DEPENDS_ON\s*:\s*(.+)", block, re.IGNORECASE)
        if dep_match:
            raw = dep_match.group(1).strip()
            if raw.lower() != "none":
                nums = re.findall(r"\d+", raw)
                depends_on = [int(n) for n in nums]

        steps.append({
            "number":      number,
            "description": description,
            "files":       files,
            "depends_on":  depends_on,
            "status":      "pending",
        })

    if steps:
        log(f"[plan_parser] Parsed {len(steps)} structured steps")
        return sorted(steps, key=lambda s: s["number"])

    # Fallback: try numbered list  (1. do something / 1) do something)
    list_pattern = re.compile(r"^\s*(\d+)[.)]\s+(.+)$", re.MULTILINE)
    for match in list_pattern.finditer(plan_text):
        number = int(match.group(1))
        description = match.group(2).strip()
        steps.append({
            "number":      number,
            "description": description,
            "files":       [],   # unknown — executor will use all context files
            "depends_on":  [number - 1] if number > 1 else [],
            "status":      "pending",
        })

    if steps:
        log(f"[plan_parser] Parsed {len(steps)} list steps (fallback)")
        return sorted(steps, key=lambda s: s["number"])

    log("[plan_parser] No structured steps found — caller should use single-step fallback")
    return []


def extract_plan_summary(steps: list) -> str:
    """
    Produces a compact summary string of the full plan.
    Used in step prompts so the model knows the overall goal.
    """
    if not steps:
        return "(no structured plan)"
    lines = [f"Step {s['number']}: {s['description']}" for s in steps]
    return "\n".join(lines)


def steps_to_status_summary(steps: list) -> str:
    """
    Returns a compact string showing which steps are done vs pending.
    Used for context compression between steps.
    """
    completed = [s for s in steps if s["status"] == "complete"]
    failed    = [s for s in steps if s["status"] == "failed"]
    pending   = [s for s in steps if s["status"] == "pending"]
    running   = [s for s in steps if s["status"] == "running"]

    parts = []
    if completed:
        nums = [str(s["number"]) for s in completed]
        descs = "; ".join(s["description"][:40] for s in completed)
        parts.append(f"COMPLETED steps {', '.join(nums)}: {descs}")
    if running:
        s = running[0]
        parts.append(f"RUNNING step {s['number']}: {s['description']}")
    if failed:
        nums = [str(s["number"]) for s in failed]
        parts.append(f"FAILED steps {', '.join(nums)} (skipped after retry)")
    if pending:
        nums = [str(s["number"]) for s in pending]
        parts.append(f"PENDING steps {', '.join(nums)}")
    return "\n".join(parts)
