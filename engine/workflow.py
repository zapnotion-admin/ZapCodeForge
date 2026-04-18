"""
engine/workflow.py
3-stage pipeline with model specialisation:

  Stage 1 (Scan)   — qwen3-coder     Fast file analysis, issue identification
  Stage 2 (Reason) — deepseek-reasoner  Deep reasoning, diagnosis, planning
  Stage 3 (Code)   — qwen3-coder     Code execution from the reasoned plan
  Stage 4 (Review) — qwen3-coder     Final review (same model, already warm)
  Stage 5 (Retry)  — qwen3-coder     Only if reviewer flags issues

VRAM discipline:
  - Only one model loaded at a time.
  - explicit unload_model() call before every model swap.
  - ensure_model() skips warmup if the model is already hot.
  - cancel_check honoured before every stage transition.
"""

import re
from engine.ollama_client import single_response, ensure_model, unload_model, resolve_model
from engine.logger import log
from core.config import (
    MODEL_CODER, MODEL_REASONER, MODEL_FALLBACK,
    MAX_CTX_CODER, MAX_CTX_REASONER,
)

CONSTRAINTS = """
CONSTRAINTS (follow always):
- Make minimal changes. Only modify what the task explicitly requires.
- Do not introduce new dependencies unless asked.
- Follow the coding style visible in the provided files.
- Do not modify files outside the stated scope.
"""


def extract_verdict(text: str) -> str:
    m = re.search(
        r"VERDICT[\s:]+\*{0,2}(PASS|NEEDS_CHANGES|FAIL)\*{0,2}",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    upper = text.upper()
    if "NEEDS_CHANGES" in upper or "NEEDS CHANGES" in upper:
        return "NEEDS_CHANGES"
    if "FAIL" in upper:
        return "FAIL"
    if "PASS" in upper:
        return "PASS"
    return "NEEDS_CHANGES"


def extract_code_blocks(text: str) -> list[str]:
    return re.findall(r"```.*?\n(.*?)```", text, re.DOTALL)


def _swap_to(model: str, display_name: str, emit, cancel_check) -> bool:
    """
    Unload current model from VRAM then warm up the target model.
    Returns False if cancelled before the swap completes.
    """
    if cancel_check():
        return False
    log(f"[pipeline] VRAM swap → {model}")
    unload_model()          # evict whatever is currently in VRAM
    ensure_model(model)     # load the next model (skipped if already warm)
    return not cancel_check()


def run_pipeline(
    task: str,
    file_context: str,
    progress_callback=None,
    cancel_check=None,
) -> dict:
    """
    Executes the 3-model pipeline:
      Scan (Qwen) → Reason (DeepSeek) → Code (Qwen) → Review (Qwen)

    progress_callback(stage, text) stages:
      status, scan_done, plan_done, code_done, review_done, retry_done

    Returns {} if cancelled at any point before completion.
    """

    def emit(stage: str, text: str) -> None:
        log(f"[pipeline] stage={stage} len={len(text)}")
        if progress_callback:
            progress_callback(stage, text)

    def is_cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    coder    = resolve_model(MODEL_CODER,    MODEL_FALLBACK)
    reasoner = resolve_model(MODEL_REASONER, MODEL_FALLBACK)

    files_section = "FILES:\n" + file_context if file_context else "No files provided."

    # ──────────────────────────────────────────────────────────────────
    # STAGE 1: SCAN — Qwen3 reads the files, identifies issues
    # ──────────────────────────────────────────────────────────────────
    emit("status", "🔍 Scanning files...")
    ensure_model(coder)
    if is_cancelled():
        return {}

    scan_prompt = f"""
TASK: {task}
{files_section}

You are a code scanner. Do NOT write any fixes yet.

Identify:
1. Which functions / classes are directly relevant to this task
2. Any existing bugs, risks, or edge cases in those areas
3. Dependencies between components that the task will affect

Output structured findings:
- File: <name>
- Function/Class: <name>
- Issue: <description>
- Relevant lines: <brief quote or line range>

Be precise and concise. A reasoning model will use your findings to form a plan.
""".strip()

    scan = single_response(coder, scan_prompt, num_ctx=MAX_CTX_CODER)
    emit("scan_done", scan)

    # ──────────────────────────────────────────────────────────────────
    # STAGE 2: REASON — DeepSeek-R1 reasons over the findings → plan
    # VRAM swap: unload Qwen, load DeepSeek
    # ──────────────────────────────────────────────────────────────────
    emit("status", "🧠 Reasoning...")
    if not _swap_to(reasoner, "DeepSeek-R1", emit, is_cancelled):
        return {}

    reason_prompt = f"""
TASK: {task}
{CONSTRAINTS}

SCAN FINDINGS (from static analysis):
{scan}

{files_section}

You are a reasoning model. Using the scan findings:
1. Diagnose the root cause of any issues
2. Identify the safest, minimal solution
3. Produce a numbered execution plan — each step: file, change, reason
4. Flag any risks or ordering dependencies

Be thorough. A coding model will execute your plan literally — ambiguity causes bugs.
""".strip()

    plan = single_response(reasoner, reason_prompt, num_ctx=MAX_CTX_REASONER)
    emit("plan_done", plan)

    # ──────────────────────────────────────────────────────────────────
    # STAGE 3: CODE — Qwen3 executes the reasoned plan
    # VRAM swap: unload DeepSeek, load Qwen
    # ──────────────────────────────────────────────────────────────────
    emit("status", "⚙️ Writing code...")
    if not _swap_to(coder, "Qwen3-Coder", emit, is_cancelled):
        return {}

    code_prompt = f"""
TASK: {task}
{CONSTRAINTS}
EXECUTION PLAN:
{plan}

{files_section}

Execute the plan exactly. For each changed file:
1. State the filename
2. Show the complete modified file (or a clearly marked diff block)
3. One-line explanation of what changed

Write production-quality code. Do not deviate from the plan.
""".strip()

    code = single_response(coder, code_prompt, num_ctx=MAX_CTX_CODER)
    emit("code_done", code)

    # ──────────────────────────────────────────────────────────────────
    # STAGE 4: REVIEW — Qwen3 reviews (already warm, no swap needed)
    # ──────────────────────────────────────────────────────────────────
    if is_cancelled():
        return {}

    emit("status", "✅ Reviewing...")

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN:
{plan}
CODE OUTPUT:
{code}

Check: correctness, logic errors, style consistency, missing error handling, security issues.

Respond in this exact format:
VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    final_code = code

    # ──────────────────────────────────────────────────────────────────
    # STAGE 5: RETRY — only if reviewer flagged issues (Qwen already warm)
    # ──────────────────────────────────────────────────────────────────
    if verdict in ("NEEDS_CHANGES", "FAIL"):
        if is_cancelled():
            return {}

        emit("status", "🔁 Fixing reviewer issues...")

        retry_prompt = f"""
The previous code attempt had issues. Fix ONLY what was flagged.

ORIGINAL TASK: {task}
REVIEWER FEEDBACK:
{review}
ORIGINAL PLAN:
{plan}
PREVIOUS CODE (fix ONLY the flagged issues):
{code}
""".strip()

        final_code = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER)
        emit("retry_done", final_code)

    log(f"[pipeline] completed. verdict={verdict}")
    return {
        "scan":       scan,
        "plan":       plan,
        "code":       code,
        "review":     review,
        "verdict":    verdict,
        "final_code": final_code,
    }
