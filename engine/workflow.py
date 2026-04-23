"""
engine/workflow.py
Agent pipeline: SCAN → REASON → DECOMPOSE → [STEP LOOP] → FINAL REVIEW

Stage model assignments:
  Scan     — qwen3-coder      Fast file analysis + cross-file consistency
  Reason   — deepseek-reasoner  Structured step plan with dependencies
  Execute  — qwen3-coder      One step at a time, re-reads files between steps
  Review   — qwen3-coder      Final review of all written files
  Retry    — qwen3-coder      Only if reviewer flags issues (single-shot fallback)

VRAM discipline:
  - Only one model in VRAM at a time
  - Explicit unload_model() before every swap
  - cancel_check honoured before every stage transition
"""

import re
import os
from engine.ollama_client import single_response, ensure_model, unload_model, resolve_model
from engine.logger import log
from engine.brief import read_brief, format_brief_for_prompt
from engine.plan_parser import parse_steps, extract_plan_summary
from engine.step_state import StepState
from engine.step_executor import run_steps
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


def extract_code_blocks(text: str) -> list:
    return re.findall(r"```.*?\n(.*?)```", text, re.DOTALL)


def _swap_to(model: str, emit, cancel_check) -> bool:
    if cancel_check():
        return False
    log(f"[pipeline] VRAM swap → {model}")
    unload_model()
    ensure_model(model)
    return not cancel_check()


def run_pipeline(
    task:              str,
    file_context:      str,
    project_dir:       str = "",
    context_files:     list = None,   # list of file paths user has in context
    coder_model:       str = None,    # override coder model (None = use config default)
    reasoner_model:    str = None,    # override reasoner model (None = use config default)
    progress_callback  = None,
    cancel_check:      object = None,
) -> dict:
    """
    Full agent pipeline.

    progress_callback(stage, text) stages:
      status, scan_done, plan_done, step_start, step_done, step_failed,
      code_done, review_done, retry_done

    Returns dict with keys: scan, plan, steps, review, verdict,
                             final_code, completed_files, diffs
    """

    def emit(stage: str, text: str) -> None:
        log(f"[pipeline] stage={stage} len={len(text)}")
        if progress_callback:
            progress_callback(stage, text)

    def is_cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    coder    = resolve_model(coder_model    or MODEL_CODER,    MODEL_FALLBACK)
    reasoner = resolve_model(reasoner_model or MODEL_REASONER, MODEL_FALLBACK)
    log(f"[pipeline] coder={coder}  reasoner={reasoner}")

    files_section = "FILES:\n" + file_context if file_context else "No files provided."
    brief_content = read_brief(project_dir) if project_dir else ""
    brief_section = format_brief_for_prompt(brief_content)

    # Extract explicit target filename from task if specified
    _explicit = re.search(
        r"(?:save (?:it )?as|save to|call it|name it|file(?:name)? (?:is )?)\s+([\w\-]+\.\w+)",
        task, re.IGNORECASE
    ) or re.search(
        r"\b([\w\-]+\.(?:py|js|ts|html|css|sh|json|rs|cpp|c|java|go|rb|php))\b", task
    )
    target_file = _explicit.group(1).strip() if _explicit else None
    target_instruction = (
        f"\nIMPORTANT: The output file MUST be named '{target_file}'. "
        f"Use 'FILE: {target_file}' in your output. Do NOT use any other filename."
    ) if target_file else ""

    # ──────────────────────────────────────────────────────────────────
    # STAGE 1: SCAN (skipped if no files to scan)
    # ──────────────────────────────────────────────────────────────────
    has_files = bool(file_context and file_context.strip() and file_context != "No files provided.")

    if has_files:
        emit("status", "🔍 Scanning files...")
        ensure_model(coder)
        if is_cancelled():
            return {}

        scan_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}{files_section}

You are a code scanner reviewing EXISTING files listed above.
Do NOT invent issues for files that do not exist yet.
Only report issues you can see in the actual code shown.
Be concise — maximum 15 lines total.

For each real issue found:
- File: <filename> | Issue: <one line description>
- Mark cross-file issues [CROSS-FILE]

If the files look correct for the task, output: "No issues found."
""".strip()

        scan = single_response(coder, scan_prompt, num_ctx=MAX_CTX_CODER)
        emit("scan_done", scan)
    else:
        # No files — skip scan entirely, no model call
        scan = "(No existing files — creating from scratch)"
        emit("status", "⏭ No files to scan — proceeding to plan...")

    # ──────────────────────────────────────────────────────────────────
    # STAGE 2: REASON — produces structured step plan
    # ──────────────────────────────────────────────────────────────────
    emit("status", "🧠 Reasoning...")
    if not _swap_to(reasoner, emit, is_cancelled):
        return {}

    reason_prompt = f"""
TASK: {task}{target_instruction}
{brief_section}
SCAN FINDINGS: {scan}

Break this task into 4-8 small steps. Each step = one logical unit (20-50 lines max).

Use EXACTLY this format — nothing else:

STEP 1: <one clear sentence>
FILES: <filename>
DEPENDS_ON: none

STEP 2: <one clear sentence>
FILES: <filename>
DEPENDS_ON: STEP 1

Rules:
- New project: step 1 creates skeleton, later steps add features one at a time
- Modifications: step 1 fixes most critical issue first
- Each step must be independently testable
- Do not include explanations outside the STEP blocks
""".strip()

    plan = single_response(reasoner, reason_prompt, num_ctx=MAX_CTX_REASONER)
    emit("plan_done", plan)

    # ──────────────────────────────────────────────────────────────────
    # STAGE 3: DECOMPOSE — parse steps, no model call
    # ──────────────────────────────────────────────────────────────────
    steps = parse_steps(plan)

    # Swap back to coder for execution
    if not _swap_to(coder, emit, is_cancelled):
        return {}

    if not steps:
        # Fallback: single-step execution (old behaviour)
        log("[pipeline] No structured steps — falling back to single-step")
        emit("status", "⚙️ Writing code (single pass)...")
        return _single_step_fallback(
            task, plan, file_context, brief_section,
            target_instruction, coder, emit, is_cancelled,
            project_dir, context_files
        )

    emit("status", f"📋 Plan: {len(steps)} steps")

    # ──────────────────────────────────────────────────────────────────
    # STAGE 4: STEP EXECUTION LOOP
    # ──────────────────────────────────────────────────────────────────
    state = StepState(
        project_dir=project_dir,
        task=task,
        steps=steps,
    )

    loop_result = run_steps(
        state             = state,
        task              = task,
        all_context_files = context_files or [],
        project_dir       = project_dir,
        brief_content     = brief_content,
        coder_model       = coder,
        progress_callback = progress_callback,
        cancel_check      = cancel_check,
    )

    if is_cancelled():
        return {}

    completed_files = loop_result.get("completed_files", [])
    diffs           = loop_result.get("diffs", [])
    failed_steps    = loop_result.get("failed_steps", [])

    if diffs:
        emit("code_done", "Files written:\n" + "\n".join(diffs))

    if failed_steps:
        emit("status", f"⚠️ {len(failed_steps)} step(s) skipped: {failed_steps}")

    # ──────────────────────────────────────────────────────────────────
    # STAGE 5: FINAL REVIEW — review all written files together
    # ──────────────────────────────────────────────────────────────────
    if is_cancelled():
        return {}

    emit("status", "✅ Final review...")

    # Read all written files for the review
    final_file_content = ""
    for fpath in completed_files:
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                rel = os.path.relpath(fpath, project_dir) if project_dir else fpath
                final_file_content += f"\nFILE: {rel}\n```\n{content}\n```\n"
            except Exception:
                pass

    if not final_file_content:
        final_file_content = file_context  # fall back to original context

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN:
{extract_plan_summary(steps)}

CODE OUTPUT:
{final_file_content}

Check ALL of the following:
- Correctness: does the code do what the task asks?
- Logic errors: off-by-one, wrong conditions, incorrect operator precedence
- Function signatures: does every function definition match exactly how it is called?
  Pay special attention to callback/handler functions registered with frameworks
  (e.g. tkinter validatecommand requires specific argument patterns)
- Dead code: is every defined function actually called somewhere? Flag unused functions.
- Cross-file consistency: if multiple files are present, do they agree on data formats,
  allowed values, and interfaces? (e.g. if one file validates input to a set of values,
  the other file must produce/accept exactly those values)
- Missing error handling: unhandled exceptions, missing edge cases
- Security issues: unsafe eval, injection risks, path traversal

Respond in this exact format:
VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list each issue on its own line, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    # ──────────────────────────────────────────────────────────────────
    # STAGE 6: RETRY — only if reviewer flagged issues
    # ──────────────────────────────────────────────────────────────────
    final_code = final_file_content

    if verdict in ("NEEDS_CHANGES", "FAIL"):
        if is_cancelled():
            return {}

        emit("status", "🔁 Fixing reviewer issues...")

        code_files = re.findall(r"FILE:\s*([^\n]+)\n```", final_file_content)
        file_list = "\n".join(f"- FILE: {f.strip()}" for f in code_files) if code_files \
                    else "- same files as in the previous code"

        retry_prompt = f"""
The previous code had issues flagged by a reviewer. Produce a complete corrected version.

ORIGINAL TASK: {task}{target_instruction}
REVIEWER FEEDBACK:
{review}
PLAN:
{extract_plan_summary(steps)}
PREVIOUS CODE:
{final_file_content}

You MUST output ALL of the following files in their COMPLETE corrected form:
{file_list}

Use this STRICT format for EVERY file:

FILE: filename.py
```python
<complete file contents — every line, no omissions>
```

Rules:
- One FILE: block per file listed above
- Complete file contents only — never partial, never snippets
- No explanations outside the FILE blocks
- If a file needs no changes, output it anyway unchanged
""".strip()

        retry_output = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER)

        # Only use retry if it produced proper FILE: blocks matching what we had
        import re as _re
        retry_count = len(_re.findall(r"FILE:\s*[^\n]+\n```", retry_output))
        code_count  = len(_re.findall(r"FILE:\s*[^\n]+\n```", final_file_content))
        if retry_count > 0 and retry_count >= code_count:
            final_code = retry_output
            emit("retry_done", retry_output)
        else:
            log(f"[pipeline] retry had {retry_count} FILE: blocks vs {code_count} — keeping original")
            emit("retry_done", f"(Retry did not improve output — keeping reviewed version)")

    log(f"[pipeline] completed. verdict={verdict} files={len(completed_files)}")
    return {
        "scan":            scan,
        "plan":            plan,
        "steps":           steps,
        "review":          review,
        "verdict":         verdict,
        "final_code":      final_code,
        "completed_files": completed_files,
        "diffs":           diffs,
    }


def _single_step_fallback(
    task, plan, file_context, brief_section,
    target_instruction, coder, emit, is_cancelled,
    project_dir, context_files
) -> dict:
    """
    Original single-shot code generation.
    Used when the REASON stage doesn't produce structured steps.
    Preserves backward compatibility.
    """
    emit("status", "⚙️ Writing code...")

    files_section = "FILES:\n" + file_context if file_context else ""

    code_prompt = f"""
TASK: {task}{target_instruction}
EXECUTION PLAN:
{plan}

{files_section}

Execute the plan exactly. Output EVERY changed or created file using this STRICT format:

FILE: relative/path/to/file.py
```python
<complete file contents here>
```

Rules:
- Always use the FILE: line before each code block
- Always show the COMPLETE file — never partial or diff output
- Use the correct language tag (python, js, etc.)
- No explanations outside the FILE blocks

Write production-quality code. Do not deviate from the plan.
""".strip()

    code = single_response(coder, code_prompt, num_ctx=MAX_CTX_CODER)
    emit("code_done", code)

    review_prompt = f"""
You are a code reviewer. Review this AI-generated code.

ORIGINAL TASK: {task}
PLAN: {plan}
CODE OUTPUT: {code}

Check: correctness, logic errors, dead code, cross-file consistency,
function signatures vs call sites, missing error handling, security issues.

VERDICT: PASS / NEEDS_CHANGES / FAIL
ISSUES: (list, or "None")
SUGGESTIONS: (specific improvements, or "None")
""".strip()

    review  = single_response(coder, review_prompt, num_ctx=MAX_CTX_CODER)
    verdict = extract_verdict(review)
    emit("review_done", f"VERDICT: {verdict}\n\n{review}")

    final_code = code
    if verdict in ("NEEDS_CHANGES", "FAIL") and not is_cancelled():
        emit("status", "🔁 Fixing...")
        from engine.apply_changes import extract_files as _ef
        code_files = re.findall(r"FILE:\s*([^\n]+)\n```", code)
        file_list = "\n".join(f"- FILE: {f.strip()}" for f in code_files) if code_files else ""

        retry_prompt = f"""
Fix the issues flagged by the reviewer. Output ALL files complete.

TASK: {task}{target_instruction}
FEEDBACK: {review}
PREVIOUS CODE: {code}
REQUIRED FILES: {file_list}

FILE: filename.py
```python
<complete file>
```
""".strip()
        retry = single_response(coder, retry_prompt, num_ctx=MAX_CTX_CODER)
        import re as _re
        if len(_re.findall(r"FILE:\s*[^\n]+\n```", retry)) > 0:
            final_code = retry
        emit("retry_done", final_code)

    return {
        "scan": "", "plan": plan, "steps": [],
        "review": review, "verdict": verdict,
        "final_code": final_code, "completed_files": [], "diffs": [],
    }
