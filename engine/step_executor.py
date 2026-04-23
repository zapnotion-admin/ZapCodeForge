"""
engine/step_executor.py
The step execution loop — core of the agent architecture.

For each step:
  1. Read current on-disk file state (file system is the memory)
  2. Build context: full content for step files, interface-only for others
  3. Generate code for this step only
  4. Parse FILE: blocks from output
  5. Stage the writes (not committed yet)
  6. Verify the step (lightweight check)
  7. On PASS → commit staged files, advance state
  8. On FAIL → rollback staged, retry once with failure reason
  9. On second FAIL → skip step, log failure, continue

Never asks the model to hold the whole program in memory.
Never commits broken code to disk.
"""

import os
import re
from engine.logger import log
from engine.ollama_client import single_response, ensure_model
from engine.apply_changes import extract_files
from engine.context_manager import (
    build_file_context_for_step,
    read_project_files,
    compute_diff,
)
from engine.plan_parser import steps_to_status_summary, extract_plan_summary
from engine.step_state import StepState
from engine.brief import format_brief_for_prompt
from core.config import MODEL_CODER, MODEL_FALLBACK, MAX_CTX_CODER


MAX_RETRIES = 1   # Retry a failed step this many times before skipping


def run_steps(
    state:             StepState,
    task:              str,
    all_context_files: list,    # all files the user has in context (filenames)
    project_dir:       str,
    brief_content:     str,
    coder_model:       str = None,  # which model to use for code generation
    progress_callback  = None,
    cancel_check       = None,
) -> dict:
    """
    Executes all pending steps in the state machine.

    Returns dict with:
      completed_files: list of all files written
      diffs:           list of diff strings per step
      failed_steps:    list of step numbers that were skipped
    """

    def emit(stage: str, text: str) -> None:
        log(f"[step_executor] {stage}: {text[:80]}")
        if progress_callback:
            progress_callback(stage, text)

    def is_cancelled() -> bool:
        return cancel_check is not None and cancel_check()

    brief_section  = format_brief_for_prompt(brief_content)
    plan_summary   = extract_plan_summary(state.steps)
    completed_files: list = []
    all_diffs:       list = []

    # Resolve which model to use — default to config
    from engine.ollama_client import resolve_model
    from core.config import MODEL_FALLBACK
    active_coder = resolve_model(coder_model or MODEL_CODER, MODEL_FALLBACK)
    # Ensure the coder model is loaded (may need to swap from reasoner)
    ensure_model(active_coder)

    while True:
        if is_cancelled():
            state.cancel()
            emit("status", "⛔ Cancelled.")
            break

        next_idx = state.next_pending_index
        if next_idx is None:
            state.complete()
            break

        step = state.steps[next_idx]
        step_num   = step["number"]
        step_total = len(state.steps)
        step_desc  = step["description"]
        step_files = step.get("files", [])

        emit("step_start", f"Step {step_num}/{step_total}: {step_desc}")
        state.begin_step(next_idx)

        # ── Determine which files to include ────────────────────────
        # If step didn't specify files, include all context files
        if not step_files:
            step_files = [os.path.basename(f) for f in all_context_files]

        # All files relevant to this project (for interface-only context)
        all_relevant = list(set(
            [os.path.basename(f) for f in all_context_files] + step_files
        ))

        retry_reason = ""
        success = False

        for attempt in range(MAX_RETRIES + 1):
            if is_cancelled():
                state.cancel()
                return _result(completed_files, all_diffs, state)

            if attempt > 0:
                emit("status", f"  ↩ Retrying step {step_num} (attempt {attempt + 1})...")
                state.retry_step()

            # ── Read current on-disk state ───────────────────────────
            current_files = read_project_files(project_dir, all_relevant)

            # ── Build context ────────────────────────────────────────
            file_context = build_file_context_for_step(
                step_files=step_files,
                all_project_files=current_files,
            )

            status_summary = steps_to_status_summary(state.steps)

            # ── Build prompt ─────────────────────────────────────────
            retry_note = (
                f"\nPREVIOUS ATTEMPT FAILED: {retry_reason}\n"
                f"Fix that specific issue in this attempt.\n"
            ) if retry_reason else ""

            prompt = f"""
TASK: {task}
{brief_section}
OVERALL PLAN:
{plan_summary}

PROGRESS:
{status_summary}

{file_context}

{retry_note}
EXECUTE THIS STEP ONLY — Step {step_num} of {step_total}:
{step_desc}

Rules:
- Only make the changes required by this step
- Do not implement anything from future steps
- Do not remove anything added by previous steps
- Output EVERY modified or created file using this STRICT format:

FILE: filename.py
```python
<complete file contents — every line>
```

- One FILE: block per file you touch
- Complete file only — no diffs, no snippets, no omissions
""".strip()

            # ── Generate ─────────────────────────────────────────────
            try:
                raw_output = single_response(
                    active_coder, prompt, num_ctx=MAX_CTX_CODER
                )
            except Exception as e:
                retry_reason = f"Model generation failed: {e}"
                log(f"[step_executor] Generation error: {e}")
                continue

            # ── Parse FILE: blocks ───────────────────────────────────
            parsed_files = extract_files(raw_output, task=step_desc)
            if not parsed_files:
                retry_reason = "No FILE: blocks found in output — model may not have followed format"
                log(f"[step_executor] No files parsed from output")
                continue

            # ── Stage the writes ──────────────────────────────────────
            for f in parsed_files:
                rel_path  = f["path"]
                full_path = os.path.join(project_dir, rel_path) if project_dir else rel_path
                state.stage_file(full_path, f["code"])

            # ── Verify the step ───────────────────────────────────────
            verify_result = _verify_step(
                step_desc   = step_desc,
                output      = raw_output,
                parsed_files = parsed_files,
            )

            if verify_result["pass"]:
                # Compute diffs before committing
                for f in parsed_files:
                    rel_path  = f["path"]
                    full_path = os.path.join(project_dir, rel_path) if project_dir else rel_path
                    old = current_files.get(os.path.basename(rel_path), "")
                    diff = compute_diff(old, f["code"], rel_path)
                    all_diffs.append(diff)
                    if full_path not in completed_files:
                        completed_files.append(full_path)

                state.step_success()
                names = [f["path"] for f in parsed_files]
                emit("step_done", f"✓ Step {step_num}: {', '.join(names)}")
                success = True
                break
            else:
                retry_reason = verify_result["reason"]
                log(f"[step_executor] Verify failed: {retry_reason}")

        if not success:
            state.step_failed(retry_reason)
            emit("step_failed", f"✗ Step {step_num} failed after {MAX_RETRIES + 1} attempts — skipping. Reason: {retry_reason}")

    return _result(completed_files, all_diffs, state)


def _verify_step(step_desc: str, output: str, parsed_files: list) -> dict:
    """
    Lightweight verification — no model call needed for basic checks.
    Returns {"pass": bool, "reason": str}

    Checks:
    - At least one FILE: block was produced
    - Output isn't suspiciously short (< 5 lines total)
    - No obvious error markers in the output
    """
    if not parsed_files:
        return {"pass": False, "reason": "No FILE: blocks in output"}

    total_lines = sum(f["code"].count("\n") + 1 for f in parsed_files)
    if total_lines < 3:
        return {"pass": False, "reason": f"Output suspiciously short ({total_lines} lines total)"}

    # Check for common model failure patterns
    failure_markers = [
        "i cannot", "i can't", "i'm unable",
        "error:", "traceback", "syntaxerror",
    ]
    output_lower = output.lower()
    for marker in failure_markers:
        if marker in output_lower and "FILE:" not in output:
            return {"pass": False, "reason": f"Output contains failure marker: '{marker}'"}

    return {"pass": True, "reason": ""}


def _result(completed_files: list, diffs: list, state: StepState) -> dict:
    return {
        "completed_files": completed_files,
        "diffs":           diffs,
        "failed_steps":    [s["number"] for s in state.steps if s["status"] == "failed"],
        "completed_steps": [s["number"] for s in state.steps if s["status"] == "complete"],
        "state":           state.state,
    }
