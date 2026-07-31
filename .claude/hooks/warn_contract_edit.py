#!/usr/bin/env python3
"""
PreToolUse hook: flag edits that touch the observation-contract constants.

CLAUDE.md's "Hard invariants" section names the exact failure mode this
guards: a wrong OBS_SCALES value, a flipped hip sign, or a changed
DECIMATION fails SILENTLY in the simulator -- the robot just walks worse,
with no exception and no red test (unless test_obs_contract.py happens to
pin that exact field). This does not block the edit -- these files are
legitimately edited (this session did, to add body-frame metrics to
harness.py) -- it injects a reminder into context so the edit gets the
scrutiny CLAUDE.md asks for, and reruns `make test` mentally before treating
the edit as safe.

Registered as a PreToolUse hook on Edit/Write in .claude/settings.json.
Reads the tool call as JSON on stdin, writes additionalContext to stdout as
JSON (see the PreToolUse hook output schema), always exits 0 -- this hook
warns, it does not block.
"""
import json
import re
import sys

# Files where the 70-dim contract's numeric constants live. Not every
# harness.py edit needs the warning (e.g. this session's metric-logging
# addition didn't touch a listed identifier) -- only these names do.
GUARDED_FILES = (
    "harness.py",
    "networks.py",
    "go2_env.py",
)
GUARDED_IDENTIFIERS = (
    "DEFAULT_JOINT_POS", "OBS_SCALES", "COMMANDS_SCALE",
    "ACTION_SCALE", "HIP_SCALE_REDUCTION", "KP", "KD",
    "DECIMATION", "HISTORY_LEN", "OBS_DIM", "PRIV_DIM",
)


def touched_identifiers(text: str) -> list[str]:
    return [name for name in GUARDED_IDENTIFIERS
            if re.search(rf"\b{re.escape(name)}\b", text or "")]


def main():
    payload = json.load(sys.stdin)
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})

    if tool_name not in ("Edit", "Write"):
        sys.exit(0)

    path = tool_input.get("file_path", "")
    if not any(path.endswith(f) for f in GUARDED_FILES):
        sys.exit(0)

    changed_text = " ".join(str(tool_input.get(k, ""))
                            for k in ("new_string", "content"))
    hits = touched_identifiers(changed_text)
    if not hits:
        sys.exit(0)

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": (
                f"REMINDER (from .claude/hooks/warn_contract_edit.py): this "
                f"edit to {path} touches {', '.join(hits)}, part of the "
                f"70-dim observation contract documented in CLAUDE.md's "
                f"'Hard invariants' section. A wrong value here fails "
                f"SILENTLY in MuJoCo -- no exception, the robot just walks "
                f"worse. Run `make test` after this edit; "
                f"tests/test_obs_contract.py and tests/test_constants.py "
                f"exist specifically to catch this class of mistake."
            ),
        }
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
