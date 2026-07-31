# Agentic Coding Workflow

How to work productively on this repository with AI coding tools (Claude Code,
Cursor, and friends), and — more importantly — **where they help and where they
will confidently hurt you.**

This is not a generic "prompt engineering" guide. It is specific to a research
codebase where a wrong sign flips a robot on its back and the failure is
silent.

---

## Contents

1. [The honest framing](#1-the-honest-framing)
2. [Context files: the highest-leverage thing you can do](#2-context-files-the-highest-leverage-thing-you-can-do)
3. [Skills, commands and reusable procedures](#3-skills-commands-and-reusable-procedures)
4. [Subagents and parallel work](#4-subagents-and-parallel-work)
5. [Hooks and automation](#5-hooks-and-automation)
6. [MCP — connecting external tools](#6-mcp--connecting-external-tools)
7. [Cursor specifics](#7-cursor-specifics)
8. [Task patterns for this repo](#8-task-patterns-for-this-repo)
9. [The verification discipline](#9-the-verification-discipline)
10. [Where agents fail on this codebase](#10-where-agents-fail-on-this-codebase)
11. [Research skills for RL/robotics](#11-research-skills-for-rlrobotics)
12. [Open-source references worth reading](#12-open-source-references-worth-reading)

---

## 1. The honest framing

An AI coding agent is very good at:

- reading 2,600 lines of unfamiliar Python and telling you what it does
- writing boilerplate (test scaffolding, CLI parsing, CSV plumbing, Dockerfiles)
- cross-referencing constants across files that duplicate them
- turning "why does the robot fall over" into a list of candidate causes
- writing and maintaining documentation like the set you are reading

It is dangerously mediocre at:

- **knowing whether a physics simulation is right.** Code that runs and produces
  plausible numbers may still be wrong. There is no test that fails.
- **respecting an undocumented external contract.** The 70-dim observation
  layout is not written down anywhere except the training source. An agent will
  cheerfully "clean up" a field ordering and break everything.
- **judging research validity.** It can compute a mean; it cannot tell you the
  metric is measuring the wrong thing (which is exactly what happened with the
  survival-vs-traversal correction — a *human* noticed).

**The rule that follows:** use agents freely for infrastructure, documentation,
analysis and scaffolding. For anything touching the observation vector, the
control constants, or a research claim, use them to *propose* and yourself to
*verify against the source of truth*.

---

## 2. Context files: the highest-leverage thing you can do

Every serious agentic tool reads a project context file at session start. It is
the single best return on effort available to you.

| Tool | File |
|---|---|
| Claude Code | `CLAUDE.md` (repo root; also `~/.claude/CLAUDE.md` for personal prefs) |
| Cursor | `.cursor/rules/*.mdc` (modern) or `.cursorrules` (legacy) |
| GitHub Copilot | `.github/copilot-instructions.md` |
| Generic / multi-tool | `AGENTS.md` — the cross-tool convention (Cursor, Copilot, Codex, Gemini CLI, Windsurf, Devin, Aider, ... all read it; Claude Code does not, natively) |

This repo ships a [`CLAUDE.md`](../CLAUDE.md) at the root — read it, it is
short and it encodes the traps — plus [`AGENTS.md`](../AGENTS.md), a symlink
to the same file. Claude Code only reads `CLAUDE.md`, so the symlink exists
purely so a non-Claude tool opening this repo gets the identical content
without a second file to keep in sync. Edit `CLAUDE.md`; `AGENTS.md` follows
automatically because it's the same inode, not a copy.

### What belongs in a context file

Things an agent **cannot infer** and will get wrong:

```markdown
✅ "The observation vector layout is an immutable external contract.
    Never reorder fields. harness.py is the reference; 04_build_obs_vector.py
    is a SUPERSEDED artefact with a known-wrong layout."

✅ "DEFAULT_JOINT_POS hip signs are FL/RL positive, FR/RR negative.
    Verified against scenes/go2_model/go2.xml lines 189-200."

✅ "Scripts 01-08 are a teaching curriculum. Duplication between them is
    intentional. Do not extract shared modules from 01-06."

✅ "Always pass absolute paths to mujoco.MjModel.from_xml_path()."
```

Things that **waste context** because the agent can read them:

```markdown
❌ "This project uses Python and NumPy."
❌ "The scripts are in stage2-go2-mujoco-inference/."
❌ A restatement of the directory tree.
```

The test: *would a competent new contributor get this wrong on day one?* If
yes, it belongs in the context file. If they would figure it out in thirty
seconds of reading, leave it out.

### Keep it current

A context file that lies is worse than none. When you change a convention,
update it in the same commit.

---

## 3. Skills, commands and reusable procedures

Once you find yourself explaining the same multi-step procedure twice, encode
it.

### Claude Code: skills

A **skill** is a folder with a `SKILL.md` describing a procedure the agent
loads on demand. Project skills live in `.claude/skills/<name>/SKILL.md`.

```
.claude/skills/
└── run-experiment/
    └── SKILL.md
```

```markdown
---
name: run-experiment
description: Run a Stage 2 experiment sweep end to end — verify the
  environment, confirm policy weights are present, execute the sweep,
  regenerate figures, and summarise results against prior findings.
  Use when asked to run exp1, exp2, exp3, or a new sweep.
---

# Running an experiment

1. `python scripts/check_env.py` — abort if policy weights are missing.
2. Run the sweep. NEVER edit constants in the experiment script to change
   parameters; copy the script to a new exp4_*.py instead, so past results
   stay reproducible from the committed source.
3. `python stage2-go2-mujoco-inference/experiments/make_figures.py`
4. Compare against results/EXPERIMENT_FINDINGS.md and report deltas.
5. Append a new findings section — never overwrite an existing one.
```

Good skill candidates here:

| Skill | What it encodes |
|---|---|
| `run-experiment` | the sweep → figures → findings pipeline above |
| `verify-obs-contract` | check a `build_obs` change against the 70-dim contract |
| `add-terrain` | builder → regenerate → commit XML → add condition to exp3 |
| `onboard` | the reading order in [docs/README.md](README.md) |

### Slash commands

Simpler than skills: a markdown file in `.claude/commands/<name>.md` becomes
`/<name>`. Good for short, frequent prompts ("summarise the diff", "check this
against the findings log").

### The general principle

Skills and commands are **procedural memory**. Context files are **declarative
memory**. You want both: the context file says *what is true*, the skill says
*what to do*.

---

## 4. Subagents and parallel work

Claude Code can dispatch subagents with their own context windows. Cursor has
analogous background-agent features.

**Where this genuinely helps on this repo:**

- *Search fan-out.* "Find every place `DEFAULT_JOINT_POS` is defined and report
  whether the values agree." Four files, one answer, none of the file contents
  polluting your main context.
- *Independent workstreams.* One agent writing tests while you refactor
  `harness.py`, in separate git worktrees.
- *Long analysis.* "Read all 8 theory chapters and list every claim about PPO
  hyperparameters that the Stage 1 code contradicts."

**Where it does not:** anything requiring shared judgement, or where the
subagent would need the same deep context you already have. A cold subagent
re-derives what you already know. Do not reach for one when a `grep` and a read
would do.

---

## 5. Hooks and automation

Hooks run shell commands on agent lifecycle events (before/after a tool call,
on session start), configured in `.claude/settings.json`. The value is a
**deterministic gate**: a hook either fires or it doesn't, unlike a rule
written in prose in a context file, which is only as reliable as the model's
attention to it deep in a long session.

**Two hooks are live in this repo** (`.claude/hooks/`, wired in
`.claude/settings.json`), both `PreToolUse` — they run *before* the tool
call, so a block actually prevents the action rather than just complaining
after:

```jsonc
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{"type": "command",
                   "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/guard_commit_footprint.py\""}]
      },
      {
        "matcher": "Edit|Write",
        "hooks": [{"type": "command",
                   "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/warn_contract_edit.py\""}]
      }
    ]
  }
}
```

**`guard_commit_footprint.py`** — hard block, exit 2. Turns the "commits
carry no agentic footprint" rule from CLAUDE.md prose into something that
mechanically cannot slip through: any `git commit` whose message contains
`Co-Authored-By: Claude`, `Generated with`, or similar is denied before it
runs, with the reason fed back to the agent so it retries clean.

**`warn_contract_edit.py`** — soft warning, exit 0, injects
`additionalContext`. Fires when an `Edit`/`Write` to `harness.py`,
`networks.py`, or `go2_env.py` touches one of the observation-contract
constants (`DEFAULT_JOINT_POS`, `OBS_SCALES`, `KP`/`KD`, `DECIMATION`, ...).
Doesn't block — these files are legitimately edited, this session added
body-frame metrics to `harness.py` — it just surfaces the exact CLAUDE.md
warning ("a wrong value fails silently in the simulator") at the moment of
the edit, and points at the two tests that exist to catch it.

The asymmetry is deliberate: mechanically enforce the rule that has no
legitimate exception (no commit should ever need an AI trailer); only remind
for the rule that has plenty of legitimate exceptions (contract files get
edited on purpose, they just need extra scrutiny when they do).

Other useful patterns for this repo, not yet wired up:

```jsonc
{
  "hooks": {
    // Compile-check every Python file the agent edits, immediately.
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{
        "type": "command",
        "command": "python -m py_compile \"$CLAUDE_FILE_PATH\" 2>&1 | head -5"
      }]
    }]
  }
}
```

> Note: hooks execute shell commands with your permissions. Only add hooks you
> would be comfortable running yourself, and read any hook config you did not
> write. `.claude/hooks/*.py` are committed and reviewable like any other
> code — read them before trusting them, same as any hook from elsewhere.

---

## 6. MCP — connecting external tools

Model Context Protocol servers give an agent typed access to external systems
(GitHub, databases, browsers, file systems). Configure per-project in
`.mcp.json`.

Realistic value here is modest — this repo has no database and no external API.
Where it could pay off:

- **GitHub MCP** — file the "add a test suite" issue, open the PR, read CI logs
  without leaving the session.
- **Filesystem MCP** scoped to a large policy-checkpoint directory outside the
  repo, so an agent can inventory available `.jit` files.

Do not install MCP servers speculatively. Each one costs context on every turn.

---

## 7. Cursor specifics

Cursor's mental model differs slightly from Claude Code's terminal-first one.

**Rules.** `.cursor/rules/*.mdc` files with frontmatter controlling when they
apply:

```mdc
---
description: Stage 2 inference invariants
globs: ["stage2-go2-mujoco-inference/**/*.py"]
alwaysApply: false
---

The 70-dim observation layout is an immutable external contract fixed by the
pretrained policy. Never reorder fields, never change OBS_SCALES,
DEFAULT_JOINT_POS, ACTION_SCALE, KP/KD or DECIMATION without explicit
instruction — these were recovered from the walk-these-ways training source
and a wrong value fails silently.

Reference implementation: experiments/harness.py
Known-wrong file: 04_build_obs_vector.py (superseded, kept for history)
```

Glob-scoped rules are strictly better than one global rule file: the Stage 2
invariants above are irrelevant when you are editing Stage 1.

**Practical Cursor habits for this repo:**

- Use `@file` / `@folder` to pin `harness.py` and `paths.py` when working on
  inference. Precision beats letting the retriever guess.
- Composer/Agent mode for multi-file work (adding a metric touches
  `harness.py` + three experiment scripts + `make_figures.py`).
- Turn off aggressive autocomplete when editing constants. Tab-completing
  `-0.1` into `0.1` in `DEFAULT_JOINT_POS` is a real risk and a silent one.

**Portability tip:** keep the substance in `CLAUDE.md` and have thin Cursor
rules reference it, or maintain `AGENTS.md` as the shared source. Duplicated
rule files drift.

---

## 8. Task patterns for this repo

Concrete prompts that work, with the context that makes them work.

### Pattern — understand before changing

```
Read stage2-go2-mujoco-inference/experiments/harness.py and
docs/ARCHITECTURE.md sections 5-7. Explain how the 70-dim observation is
assembled and which constants must match the training config. Do not change
anything yet.
```

Front-load understanding. Agents that start editing immediately are the ones
that break invariants.

### Pattern — add a metric (touches 5 files)

```
Add a "cost of transport" metric to the experiment harness.

Constraints:
- Log the quantity inside run_trial's measurement window only
- Add it to the returned dict AND to CSV_FIELDS in exp1, exp2, exp3
- Existing CSVs must stay readable — this is additive
- Do not change any control constant
- Show me the diff before writing files
```

Naming the blast radius up front prevents a half-applied change.

### Pattern — write the missing tests

```
Write tests/test_contracts.py using pytest. It must run WITHOUT the policy
checkpoints. Cover:
  1. go2_flat.xml loads with nq=19, nv=18, nu=12
  2. actuator indices 0,3,6,9 are the four hip joints (check names via
     mujoco.mj_id2name)
  3. quat_rotate_inverse with the identity quaternion returns v unchanged
  4. each GAIT_PRESETS entry produces a distinct 4-vector of clock signals
  5. regenerating terrain scenes produces files identical to the committed ones

Run them and show me the output.
```

That last line matters more than the rest of the prompt. See §9.

### Pattern — investigate a discrepancy

```
The root README claims ~0.28 m/s at commanded 0.5 m/s. Experiment 1's CSV
reports 0.227 ± 0.005 for the same command. Read both sources and the harness,
and give me the most likely explanation. Do not edit anything.
```

Agents are good at this. It is reading and cross-referencing, which is their
strongest mode.

### Pattern — the anti-pattern

```
❌ "Clean up and refactor the stage2 scripts."
```

This will produce a beautifully DRY codebase with a broken observation vector
and a destroyed curriculum. Scope every request.

---

## 9. The verification discipline

The one habit that separates useful agentic work from expensive rework.

**Never accept a claim of success without an artefact.**

| Claim | Acceptable evidence |
|---|---|
| "I added tests" | the pytest output, pasted |
| "The environment is set up" | `python scripts/check_env.py` output |
| "The figures regenerate" | the script's stdout listing four PNGs |
| "Scenes are reproducible" | `git diff --stat` showing empty |
| "It runs in Docker" | the container command and its output |
| "The refactor is safe" | tests passing before *and* after |

This repo makes verification unusually cheap, which you should exploit:

```bash
python scripts/check_env.py                                   # 4-layer check
python -m py_compile stage2-go2-mujoco-inference/*.py          # syntax
python stage2-go2-mujoco-inference/paths.py                    # path resolution
python .../experiments/make_figures.py                         # full data path
python .../experiments/generate_terrain_scenes.py && git diff   # reproducibility
```

None of these need the policy weights. All of them run in seconds. Ask for them
by name.

---

## 10. Where agents fail on this codebase

Specific, observed failure modes. Guard against each explicitly.

| # | Failure | Why it happens | Guard |
|---|---|---|---|
| 1 | Reorders observation fields "for clarity" | the layout looks arbitrary because its authority is external | state the contract in `CLAUDE.md`; review any `build_obs` diff by hand |
| 2 | Copies from `04_build_obs_vector.py` | it is named like the canonical implementation and appears earlier | banner the file; name `harness.py` as the reference |
| 3 | Deduplicates `01`–`06` | duplication genuinely looks like a defect | say "curriculum, not application" in the context file |
| 4 | Flips a hip sign | `+0.1`/`−0.1` alternating looks like a typo | cite `go2.xml` lines 189–200 as the authority |
| 5 | Changes `DECIMATION` while "optimising" | it looks like a tunable | mark control constants immutable |
| 6 | Guesses/invents a download URL, or asserts none exists without checking | wants to be helpful, or trusts stale docs | the real one is verified in [DEPENDENCIES.md §6](DEPENDENCIES.md#6-the-policy-weights-are-not-committed-here-but-are-downloadable) — use it or re-verify, don't guess |
| 7 | Reports "tests pass" without running them | plausible-sounding completion | demand pasted output (§9) |
| 8 | Edits experiment constants to re-run a sweep | it is the shortest path to the request | new `exp4_*.py` instead; keep past results reproducible |
| 9 | "Fixes" `null` metrics for fallen trials | nulls look like a bug | nulls are deliberate — see [TDD.md §5](TDD.md#5-error-handling) |

---

## 11. Research skills for RL/robotics

The non-coding skills that determine whether this project produces anything
worth publishing. Agents assist with all of them; none of them are automatable.

### Reading source over documentation

This project's central methodological claim, and it is right. The
`walk-these-ways` observation layout, scale factors and sign conventions are
documented *nowhere* — the training code is the only authority. Every
sim-to-sim bug in the Stage 2 README was found by reading upstream source.

**Agent leverage:** excellent. "Read this training config dump and tell me the
order in which observation components are concatenated" is exactly the task
they are best at. Verify the answer against the code, not the summary.

### Designing an experiment that can fail

Experiment 3 is the model. It defined *in advance* what "traversable" meant,
ran 5 seeds per condition, and reported 0% survival at 20° without hedging. A
study that cannot produce a negative result is not a study.

**Agent leverage:** moderate. Good at scaffolding sweeps, poor at deciding what
would falsify the hypothesis.

### Noticing when your metric is wrong ⭐

The single best moment in this repo's history: 100% survival on 16 cm stairs
looked like a triumph until someone checked `distance_traveled` and found a
deterministic 0.47 m — the robot was standing politely at the bottom of the
staircase. Survival was measuring the wrong thing.

**Agent leverage:** low, and this is the important lesson. An agent would have
reported 100% survival and moved on. *Suspicion of a good result* is a human
contribution.

### Statistical honesty

5 seeds per condition, mean ± std, survivors-only aggregation, limitations
stated in the findings document itself ("tested only at cmd=0.5 m/s"). The
findings log names its own weakness before a reviewer can.

**Agent leverage:** good for computation, poor for judgement about what caveat
matters.

### Keeping a findings log

`EXPERIMENT_FINDINGS.md` is written immediately after each run, while the
analysis is fresh, and separates *what happened* (CSV) from *what it means*
(prose). Every finding is numbered (F1.1, F3.4) so later work can cite it.

**Agent leverage:** high. Drafting a findings entry from a CSV plus run log is
a genuinely good use of an agent — then you edit the interpretation.

---

## 12. Open-source references worth reading

### Directly upstream of this project

| Project | Why |
|---|---|
| [`Improbable-AI/walk-these-ways`](https://github.com/Improbable-AI/walk-these-ways) | the policy this repo deploys. The **only** authority on the observation contract. Read `legged_robot.py`'s observation assembly and the gait clock. |
| [`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie) | source of the Go2 MJCF vendored in `scenes/go2_model/` |
| [`google-deepmind/mujoco`](https://github.com/google-deepmind/mujoco) | the simulator; the Python bindings' source is very readable |
| [`google-deepmind/mujoco_playground`](https://github.com/google-deepmind/mujoco_playground) | MJX-based RL environments — the likely Stage 3 foundation |
| [`DLR-RM/stable-baselines3`](https://github.com/DLR-RM/stable-baselines3) | the PPO used in Stage 1; read `ppo.py` next to theory chapter 5 |
| [`Farama-Foundation/Gymnasium`](https://github.com/Farama-Foundation/Gymnasium) | the environment API |

### Worth studying for how they organise research code

| Project | Lesson |
|---|---|
| [`leggedrobotics/legged_gym`](https://github.com/leggedrobotics/legged_gym) | the config-class pattern most legged-RL repos inherit |
| [`DLR-RM/rl-baselines3-zoo`](https://github.com/DLR-RM/rl-baselines3-zoo) | how to structure hyperparameter sweeps and store results |
| [`google/brax`](https://github.com/google/brax) | JAX-based physics; relevant if Stage 3 goes the MJX route |

### Agentic tooling references

| Resource | Why |
|---|---|
| [Claude Code docs](https://docs.claude.com/en/docs/claude-code) | skills, hooks, subagents, settings, MCP |
| [Model Context Protocol](https://modelcontextprotocol.io) | the MCP spec and reference servers |
| [Cursor docs](https://docs.cursor.com) | rules, `.mdc` frontmatter, Composer |
| [`anthropics/claude-code`](https://github.com/anthropics/claude-code) | issues and discussions are where real usage patterns surface |

### How to actually read an unfamiliar research repo

The method that produced the docs in this folder:

1. **Entry points first.** Find what is executable (`if __name__ == "__main__"`),
   not what is imported. That tells you what the project *does*.
2. **Follow the data, not the call graph.** Where does state come from, what
   transforms it, where does it land? For this repo: `MjData` → obs → policy →
   torque → `MjData`.
3. **Find the contracts.** Any fixed-size array crossing a module boundary is a
   contract. Here: 70, 2100, 2102, 12. Those numbers organise everything.
4. **Read the git log for *why*.** `2810d2b "Add reproducible self-contained
   Go2 model"` explains a design decision no comment does.
5. **Run the cheapest thing that produces output.** Here that is
   `make_figures.py` — it needs no policy weights and exercises the whole data
   path.
6. **Write down what surprised you.** That list becomes the onboarding doc, and
   it is the highest-value artefact a new contributor can produce.

---

## Related

- [`../CLAUDE.md`](../CLAUDE.md) — the context file itself
- [ARCHITECTURE.md](ARCHITECTURE.md) — what agents need to understand first
- [REVERSE-ENGINEERING.md](REVERSE-ENGINEERING.md) — verified findings and first tasks
- [TECH-STACK-PRIMER.md](TECH-STACK-PRIMER.md) — the technology basics
