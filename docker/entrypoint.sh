#!/usr/bin/env bash
# Container entrypoint: print a short orientation banner, then exec the command.
# Keeps the "what can I actually run right now?" answer visible on every start.
set -e

if [[ -t 1 && "${RLL_QUIET:-0}" != "1" ]]; then
  POLICY_DIR="${GO2_POLICY_DIR:-/workspace/policies/walk-these-ways-go2}"
  if [[ -f "${POLICY_DIR}/body_latest.jit" ]]; then
    POLICY_STATUS=$'\033[32mfound\033[0m'
  else
    POLICY_STATUS=$'\033[33mMISSING — mount it, see docs/DEPENDENCIES.md\033[0m'
  fi

  cat <<BANNER

  rl-locomotion-learning container
  --------------------------------
  python        $(python --version 2>&1 | awk '{print $2}')
  mujoco        $(python -c 'import mujoco;print(mujoco.__version__)' 2>/dev/null || echo '?')
  torch         $(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '?')
  MUJOCO_GL     ${MUJOCO_GL:-unset}
  policy        ${POLICY_STATUS}

  Runs with no policy weights:
    python scripts/check_env.py
    python stage2-go2-mujoco-inference/experiments/make_figures.py
    python stage2-go2-mujoco-inference/experiments/generate_terrain_scenes.py
    python stage1-rl-fundamentals/01_cartpole_ppo.py

  Needs policy weights (mount into ${POLICY_DIR}):
    python stage2-go2-mujoco-inference/experiments/harness.py
    python stage2-go2-mujoco-inference/experiments/exp1_velocity_sweep.py

BANNER
fi

exec "$@"
