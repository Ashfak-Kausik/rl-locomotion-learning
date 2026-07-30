# =============================================================================
# rl-locomotion-learning — convenience targets
# =============================================================================
# `make help` lists everything. Targets marked [no-policy] work without the
# walk-these-ways checkpoints; see docs/DEPENDENCIES.md.
# =============================================================================

VENV        := .venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
STAGE1      := stage1-rl-fundamentals
STAGE2      := stage2-go2-mujoco-inference
EXP         := $(STAGE2)/experiments
STAGE3      := stage3-go2-training
RUN         ?= v1
COMPOSE     := docker compose -f docker/compose.yaml
DOCKER_ENV  := UID=$(shell id -u) GID=$(shell id -g)

.DEFAULT_GOAL := help
.PHONY: help setup check test test-all clean clean-all \
        train train-smoke export evaluate \
        figures scenes hello inspect pose walk teleop \
        cartpole lunarlander pendulum tensorboard \
        exp1 exp2 exp3 experiments \
        export-random docker-build docker-shell docker-check docker-figures docker-viewer \
        compile lint-imports

# --- meta --------------------------------------------------------------------

help:  ## Show this help
	@echo ""
	@echo "  rl-locomotion-learning"
	@echo "  ─────────────────────────────────────────────────────────────────"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  [no-policy] targets need no walk-these-ways checkpoints."
	@echo ""

# --- environment -------------------------------------------------------------

setup:  ## Install everything (native pkgs + venv + python deps), then verify
	./scripts/setup_env.sh

check:  ## Verify the environment — 4-layer report [no-policy]
	@$(PY) scripts/check_env.py 2>/dev/null || python3 scripts/check_env.py

test:  ## Run the test suite, skipping slow tests [no-policy]
	$(PY) -m pytest -m "not slow"

test-all:  ## Run every test including the training smoke run [no-policy]
	$(PY) -m pytest

compile:  ## Syntax-check every Python file [no-policy]
	@$(PY) -m py_compile $(STAGE1)/*.py $(STAGE2)/*.py $(EXP)/*.py \
	  $(STAGE3)/*.py $(STAGE3)/env/*.py scripts/*.py tests/*.py \
	  && echo "all files compile"

# --- Stage 2: no policy weights required -------------------------------------

figures:  ## Regenerate all 4 data figures from committed CSVs [no-policy]
	$(PY) $(EXP)/make_figures.py

scenes:  ## Regenerate the 10 terrain scenes; diff must be empty [no-policy]
	$(PY) $(EXP)/generate_terrain_scenes.py
	@git diff --stat $(STAGE2)/scenes/ \
	  && echo "(empty diff above = byte-identical regeneration)"

hello:  ## Open the Go2 in the MuJoCo viewer [no-policy]
	$(PY) $(STAGE2)/01_hello_go2.py

inspect:  ## Print the Go2 model structure [no-policy]
	$(PY) $(STAGE2)/02_inspect_go2.py

pose:  ## Hold a standing pose with a hand-written PD controller [no-policy]
	$(PY) $(STAGE2)/03_pose_go2.py

# --- Stage 2: policy weights required ----------------------------------------

walk:  ## Run the pretrained policy — the robot walks
	$(PY) $(STAGE2)/06_run_policy.py

teleop:  ## Interactive keyboard control with gait switching
	$(PY) $(STAGE2)/07_run_policy_interactive.py

exp1:  ## Experiment 1 — velocity sweep (6 x 5 trials)
	cd $(EXP) && ../../$(PY) exp1_velocity_sweep.py

exp2:  ## Experiment 2 — gait robustness (3 x 5 trials)
	cd $(EXP) && ../../$(PY) exp2_gait_robustness.py

exp3:  ## Experiment 3 — terrain robustness (10 x 5 trials)
	cd $(EXP) && ../../$(PY) exp3_terrain.py

experiments: exp1 exp2 exp3 figures  ## Run all three experiments, then figures

# --- Stage 3: custom policy training [no-policy weights needed] ---------------

train-smoke:  ## 30 s end-to-end training sanity check [no-policy]
	cd $(STAGE3) && ../$(PY) train.py --smoke

train:  ## Train a policy (long; see stage3-go2-training/README.md) [no-policy]
	cd $(STAGE3) && ../$(PY) train.py --run-name $(RUN)

export:  ## Export a checkpoint to TorchScript in the Stage 2 contract
	$(PY) $(STAGE3)/export.py --checkpoint $(STAGE3)/runs/$(RUN)/checkpoint_final.pt

export-random:  ## Export untrained but contract-valid nets (pipeline test) [no-policy]
	$(PY) $(STAGE3)/export.py --random

evaluate:  ## Compare an exported policy against the baseline via Stage 2's harness
	$(PY) $(STAGE3)/evaluate.py --terrain

# --- Stage 1 ------------------------------------------------------------------

cartpole:  ## Train PPO on CartPole-v1 (~2 min, CPU) [no-policy]
	cd $(STAGE1) && ../$(PY) 01_cartpole_ppo.py

lunarlander:  ## Train PPO on LunarLander-v3 (~15 min, CPU) [no-policy]
	cd $(STAGE1) && ../$(PY) 03_lunarlander_ppo.py

pendulum:  ## Train PPO on Pendulum-v1 (~10 min, CPU) [no-policy]
	cd $(STAGE1) && ../$(PY) 05_pendulum_ppo.py

tensorboard:  ## Serve training curves at http://localhost:6006 [no-policy]
	$(VENV)/bin/tensorboard --logdir $(STAGE1)/tb_logs

# --- Docker -------------------------------------------------------------------

docker-build:  ## Build the CPU image (~2 GB)
	$(DOCKER_ENV) $(COMPOSE) build

docker-shell:  ## Interactive shell in the container
	$(DOCKER_ENV) $(COMPOSE) run --rm lab

docker-check:  ## Environment report inside the container [no-policy]
	$(DOCKER_ENV) $(COMPOSE) run --rm headless python scripts/check_env.py

docker-figures:  ## Regenerate figures inside the container [no-policy]
	$(DOCKER_ENV) $(COMPOSE) run --rm headless \
	  python $(EXP)/make_figures.py

docker-viewer:  ## MuJoCo GUI via host X11 (run `xhost +local:docker` first)
	$(DOCKER_ENV) $(COMPOSE) run --rm viewer

# --- housekeeping -------------------------------------------------------------

clean:  ## Remove bytecode and MuJoCo logs
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -f MUJOCO_LOG.TXT
	@echo "cleaned"

clean-all: clean  ## Also remove the virtualenv
	rm -rf $(VENV)
	@echo "removed $(VENV)"
