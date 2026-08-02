"""
PPO training for Go2 locomotion — RMA, both phases.

    Phase 1  PPO trains {privileged_encoder, body, critic}. The latent comes
             from ground-truth simulation parameters the real robot could
             never measure.

    Phase 2  Supervised regression: the adaptation module learns to predict
             that latent from proprioceptive history alone. This is the
             network that actually deploys.

Only phase 2's adaptation module and phase 1's body are exported (export.py).

Checkpointing
-------------
Free-tier GPU sessions get killed without warning, so state is written every
`save_every_updates` and `--resume` restores optimiser state, the RNG, the
curriculum level and the timestep counter. Resuming is the normal case here,
not an edge case.

Usage
-----
    python train.py --timesteps 200000 --run-name my_run
    python train.py --resume runs/my_run/checkpoint_latest.pt
    python train.py --smoke                  # 30 s end-to-end sanity run
    python train.py --no-curriculum --no-domain-rand   # debug the reward
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, resolve_device, tune_for_device  # noqa: E402
from env import Curriculum, Go2Env     # noqa: E402
from networks import (                 # noqa: E402
    ACTION_DIM,
    HISTORY_DIM,
    Go2Agent,
    assert_contract,
)


# ===========================================================================
# Rollout storage
# ===========================================================================
class RolloutBuffer:
    """Flat storage for one PPO update. Shapes are (steps, envs, ...)."""

    def __init__(self, steps, num_envs, device):
        self.obs = torch.zeros(steps, num_envs, HISTORY_DIM, device=device)
        self.priv = torch.zeros(steps, num_envs, 8, device=device)
        self.actions = torch.zeros(steps, num_envs, ACTION_DIM, device=device)
        self.logprobs = torch.zeros(steps, num_envs, device=device)
        self.rewards = torch.zeros(steps, num_envs, device=device)
        self.dones = torch.zeros(steps, num_envs, device=device)
        self.values = torch.zeros(steps, num_envs, device=device)
        self.steps = steps
        self.num_envs = num_envs
        self.device = device

    def compute_gae(self, last_value, gamma, gae_lambda):
        """
        Generalised Advantage Estimation.

        `dones` marks TERMINATION (a fall), not truncation. Bootstrapping is
        cut only on real terminals — cutting it on time-limit truncation would
        teach the policy that the episode genuinely ends at 10 s, which biases
        the value function badly.
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_nonterminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            delta = (self.rewards[t]
                     + gamma * next_value * next_nonterminal
                     - self.values[t])
            last_gae = (delta
                        + gamma * gae_lambda * next_nonterminal * last_gae)
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def flat(self, advantages, returns):
        n = self.steps * self.num_envs
        return (self.obs.reshape(n, -1),
                self.priv.reshape(n, -1),
                self.actions.reshape(n, -1),
                self.logprobs.reshape(n),
                advantages.reshape(n),
                returns.reshape(n),
                self.values.reshape(n))


# ===========================================================================
# Trainer
# ===========================================================================
def _resolve_rollout_workers(requested, num_envs):
    """
    0 / None → auto = min(num_envs, cpu_count).
    1 → serial (legacy path, useful for debugging race hypotheses).
    N → clamp to [1, num_envs].
    """
    if requested is None or requested == 0:
        cpus = os.cpu_count() or 1
        return max(1, min(num_envs, cpus))
    return max(1, min(int(requested), num_envs))


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Resolve "auto" to a real device, say why, and scale the PPO update
        # to match it. Never silently falls back to a 40x slower run.
        self.device = resolve_device(cfg.device)
        tune_for_device(cfg, self.device, profile_name=cfg.tune_profile or None)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # A relative out_dir is anchored to this stage directory, not to the
        # shell's CWD — otherwise `python stage3-go2-training/train.py` from
        # the repo root scatters checkpoints where export.py will not find
        # them. Matches how export.py anchors its policies/ default.
        out_dir = Path(cfg.out_dir)
        if not out_dir.is_absolute():
            out_dir = Path(__file__).resolve().parent / out_dir
        self.run_dir = out_dir / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.curriculum = Curriculum(
            enabled=cfg.curriculum,
            promote_threshold=cfg.promote_threshold,
            demote_threshold=cfg.demote_threshold,
            promote_min_body_vx=cfg.promote_min_body_vx,
            promote_min_distance_m=cfg.promote_min_distance_m,
            window=cfg.curriculum_window,
        )

        # Sequential MuJoCo instances. Plain MuJoCo has no batched stepping;
        # this is the honest CPU implementation. The MJX/GPU path replaces
        # exactly this list with a vmapped batch (see README).
        self.envs = [
            Go2Env(cfg, seed=cfg.seed + i,
                   scene=self.curriculum.level.scene)
            for i in range(cfg.num_envs)
        ]

        # Rollout collection is CPU-bound on mj_step. Each env owns its own
        # MjModel/MjData, and mj_step releases the GIL, so a small thread
        # pool turns the serial for-loop into near-linear scaling across
        # cores (measured ~3.7x on an 8-core i7-9700K for the physics
        # alone). Curriculum / logging stay on the main thread.
        self._rollout_workers = _resolve_rollout_workers(
            cfg.rollout_workers, cfg.num_envs)
        self._rollout_pool = (
            ThreadPoolExecutor(max_workers=self._rollout_workers)
            if self._rollout_workers > 1 else None
        )
        if self._rollout_workers > 1:
            print(f"[rollout] threaded env stepping: "
                  f"{self._rollout_workers} workers × {cfg.num_envs} envs "
                  f"(mj_step releases the GIL; see docs/TRAINING-SPEED.md)")
        else:
            print("[rollout] serial env stepping "
                  "(--rollout-workers 1 to force, 0 = auto)")

        self.agent = Go2Agent(init_log_std=cfg.init_log_std).to(self.device)
        self.optimizer = torch.optim.Adam(
            [p for n, p in self.agent.named_parameters()
             if not n.startswith("adaptation_module")],
            lr=cfg.learning_rate, eps=1e-5,
        )
        self.adapt_optimizer = torch.optim.Adam(
            self.agent.adaptation_module.parameters(),
            lr=cfg.adapt_learning_rate,
        )

        self.global_step = 0
        self.update = 0
        self.episode_returns = []
        self.start_time = time.time()
        # Tracks the best recent-mean-return seen so far; checkpoint_best.pt
        # is only overwritten on improvement. checkpoint_latest.pt is
        # overwritten every save_every_updates regardless, so a late
        # divergence (PPO update that blows past target_kl -- see
        # ppo_update's per-minibatch guard) can otherwise destroy the only
        # saved copy of the run's actual best policy. Cheap insurance.
        self.best_mean_return = -float("inf")

        # Live env state
        self._obs = np.zeros((cfg.num_envs, HISTORY_DIM), dtype=np.float32)
        self._priv = np.zeros((cfg.num_envs, 8), dtype=np.float32)
        self._reset_all()

    def _reset_all(self):
        for i, env in enumerate(self.envs):
            cmd = self.curriculum.sample_command(env.rng)
            obs, priv = env.reset(scene=self.curriculum.level.scene,
                                  command=cmd)
            self._obs[i] = obs
            self._priv[i] = priv

    # ------------------------------------------------------------------
    # Phase 1 — PPO
    # ------------------------------------------------------------------
    def collect_rollout(self, buf: RolloutBuffer):
        for step in range(self.cfg.rollout_steps):
            obs_t = torch.as_tensor(self._obs, device=self.device)
            priv_t = torch.as_tensor(self._priv, device=self.device)

            with torch.no_grad():
                action, logprob, _, value = self.agent.act(obs_t, priv_t)

            buf.obs[step] = obs_t
            buf.priv[step] = priv_t
            buf.actions[step] = action
            buf.logprobs[step] = logprob
            buf.values[step] = value

            actions_np = action.cpu().numpy()
            results = self._step_envs(actions_np)

            curriculum_changed = False
            for i, (obs, priv, reward, term, trunc, info) in results:
                buf.rewards[step, i] = reward
                buf.dones[step, i] = float(term)  # NOT trunc — see compute_gae

                if term or trunc:
                    ep = info["episode"]
                    self.episode_returns.append(ep["r"])
                    change = self.curriculum.record(
                        ep["r"], ep["max_r"],
                        distance_m=ep.get("distance_m", 0.0),
                        mean_body_vx=ep.get("mean_body_vx", 0.0),
                    )
                    if change:
                        print(f"  [curriculum] {change}")
                        self._reset_all()
                        curriculum_changed = True
                        break
                    cmd = self.curriculum.sample_command(self.envs[i].rng)
                    obs, priv = self.envs[i].reset(
                        scene=self.curriculum.level.scene, command=cmd)

                self._obs[i] = obs
                self._priv[i] = priv

            # After a level change every env was reset; skip writing the
            # pre-reset obs from envs that had not yet been processed in
            # this step's result loop (they are stale).
            if curriculum_changed:
                pass

            self.global_step += self.cfg.num_envs

    def _step_envs(self, actions_np):
        """
        Step every env, optionally in parallel.

        Returns a list of (i, (obs, priv, reward, term, trunc, info)) sorted
        by env index so curriculum / buffer writes stay deterministic.
        """
        if self._rollout_pool is None:
            return [
                (i, self.envs[i].step(actions_np[i]))
                for i in range(self.cfg.num_envs)
            ]

        def _one(i):
            return i, self.envs[i].step(actions_np[i])

        return sorted(
            self._rollout_pool.map(_one, range(self.cfg.num_envs)),
            key=lambda pair: pair[0],
        )
    def ppo_update(self, buf: RolloutBuffer):
        cfg = self.cfg
        with torch.no_grad():
            obs_t = torch.as_tensor(self._obs, device=self.device)
            priv_t = torch.as_tensor(self._priv, device=self.device)
            _, _, _, last_value = self.agent.act(obs_t, priv_t)

        advantages, returns = buf.compute_gae(
            last_value, cfg.gamma, cfg.gae_lambda)
        b_obs, b_priv, b_act, b_logp, b_adv, b_ret, b_val = buf.flat(
            advantages, returns)

        n = b_obs.shape[0]
        indices = np.arange(n)
        clipfracs, kls, applied_kls = [], [], []
        pg_loss = v_loss = ent_loss = torch.tensor(0.0)
        n_minibatches = n_rejected = 0

        stop_early = False
        for _ in range(cfg.update_epochs):
            if stop_early:
                break
            np.random.shuffle(indices)
            for start in range(0, n, cfg.minibatch_size):
                mb = indices[start:start + cfg.minibatch_size]
                if len(mb) < 2:
                    continue
                n_minibatches += 1

                new_logp, entropy, new_val = self.agent.evaluate_actions(
                    b_obs[mb], b_priv[mb], b_act[mb])

                logratio = new_logp - b_logp[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    kls.append(approx_kl.item())
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float()
                        .mean().item())

                # Checked BEFORE applying this minibatch's gradient, not
                # after -- a bad minibatch found mid-epoch must not update
                # the network at all. The previous version only checked
                # kls[-1] once per epoch (the LAST minibatch of a shuffled
                # order, so it could miss a spike entirely) and only stopped
                # FUTURE epochs -- the epoch that overshot had already fully
                # applied its update. That let one bad minibatch push a
                # 15M-step run to catastrophic divergence (return -600 ->
                # -65,535, std frozen, over ~800k steps) with the guard never
                # firing. See EXPERIMENT_FINDINGS.md / stage3 README.
                #
                # kls (above) logs EVERY minibatch seen, applied or not, so
                # a rejected minibatch's kl still shows up in a naive mean --
                # a printed "kl 35" can look catastrophic even when nothing
                # that bad was ever applied. applied_kls tracks only the ones
                # that actually updated the network, which is what "did this
                # update destabilise the policy" should be judged on. Found
                # the hard way: a printed kl of 35 turned out to include
                # rejected-but-never-applied minibatches.
                # Never let an update apply NOTHING. If the very first
                # minibatch trips the guard, training silently stops while
                # still logging normally -- multigait_v4 sat at `rej 1/1`
                # for its last ~10M steps, applying zero gradients. That is
                # worse than one slightly-too-large step, because it is
                # invisible. The first minibatch's true KL is 0 by
                # construction anyway (same policy, same data); any non-zero
                # reading is numerical noise, which is exactly what the
                # tf32=False default addresses at source.
                if (cfg.target_kl and approx_kl.item() > cfg.target_kl
                        and applied_kls):
                    stop_early = True
                    n_rejected += 1
                    break
                applied_kls.append(approx_kl.item())

                # Advantage normalisation per minibatch.
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef,
                                          1 + cfg.clip_coef),
                ).mean()
                v_loss = 0.5 * ((new_val - b_ret[mb]) ** 2).mean()
                ent_loss = entropy.mean()

                loss = (pg_loss
                        - cfg.ent_coef * ent_loss
                        + cfg.vf_coef * v_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(),
                                         cfg.max_grad_norm)
                self.optimizer.step()

        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (np.nan if var_y == 0
                         else 1 - np.var(y_true - y_pred) / var_y)

        return {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": ent_loss.item(),
            # kl: mean of ONLY the minibatches that actually updated the
            # network -- this is what "did we take a bad step" should be
            # judged on. kl_max: worst kl seen among ALL minibatches this
            # update, applied or rejected -- how close to the edge things
            # got, even for a step that was correctly blocked.
            "approx_kl": float(np.mean(applied_kls)) if applied_kls else 0.0,
            "kl_max": float(np.max(kls)) if kls else 0.0,
            "n_rejected": n_rejected,
            "n_minibatches": n_minibatches,
            "clip_fraction": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_variance": float(explained_var),
            "std": float(self.agent.log_std.exp().mean().item()),
        }

    # ------------------------------------------------------------------
    # Phase 2 — adaptation module distillation
    # ------------------------------------------------------------------
    def train_adaptation_module(self, steps=None):
        """
        Regress adaptation_module(history) onto privileged_encoder(priv).

        Data is collected on-policy with the frozen phase-1 policy, so the
        history distribution matches what the deployed policy will actually
        see. Training it on off-distribution data is the classic way to get a
        module that works in the lab and fails on the robot.
        """
        cfg = self.cfg
        steps = steps or cfg.adapt_steps
        print(f"\n=== Phase 2: adaptation module ({steps:,} samples) ===")

        self.agent.privileged_encoder.eval()
        self.agent.body.eval()
        self.agent.adaptation_module.train()

        obs_buf, latent_buf = [], []
        collected = 0
        losses = []
        t0 = time.time()

        while collected < steps:
            obs_t = torch.as_tensor(self._obs, device=self.device)
            priv_t = torch.as_tensor(self._priv, device=self.device)

            with torch.no_grad():
                target_latent = self.agent.privileged_encoder(priv_t)
                action, _, _, _ = self.agent.act(obs_t, priv_t)

            obs_buf.append(obs_t.clone())
            latent_buf.append(target_latent.clone())
            collected += self.cfg.num_envs

            actions_np = action.cpu().numpy()
            for i, env in enumerate(self.envs):
                obs, priv, _, term, trunc, _ = env.step(actions_np[i])
                if term or trunc:
                    cmd = self.curriculum.sample_command(env.rng)
                    obs, priv = env.reset(
                        scene=self.curriculum.level.scene, command=cmd)
                self._obs[i] = obs
                self._priv[i] = priv

            # Train on the buffer once it is big enough, then clear it.
            buffered = sum(o.shape[0] for o in obs_buf)
            if buffered >= cfg.adapt_batch_size:
                obs_all = torch.cat(obs_buf)
                lat_all = torch.cat(latent_buf)
                pred = self.agent.adaptation_module(obs_all)
                loss = ((pred - lat_all) ** 2).mean()

                self.adapt_optimizer.zero_grad()
                loss.backward()
                self.adapt_optimizer.step()

                losses.append(loss.item())
                obs_buf, latent_buf = [], []

                if len(losses) % 50 == 0:
                    print(f"  {collected:>8,}/{steps:,} samples  "
                          f"mse={np.mean(losses[-50:]):.5f}")

        final = float(np.mean(losses[-20:])) if losses else float("nan")
        print(f"  done in {time.time()-t0:.1f}s | final MSE {final:.5f}")
        return final

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, tag="latest"):
        path = self.run_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "adapt_optimizer": self.adapt_optimizer.state_dict(),
            "global_step": self.global_step,
            "update": self.update,
            "curriculum": self.curriculum.state_dict(),
            "episode_returns": self.episode_returns[-200:],
            "best_mean_return": self.best_mean_return,
            "config": {k: v for k, v in self.cfg.__dict__.items()},
            "torch_rng": torch.get_rng_state(),
        }, path)
        return path

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(ckpt["agent"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.adapt_optimizer.load_state_dict(ckpt["adapt_optimizer"])
        self.global_step = ckpt["global_step"]
        self.update = ckpt["update"]
        self.curriculum.load_state_dict(ckpt["curriculum"])
        self.episode_returns = list(ckpt.get("episode_returns", []))
        self.best_mean_return = ckpt.get("best_mean_return", -float("inf"))
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        self._reset_all()
        print(f"resumed from {path} at step {self.global_step:,} "
              f"(update {self.update}, level '{self.curriculum.level.name}')")

    def init_weights(self, path):
        """
        Warm-start the policy/critic from another run WITHOUT restoring
        optimiser, RNG, curriculum or step counter.

        Use this after a reward rebalance: --resume would keep a value
        function trained under the old reward, which is how broken runs
        get laundered into "new" experiments. Loading only the agent
        transfers a walking gait and lets PPO re-fit the critic.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(ckpt["agent"])
        src_step = ckpt.get("global_step", None)
        step_s = f"{src_step:,}" if isinstance(src_step, int) else "?"
        print(f"warm-started agent from {path} "
              f"(source step {step_s}; optimiser/curriculum/step fresh)")
        self._reset_all()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        buf = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, self.device)
        steps_per_update = cfg.rollout_steps * cfg.num_envs

        print(f"\n=== Phase 1: PPO ({cfg.total_timesteps:,} timesteps) ===")
        print(f"{steps_per_update} steps/update -> "
              f"{cfg.total_timesteps // steps_per_update:,} updates\n")

        while self.global_step < cfg.total_timesteps:
            self.collect_rollout(buf)
            stats = self.ppo_update(buf)
            self.update += 1

            if self.update % cfg.log_every_updates == 0:
                recent = self.episode_returns[-20:]
                # Episodes are longer than one rollout, so no episode has
                # finished for the first few updates. Show "--" rather than
                # a nan that reads like a divergence bug.
                if recent:
                    mean_ret_val = float(np.mean(recent))
                    mean_ret = f"{mean_ret_val:>8.1f}"
                    # Require a decent-sized window before trusting it as
                    # "best" -- the first few episodes after a reset/resume
                    # are noisy and would otherwise falsely win.
                    if (len(recent) >= 20
                            and mean_ret_val > self.best_mean_return):
                        self.best_mean_return = mean_ret_val
                        self.save("best")
                else:
                    mean_ret = f"{'--':>8}"
                sps = self.global_step / max(time.time() - self.start_time, 1e-6)
                print(
                    f"upd {self.update:>5} | step {self.global_step:>9,} | "
                    f"ret {mean_ret} | "
                    f"kl {stats['approx_kl']:.4f} (max {stats['kl_max']:.2f}) | "
                    f"rej {stats['n_rejected']}/{stats['n_minibatches']} | "
                    f"clip {stats['clip_fraction']:.2f} | "
                    f"ev {stats['explained_variance']:>6.3f} | "
                    f"std {stats['std']:.3f} | "
                    f"lvl {self.curriculum.level.name} | "
                    f"{sps:.0f} sps"
                )

            if self.update % cfg.save_every_updates == 0:
                self.save("latest")

        self.save("phase1")
        print(f"\nPhase 1 complete at {self.global_step:,} steps.")

        adapt_mse = self.train_adaptation_module()
        self.save("latest")
        self.save("final")

        assert_contract(self.agent.adaptation_module, self.agent.body)
        print("\ncontract check: PASS "
              "(2100 -> 2, 2102 -> 12; harness.py can evaluate this policy)")

        summary = {
            "run_name": cfg.run_name,
            "global_step": self.global_step,
            "updates": self.update,
            "final_level": self.curriculum.level.name,
            "mean_return_last20": float(np.mean(self.episode_returns[-20:]))
                                  if self.episode_returns else None,
            "adaptation_mse": adapt_mse,
            "wall_time_s": round(time.time() - self.start_time, 1),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"\nNext:  python stage3-go2-training/export.py "
              f"--checkpoint {self.run_dir}/checkpoint_final.pt")
        return summary


# ===========================================================================
# CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="go2_ppo")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda", "xpu", "mps"],
                    help="auto (default) uses the GPU if one is usable, "
                         "and says so either way. xpu = Intel Arc, "
                         "mps = Apple Silicon")
    ap.add_argument("--minibatch-size", type=int, default=None)
    ap.add_argument("--tune-profile", default=None,
                    help="force a PPO sizing profile from tune_profiles.py "
                         "instead of detecting one "
                         "(list them: python tune_profiles.py --all)")
    ap.add_argument("--rollout-workers", type=int, default=None,
                    help="threads for env.step during rollout "
                         "(0=auto=min(num_envs,cpus), 1=serial). "
                         "mj_step releases the GIL; see docs/TRAINING-SPEED.md")
    ap.add_argument("--resume", default=None, metavar="CHECKPOINT.pt")
    ap.add_argument("--init-from", default=None, metavar="CHECKPOINT.pt",
                    help="warm-start agent weights only (not optimiser / "
                         "curriculum / step). Use after a reward change "
                         "instead of --resume")
    ap.add_argument("--flat-bootstrap", action="store_true",
                    help="phase-1 flat trot only: no curriculum, no domain "
                         "rand, fixed gait — use until diagnose shows walking")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="pin level 0 — use when debugging the reward function")
    ap.add_argument("--no-domain-rand", action="store_true")
    ap.add_argument("--fixed-gait", action="store_true",
                    help="disable per-episode gait randomisation (use --gait "
                         "or Config.gait = trot)")
    ap.add_argument("--adapt-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="~30 s end-to-end run: both phases, tiny budgets")
    args = ap.parse_args()

    cfg = Config(run_name=args.run_name, seed=args.seed, device=args.device)
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
        cfg.num_envs_override = args.num_envs
    if args.minibatch_size is not None:
        cfg.minibatch_size_override = args.minibatch_size
    if args.tune_profile is not None:
        cfg.tune_profile = args.tune_profile
    if args.rollout_workers is not None:
        cfg.rollout_workers = args.rollout_workers
    if args.adapt_steps is not None:
        cfg.adapt_steps = args.adapt_steps
    if args.no_curriculum:
        cfg.curriculum = False
    if args.no_domain_rand:
        cfg.domain_rand = False
    if args.fixed_gait:
        cfg.randomize_gait = False
        cfg.gait = "trot"
    if args.flat_bootstrap:
        cfg.curriculum = False
        cfg.domain_rand = False
        cfg.randomize_gait = False
        cfg.gait = "trot"
        print("FLAT BOOTSTRAP — flat trot only; no curriculum or domain rand.")
        print("  Run diagnose.py before any long budget. See docs/TRAINING-SPEED.md\n")

    if args.smoke:
        cfg.run_name = args.run_name if args.run_name != "go2_ppo" else "smoke"
        cfg.total_timesteps = 4096
        cfg.num_envs = 4
        cfg.rollout_steps = 32
        cfg.minibatch_size = 64
        cfg.update_epochs = 2
        cfg.adapt_steps = 2048
        cfg.episode_seconds = 4.0
        cfg.save_every_updates = 5
        print("SMOKE MODE — tiny budgets, verifies the pipeline end to end.\n")

    # Device resolution and the CPU-fallback warning are handled by
    # resolve_device() inside Trainer.__init__, which also diagnoses *why*
    # a GPU was unavailable and scales the PPO update to the device.
    print(cfg.describe())
    trainer = Trainer(cfg)
    if args.resume and args.init_from:
        raise SystemExit("use --resume OR --init-from, not both")
    if args.resume:
        trainer.load(args.resume)
    if args.init_from:
        trainer.init_weights(args.init_from)
    trainer.train()


if __name__ == "__main__":
    main()
