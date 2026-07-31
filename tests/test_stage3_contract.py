"""
Stage 3 must honour the Stage 2 contract.

The single requirement that makes Stage 3 worth building on top of Stage 2:
a policy trained here must be evaluable by harness.py with ZERO modifications.
If that holds, the new policy is directly comparable against the
walk-these-ways baseline on the same scenes with the same metrics. If it
breaks, every measurement Stage 2 produced becomes useless for comparison.

These tests need no trained checkpoint — they exercise the architecture, the
export path and the environment's observation construction, all of which are
deterministic given the code.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE3 = REPO_ROOT / "stage3-go2-training"
sys.path.insert(0, str(STAGE3))

import networks  # noqa: E402
from networks import (  # noqa: E402
    ACTION_DIM,
    BODY_INPUT_DIM,
    HISTORY_DIM,
    HISTORY_LEN,
    LATENT_DIM,
    OBS_DIM,
    AdaptationModule,
    Body,
    Go2Agent,
    assert_contract,
)


# ===========================================================================
# The numbers
# ===========================================================================
def test_stage3_constants_match_stage2():
    """
    Stage 3 declares OBS_DIM/HISTORY_LEN independently (it must not import
    MuJoCo just to know its own shapes). Those declarations must agree with
    Stage 2's, or the exported policy will not load.
    """
    import harness

    assert networks.OBS_DIM == harness.OBS_DIM
    assert networks.HISTORY_LEN == harness.HISTORY_LEN


def test_contract_dimensions():
    assert OBS_DIM == 70
    assert HISTORY_LEN == 30
    assert HISTORY_DIM == 2100
    assert LATENT_DIM == 2
    assert BODY_INPUT_DIM == 2102
    assert ACTION_DIM == 12


def test_privileged_dim_matches_domain_rand():
    """networks.PRIV_DIM must equal the number of randomised parameters."""
    from env.domain_rand import PARAM_SPECS

    assert networks.PRIV_DIM == len(PARAM_SPECS), (
        f"networks.PRIV_DIM={networks.PRIV_DIM} but domain_rand defines "
        f"{len(PARAM_SPECS)} parameters — the privileged encoder would "
        "receive the wrong input width"
    )


# ===========================================================================
# Network shapes
# ===========================================================================
def test_adaptation_module_shape():
    net = AdaptationModule().eval()
    with torch.no_grad():
        out = net(torch.zeros(4, HISTORY_DIM))
    assert tuple(out.shape) == (4, LATENT_DIM)


def test_body_shape():
    net = Body().eval()
    with torch.no_grad():
        out = net(torch.zeros(4, BODY_INPUT_DIM))
    assert tuple(out.shape) == (4, ACTION_DIM)


def test_assert_contract_passes_on_fresh_networks():
    assert assert_contract(AdaptationModule(), Body()) is True


def test_assert_contract_rejects_wrong_output_width():
    """The guard must actually catch a violation, not just always pass."""
    class WrongLatent(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 5)  # 5 != LATENT_DIM

    with pytest.raises(ValueError, match="adaptation_module"):
        assert_contract(WrongLatent(), Body())


def test_assert_contract_rejects_wrong_action_width():
    class WrongAction(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 7)  # 7 != ACTION_DIM

    with pytest.raises(ValueError, match="body"):
        assert_contract(AdaptationModule(), WrongAction())


def test_deployment_path_matches_manual_composition():
    """
    `act_deployed` must do exactly what harness.py does by hand:
        latent = adapt(history); action = body(cat([history, latent]))
    If these diverge, training optimises something the robot never runs.
    """
    torch.manual_seed(0)
    agent = Go2Agent().eval()
    history = torch.randn(2, HISTORY_DIM)

    with torch.no_grad():
        via_helper = agent.act_deployed(history)
        latent = agent.adaptation_module(history)
        via_manual = agent.body(torch.cat([history, latent], dim=1))

    torch.testing.assert_close(via_helper, via_manual)


# ===========================================================================
# Export
# ===========================================================================
@pytest.fixture(scope="module")
def exported_policy(tmp_path_factory):
    """Export a randomly initialised, contract-valid policy once per module."""
    from export import export

    torch.manual_seed(0)
    out = tmp_path_factory.mktemp("exported")
    return export(AdaptationModule(), Body(), out, {"trained": False})


def test_export_writes_the_expected_filenames(exported_policy):
    """Stage 2 looks for these exact names — see paths.BODY_JIT/ADAPT_JIT."""
    import paths

    assert (exported_policy / paths.BODY_JIT).is_file()
    assert (exported_policy / paths.ADAPT_JIT).is_file()
    assert (exported_policy / "export_metadata.json").is_file()


def test_exported_policy_loads_via_stage2_paths(exported_policy, monkeypatch):
    """
    The real integration point: `paths.load_policy()` — the function every
    Stage 2 script uses — must load a Stage 3 export without complaint.
    """
    import importlib
    import paths

    monkeypatch.setenv("GO2_POLICY_DIR", str(exported_policy))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.policy_available()
        body_net, adapt_net = reloaded.load_policy()

        with torch.no_grad():
            history = torch.zeros(1, HISTORY_DIM)
            latent = adapt_net(history)
            assert tuple(latent.shape) == (1, LATENT_DIM)
            action = body_net(torch.cat([history, latent], dim=1))
            assert tuple(action.shape) == (1, ACTION_DIM)
    finally:
        monkeypatch.delenv("GO2_POLICY_DIR", raising=False)
        importlib.reload(paths)


@pytest.mark.slow
def test_exported_policy_runs_through_stage2_harness(exported_policy):
    """
    THE test. Run a Stage 3 export through Stage 2's run_trial() unmodified
    and confirm it returns a well-formed metrics record.

    An untrained policy will not walk — that is fine and expected. What is
    being verified is that the interface holds end to end: observation
    construction, history buffering, adaptation module, body network, action
    scaling, PD control and metric computation.
    """
    from harness import run_trial

    scene = str((REPO_ROOT / "stage2-go2-mujoco-inference" / "scenes"
                 / "go2_flat.xml").resolve())
    body_net = torch.jit.load(str(exported_policy / "body_latest.jit")).eval()
    adapt_net = torch.jit.load(
        str(exported_policy / "adaptation_module_latest.jit")).eval()

    result = run_trial(scene, lin_vel_x=0.5, gait="trot",
                       settle_s=0.5, measure_s=1.0,
                       body_net=body_net, adapt_net=adapt_net, seed=0)

    for key in ("scene", "cmd_vx", "gait", "seed", "fell"):
        assert key in result, f"harness did not return '{key}'"
    assert result["cmd_vx"] == 0.5
    assert isinstance(result["fell"], bool)
    if not result["fell"]:
        assert result["mean_vx"] is not None
        assert np.isfinite(result["mean_vx"])
        assert result["height_mean"] is not None


# ===========================================================================
# Environment
# ===========================================================================
@pytest.fixture(scope="module")
def env():
    from config import Config
    from env import Go2Env

    return Go2Env(Config(episode_seconds=1.0, domain_rand=True), seed=0)


def test_env_observation_is_the_contract_width(env):
    obs, priv = env.reset()
    assert obs.shape == (HISTORY_DIM,), (
        f"env emits {obs.shape}, contract requires ({HISTORY_DIM},)"
    )
    assert priv.shape == (networks.PRIV_DIM,)
    assert obs.dtype == np.float32


def test_env_single_frame_is_70_dims(env):
    env.reset()
    assert len(env.obs_history) == HISTORY_LEN
    for frame in env.obs_history:
        assert frame.shape == (OBS_DIM,)


def test_env_step_returns_contract_shapes(env):
    env.reset()
    obs, priv, reward, term, trunc, info = env.step(np.zeros(ACTION_DIM))
    assert obs.shape == (HISTORY_DIM,)
    assert priv.shape == (networks.PRIV_DIM,)
    assert np.isfinite(reward)
    assert isinstance(term, (bool, np.bool_))
    assert isinstance(trunc, (bool, np.bool_))


def test_env_uses_stage2_control_constants(env):
    """
    The env imports its constants from harness.py rather than redeclaring
    them. This test proves the import actually took effect.
    """
    import harness
    from env import go2_env

    assert go2_env.DECIMATION == harness.DECIMATION
    assert go2_env.ACTION_SCALE == harness.ACTION_SCALE
    assert go2_env.KP == harness.KP and go2_env.KD == harness.KD
    np.testing.assert_allclose(go2_env.DEFAULT_JOINT_POS,
                               harness.DEFAULT_JOINT_POS)
    assert env.dt_policy == pytest.approx(0.02), "policy must run at 50 Hz"


def test_env_hip_action_scale_matches_stage2(env):
    """Hips get 0.125, everything else 0.25 — same as the deployed pipeline."""
    for hip in (0, 3, 6, 9):
        assert env.action_scale[hip] == pytest.approx(0.125)
    for other in (1, 2, 4, 5, 7, 8, 10, 11):
        assert env.action_scale[other] == pytest.approx(0.25)


def test_env_zero_action_holds_the_pose(env):
    """
    A zero action means "the default pose". The robot must stay standing for
    a second under PD control — if this fails, the control loop is wrong.
    """
    env.reset(command=(0.0, 0.0, 0.0))
    for _ in range(50):  # 1 s at 50 Hz
        _, _, _, term, trunc, info = env.step(np.zeros(ACTION_DIM))
        if term or trunc:
            break
    assert not term, f"robot fell holding the default pose (h={info['height']})"
    assert info["height"] > 0.15


def test_env_termination_matches_harness_criteria():
    """
    Training and evaluation must agree on what "fell" means, or a policy will
    be optimised against a different survival bar than it is scored on.
    """
    import harness
    from env import go2_env

    assert go2_env.FALL_HEIGHT_THRESHOLD == harness.FALL_HEIGHT_THRESHOLD
    assert go2_env.FALL_TILT_THRESHOLD_DEG == harness.FALL_TILT_THRESHOLD_DEG


def test_env_is_deterministic_given_a_seed():
    """Same seed, same trajectory — required for reproducible experiments."""
    from config import Config
    from env import Go2Env

    def rollout(seed):
        e = Go2Env(Config(episode_seconds=1.0), seed=seed)
        e.reset(command=(0.5, 0.0, 0.0))
        rewards = []
        for _ in range(20):
            _, _, r, term, trunc, _ = e.step(np.full(ACTION_DIM, 0.1))
            rewards.append(r)
            if term or trunc:
                break
        return rewards

    np.testing.assert_allclose(rollout(7), rollout(7))


# ===========================================================================
# Curriculum and domain randomisation
# ===========================================================================
def test_curriculum_levels_reference_real_scenes():
    from env.curriculum import LEVELS

    scenes = REPO_ROOT / "stage2-go2-mujoco-inference" / "scenes"
    for level in LEVELS:
        assert (scenes / level.scene).is_file(), (
            f"curriculum level '{level.name}' references a missing scene: "
            f"{level.scene}"
        )


def test_curriculum_starts_flat_and_escalates():
    from env.curriculum import BASELINE_CEILING, LEVELS

    assert LEVELS[0].scene == "go2_flat.xml"
    assert 0 < BASELINE_CEILING < len(LEVELS)
    # Everything from BASELINE_CEILING on must be terrain, not flat ground.
    for level in LEVELS[BASELINE_CEILING:]:
        assert level.scene != "go2_flat.xml"


def test_curriculum_promotes_and_demotes():
    from env.curriculum import Curriculum

    c = Curriculum(window=5, promote_threshold=0.75, demote_threshold=0.30)
    assert c.level_idx == 0
    for _ in range(5):
        c.record(90.0, 100.0)          # 0.9 >= promote threshold
    assert c.level_idx == 1

    for _ in range(5):
        c.record(10.0, 100.0)          # 0.1 <= demote threshold
    assert c.level_idx == 0


def test_curriculum_disabled_pins_level_zero():
    from env.curriculum import Curriculum

    c = Curriculum(enabled=False, window=2)
    for _ in range(20):
        c.record(100.0, 100.0)
    assert c.level_idx == 0


def test_privileged_vector_is_normalised():
    """Raw ranges span 0.3 to 3.0; unnormalised, mass would dominate."""
    from env.domain_rand import DomainParams

    rng = np.random.default_rng(0)
    for _ in range(50):
        vec = DomainParams.sample(rng).as_privileged_vector()
        assert vec.shape == (networks.PRIV_DIM,)
        assert np.all(np.abs(vec) <= 1.0 + 1e-6), f"out of [-1, 1]: {vec}"


def test_nominal_domain_params_are_neutral():
    """Nominal friction is mid-ish; the point is it must be reproducible."""
    from env.domain_rand import DomainParams

    a = DomainParams.nominal().as_privileged_vector()
    b = DomainParams.nominal().as_privileged_vector()
    np.testing.assert_array_equal(a, b)


# ===========================================================================
# End-to-end
# ===========================================================================
@pytest.mark.slow
def test_training_smoke_run_completes():
    """
    Run train.py --smoke as a subprocess: both RMA phases, checkpointing and
    the final contract check. Catches import errors, shape bugs and optimiser
    misconfiguration that unit tests miss.

    Deliberately run from the REPO ROOT, not from STAGE3. A relative out_dir
    resolved against the CWD would scatter checkpoints into ./runs, where
    export.py's documented command cannot find them; the assertion below only
    means something because the CWD is somewhere else.
    """
    result = subprocess.run(
        [sys.executable, str(STAGE3 / "train.py"), "--smoke",
         "--run-name", "pytest_smoke"],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=900,
    )
    assert result.returncode == 0, (
        f"smoke training failed:\nSTDOUT:\n{result.stdout[-3000:]}\n"
        f"STDERR:\n{result.stderr[-3000:]}"
    )
    assert "contract check: PASS" in result.stdout
    assert (STAGE3 / "runs" / "pytest_smoke" / "checkpoint_final.pt").is_file()
