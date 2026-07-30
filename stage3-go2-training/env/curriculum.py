"""
Terrain and velocity curriculum.

The schedule is not guesswork — it is read directly off Experiment 3's measured
failure boundaries for the flat-trained baseline:

    slopes  100% survival at 5 deg and 10 deg
             20% survival at 15 deg
              0% survival at >= 20 deg      -> boundary is 10-15 deg
    stairs  traversable at 2 cm (marginally)
            "safe stall" at >= 5 cm         -> boundary is 2-5 cm

So the curriculum starts at flat, promotes through terrain the baseline already
handles, and then pushes past the boundary. Levels 0-2 reproduce the baseline's
competence; levels 3+ are where a new policy has to beat it. That makes
progress measurable against a real number instead of a vibe.

Promotion rule: a level is cleared when the agent's recent mean episode return
exceeds `promote_threshold` * the return achievable at that level. Demotion on
sustained failure prevents the classic curriculum failure mode, where an agent
is pushed past its competence and collapses with no way back.
"""

from dataclasses import dataclass, field


@dataclass
class Level:
    name: str
    scene: str
    # Commanded forward velocity range sampled per episode.
    vx_range: tuple = (0.0, 0.5)
    vy_range: tuple = (0.0, 0.0)
    yaw_range: tuple = (0.0, 0.0)
    note: str = ""


# Ordered easiest -> hardest. Scenes are the ones committed in
# stage2-go2-mujoco-inference/scenes/, so terrain is identical to Experiment 3
# and the results are directly comparable.
LEVELS = [
    Level("flat-slow", "go2_flat.xml", (0.0, 0.5), (0.0, 0.0), (0.0, 0.0),
          "learn to stand and walk at all"),
    Level("flat-fast", "go2_flat.xml", (0.0, 1.0), (-0.3, 0.3), (-0.5, 0.5),
          "full command range on flat ground"),
    Level("slope-10", "go2_slope_10.xml", (0.0, 0.75), (-0.2, 0.2), (-0.3, 0.3),
          "baseline manages this: 100% survival (Exp 3)"),
    Level("slope-15", "go2_slope_15.xml", (0.0, 0.75), (-0.2, 0.2), (-0.3, 0.3),
          "baseline BREAKS here: 20% survival (Exp 3)"),
    Level("stairs-5", "go2_stairs_05.xml", (0.0, 0.5), (0.0, 0.0), (-0.2, 0.2),
          "baseline safe-stalls here: 0.47 m, never climbs (Exp 3 F3.4)"),
    Level("slope-20", "go2_slope_20.xml", (0.0, 0.5), (-0.2, 0.2), (-0.3, 0.3),
          "baseline: 0% survival (Exp 3)"),
    Level("stairs-8", "go2_stairs_08.xml", (0.0, 0.5), (0.0, 0.0), (-0.2, 0.2),
          "well beyond baseline capability"),
]

# Index of the first level the flat-trained baseline cannot clear. Beating this
# is the headline claim a Stage 3 policy would be making.
BASELINE_CEILING = 3


@dataclass
class Curriculum:
    """
    Tracks the current level and decides promotion/demotion.

    `enabled=False` pins level 0, which is what you want when debugging the
    reward function — a moving terrain distribution makes reward bugs
    impossible to isolate.
    """

    enabled: bool = True
    level_idx: int = 0
    promote_threshold: float = 0.75
    demote_threshold: float = 0.30
    window: int = 20
    _returns: list = field(default_factory=list)

    @property
    def level(self) -> Level:
        return LEVELS[self.level_idx]

    @property
    def max_idx(self) -> int:
        return len(LEVELS) - 1

    def record(self, episode_return, max_possible_return):
        """
        Log one finished episode and maybe change level.
        Returns a string describing any change, else None.
        """
        if not self.enabled:
            return None

        self._returns.append(episode_return / max(max_possible_return, 1e-6))
        if len(self._returns) < self.window:
            return None

        recent = self._returns[-self.window:]
        score = sum(recent) / len(recent)

        if score >= self.promote_threshold and self.level_idx < self.max_idx:
            self.level_idx += 1
            self._returns.clear()
            return f"PROMOTE -> {self.level.name} ({self.level.note})"

        if score <= self.demote_threshold and self.level_idx > 0:
            self.level_idx -= 1
            self._returns.clear()
            return f"demote -> {self.level.name}"

        return None

    def sample_command(self, rng):
        """Sample (vx, vy, yaw) from the current level's ranges."""
        lvl = self.level
        return (
            float(rng.uniform(*lvl.vx_range)),
            float(rng.uniform(*lvl.vy_range)),
            float(rng.uniform(*lvl.yaw_range)),
        )

    def state_dict(self):
        return {"level_idx": self.level_idx, "returns": list(self._returns)}

    def load_state_dict(self, state):
        self.level_idx = state.get("level_idx", 0)
        self._returns = list(state.get("returns", []))
