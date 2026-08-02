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
handles, and then pushes past the boundary. Levels 0-3 (flat, incl. a running-
speed level) reproduce or exceed the baseline's competence; levels 4+ are
where a new policy has to beat it on terrain the baseline cannot handle. That
makes
progress measurable against a real number instead of a vibe.

Promotion rule (two gates — both required on flat levels):

  1. Recent mean normalized return >= `promote_threshold`.
  2. Recent mean body-frame forward speed OR distance per episode proves the
     robot is actually locomoting, not standing/crouching for free return.

Return-only promotion is what let multigait_v4–v7 sit on `flat-slow` for 15M
steps: standing still scored high `alive` / ang_vel while body vx stayed ~0.

Demotion stays return-based so a promoted policy that collapses can step back.
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
    Level("flat-slow", "go2_flat.xml", (0.2, 0.5), (0.0, 0.0), (0.0, 0.0),
          "learn to stand and walk at all; vx >= 0.2 so movement can be taught"),
    Level("flat-fast", "go2_flat.xml", (0.0, 1.0), (-0.3, 0.3), (-0.5, 0.5),
          "full command range on flat ground"),
    Level("flat-run", "go2_flat.xml", (0.5, 2.5), (-0.3, 0.3), (-0.5, 0.5),
          "running speed regime — pushes past walk-these-ways' measured "
          "~0.55 m/s ceiling (Exp 1 F1.2), commanded, not just fast trot"),
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
    Level("obstacles-easy", "go2_obstacles_easy.xml", (0.0, 0.75),
          (-0.3, 0.3), (-0.4, 0.4),
          "free-world scattered obstacles, 12 boxes 10-20cm — go around or "
          "step over, blind (no perception input, proprioception only)"),
    Level("obstacles-hard", "go2_obstacles_hard.xml", (0.0, 0.75),
          (-0.3, 0.3), (-0.4, 0.4),
          "24 boxes 15-35cm, denser field"),
    Level("gauntlet", "go2_gauntlet.xml", (0.0, 0.75), (-0.3, 0.3), (-0.4, 0.4),
          "5cm stairs directly into a 30-obstacle field, no recovery gap — "
          "the walk-these-ways baseline safe-stalls at the very first step "
          "(verified: 1.34m in 25s, height never drops). The hardest level."),
]

# Index of the first level the flat-trained baseline cannot clear. Beating this
# is the headline claim a Stage 3 policy would be making. Bumped 3 -> 4 when
# flat-run was inserted before it; slope-15 is still the actual boundary.
BASELINE_CEILING = 4


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
    # Movement gates — must pass in addition to return before promoting off flat.
    promote_min_body_vx: float = 0.15   # m/s, body frame, episode mean
    promote_min_distance_m: float = 1.5   # world-frame travel per episode
    window: int = 20
    _returns: list = field(default_factory=list)
    _body_vxs: list = field(default_factory=list)
    _distances: list = field(default_factory=list)

    @property
    def level(self) -> Level:
        return LEVELS[self.level_idx]

    @property
    def max_idx(self) -> int:
        return len(LEVELS) - 1

    def record(self, episode_return, max_possible_return,
               distance_m=0.0, mean_body_vx=0.0):
        """
        Log one finished episode and maybe change level.
        Returns a string describing any change, else None.
        """
        if not self.enabled:
            return None

        self._returns.append(episode_return / max(max_possible_return, 1e-6))
        self._body_vxs.append(float(mean_body_vx))
        self._distances.append(float(distance_m))
        if len(self._returns) < self.window:
            return None

        recent = self._returns[-self.window:]
        score = sum(recent) / len(recent)
        mean_vx = sum(self._body_vxs[-self.window:]) / self.window
        mean_dist = sum(self._distances[-self.window:]) / self.window
        moving = (mean_vx >= self.promote_min_body_vx
                  or mean_dist >= self.promote_min_distance_m)

        if (score >= self.promote_threshold and moving
                and self.level_idx < self.max_idx):
            self.level_idx += 1
            self._returns.clear()
            self._body_vxs.clear()
            self._distances.clear()
            return (f"PROMOTE -> {self.level.name} ({self.level.note}) "
                    f"[vx={mean_vx:.2f} m/s dist={mean_dist:.1f} m]")

        if score <= self.demote_threshold and self.level_idx > 0:
            self.level_idx -= 1
            self._returns.clear()
            self._body_vxs.clear()
            self._distances.clear()
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
        return {
            "level_idx": self.level_idx,
            "returns": list(self._returns),
            "body_vxs": list(self._body_vxs),
            "distances": list(self._distances),
        }

    def load_state_dict(self, state):
        self.level_idx = state.get("level_idx", 0)
        self._returns = list(state.get("returns", []))
        self._body_vxs = list(state.get("body_vxs", []))
        self._distances = list(state.get("distances", []))
