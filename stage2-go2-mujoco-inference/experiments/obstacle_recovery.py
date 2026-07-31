"""
Reactive obstacle negotiation: detect the stall, back off, steer around.

WHY
---
On `go2_obstacles_hard` the baseline walks ~5 m, drives into a box, rides up
onto it and high-centres -- belly resting on the obstacle, feet without
traction, forward speed collapsing 0.29 -> 0.02 m/s and never recovering.
It does not fall (height stays ~0.29 m); it is the same "safe stall" failure
mode Experiment 3 found on stairs (F3.4), just triggered by a box.

WHAT DOESN'T WORK
-----------------
Raising `footswing_height_cmd` (0.06 -> 0.20) or `body_height_cmd`
(0.0 -> 0.10) does NOT help -- measured, every combination still stalls at
4.6-5.0 m, and high footswing is marginally *worse*. Once the belly is
resting on the box no gait parameter recovers it, because the feet are off
the ground. The fix has to prevent the beaching, not cure it.

WHAT DOES
---------
A supervisory state machine that writes only `lin_vel_x` and `ang_vel_yaw`
-- command dimensions the policy already accepts. No retraining, no change
to the 70-dim observation contract, same spirit as the heading-hold fix in
Experiment 4.

    CRUISE   walk toward target_heading with heading-hold engaged.
             Forward speed below `stall_vx` for `stall_s` seconds -> BACKUP.
    BACKUP   drive backwards to disengage from the obstacle.
    TURN     rotate toward a detour heading, alternating left/right so a
             repeated failure explores both ways around.
    CRUISE   resumes on the detour heading, and only decays back toward
             straight once COMMIT_DIST from where it got stuck -- decaying
             immediately steers it straight back into the same box
             (measured: 10 consecutive recoveries all re-wedging at x=4.55).

DETECTION SENSITIVITY IS THE WHOLE BALL GAME
--------------------------------------------
Detecting late (stall_vx=0.08, i.e. "already stopped") is nearly useless --
by then it is beached and backing up cannot free it. Detecting early, while
it is merely *slowing* against the obstacle, avoids the beaching entirely:

    stall_vx  stall_s |  max_x on go2_obstacles_hard (60 s, cmd_vx 0.5)
        --        --  |   5.02   <- no recovery
       0.08       1.2 |   4.75   <- late detection, no better
       0.15       0.8 |  11.16
       0.20       0.8 |  10.96
       0.25       0.4 |  13.00   <- default here

Verified not to false-trigger: on `go2_flat`, all three settings give 0
recoveries and exactly the baseline 18.12 m. On `go2_obstacles_easy` (which
the baseline already clears) the cost is <=0.3 m.

CAVEAT: `run_trial`-style rollouts here are deterministic -- no
initial-condition randomisation -- so each number above is a single sample
on one committed obstacle layout. Treat the defaults as tuned-on-one-course,
not as validated across layouts.
"""

import numpy as np

DEFAULTS = dict(
    k_heading=1.5,
    max_yaw=0.6,
    stall_vx=0.25,      # m/s, body frame. Close to the ~0.29 it walks at:
                        # that is deliberate, it detects *slowing*, not
                        # stopping. See the sweep above.
    stall_s=0.4,
    backup_s=2.0,
    backup_vx=-0.4,
    turn_s=1.8,
    turn_yaw=0.6,
    detour_deg=60.0,
    return_rate=12.0,   # deg/s the detour decays back toward straight
    commit_dist=1.5,    # m from the stall point before decaying starts
)


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(quat):
    """Heading in radians about world vertical. 0 = facing +x."""
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class RecoveryController:
    """
    Supervisory controller producing (lin_vel_x, ang_vel_yaw) commands.

    Call `update(v_body_x, yaw, xy, policy_dt)` once per POLICY step (not per
    physics step) and write the returned commands into the command vector.

    `enabled=False` reduces this to plain heading-hold, which is what
    watch_walk.py / make_video.py do by default -- so the same code path
    serves both the with- and without-recovery conditions.
    """

    def __init__(self, enabled=True, **overrides):
        self.enabled = enabled
        self.p = {**DEFAULTS, **overrides}
        self.state = "CRUISE"
        self.state_t = 0.0
        self.stall_t = 0.0
        self.target_heading = 0.0
        self.turn_sign = 1
        self.stuck_xy = None
        self.n_recoveries = 0
        self.last_event = None      # set on the step a transition happens

    def update(self, v_body_x, yaw, xy, policy_dt, cmd_vx):
        """Returns (lin_vel_x, ang_vel_yaw) for this policy step."""
        p = self.p
        self.last_event = None

        if not self.enabled:
            return cmd_vx, float(np.clip(
                p["k_heading"] * wrap_angle(self.target_heading - yaw),
                -p["max_yaw"], p["max_yaw"]))

        self.state_t += policy_dt

        if self.state == "CRUISE":
            if v_body_x < p["stall_vx"]:
                self.stall_t += policy_dt
            else:
                self.stall_t = 0.0

            if self.stall_t > p["stall_s"]:
                self.state, self.state_t, self.stall_t = "BACKUP", 0.0, 0.0
                self.stuck_xy = np.asarray(xy, dtype=float).copy()
                self.n_recoveries += 1
                self.last_event = "stuck"
            else:
                # Only unwind the detour once clear of what caused it.
                cleared = (self.stuck_xy is None
                           or np.linalg.norm(np.asarray(xy) - self.stuck_xy)
                           > p["commit_dist"])
                if cleared and self.target_heading != 0.0:
                    step = np.radians(min(p["return_rate"] * policy_dt,
                                          abs(np.degrees(self.target_heading))))
                    self.target_heading -= np.sign(self.target_heading) * step

        elif self.state == "BACKUP":
            if self.state_t > p["backup_s"]:
                self.state, self.state_t = "TURN", 0.0
                self.target_heading = wrap_angle(
                    self.target_heading
                    + self.turn_sign * np.radians(p["detour_deg"]))
                self.turn_sign *= -1
                self.last_event = "turn"

        elif self.state == "TURN":
            if self.state_t > p["turn_s"]:
                self.state, self.state_t = "CRUISE", 0.0
                self.last_event = "cruise"

        if self.state == "BACKUP":
            return p["backup_vx"], 0.0
        if self.state == "TURN":
            return 0.1, float(np.clip(
                p["k_heading"] * wrap_angle(self.target_heading - yaw),
                -p["turn_yaw"], p["turn_yaw"]))
        return cmd_vx, float(np.clip(
            p["k_heading"] * wrap_angle(self.target_heading - yaw),
            -p["max_yaw"], p["max_yaw"]))
