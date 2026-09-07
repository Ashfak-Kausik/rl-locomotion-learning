## Stage 2: Go2 Locomotion in MuJoCo

Goal: deploy a pretrained `walk-these-ways` policy on a Unitree Go2 in MuJoCo.

### Sub-stages
- **2.1** Load Go2 in MuJoCo, verify viewer
- **2.2** Inspect model structure (bodies, joints, actuators)
- **2.3** PD control to hold a standing pose
- **2.4** Construct the 70-dim observation vector
- **2.5** Load the trained policy and run inference — **robot walks**

### Key debugging insights
- Clock signal is 4 sines (one per foot), not sin/cos pairs
- Training default pose ≠ MJCF keyframe pose — must override
- Hip sign conventions: FL/RL positive, FR/RR negative
- Decimation = 10 (sim 500 Hz, policy 50 Hz)
- Action scale: 0.25, with 0.5× reduction for hips

### Result
Pretrained policy walks the Go2 forward at ~0.28 m/s sustained for 60s+.
Minor yaw drift remains (open issue, low priority).

### Reproducing this locally
- Checkpoint: not committed (binary, git-ignored). Run `../download_policy.sh`
  from the repo root to fetch and hash-verify it into `checkpoints/`
  (override the location with the `GO2_POLICY_DIR` env var).
- Dependencies: pinned in `../requirements.txt`.
- Training configuration and provenance: see `TRAINING_CONFIG.md`.
