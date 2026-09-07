"""
Stage 2.5 (prep): Load the .jit policy files and check their input/output shapes.
Goal: Confirm body expects 2102 dims (2100 history + 2 from adaptation module)
      and adaptation module expects 2100 dims (30 timesteps x 70 obs).
"""

import os
import torch

# Repo-relative default; override with GO2_POLICY_DIR to point at a
# different checkpoint directory (e.g. one restored by download_policy.sh).
POLICY_DIR = os.environ.get(
    "GO2_POLICY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
)

body = torch.jit.load(f"{POLICY_DIR}/body_latest.jit")
adaptation = torch.jit.load(f"{POLICY_DIR}/adaptation_module_latest.jit")

print("=== Body (actor) network ===")
print(body)
print()

print("=== Adaptation module ===")
print(adaptation)
print()

# Test adaptation module: takes 30 frames x 70 dims = 2100
dummy_obs_history = torch.zeros(1, 30 * 70)  # [1, 2100]
adapt_out = adaptation(dummy_obs_history)
print(f"Adaptation input:  {dummy_obs_history.shape}")
print(f"Adaptation output: {adapt_out.shape}")

# Body takes full history (2100) + adaptation output (2) = 2102
body_input = torch.cat([dummy_obs_history, adapt_out], dim=1)  # [1, 2102]
print(f"\nBody input: history (2100) + adapt_out ({adapt_out.shape[1]}) = {body_input.shape}")

body_out = body(body_input)
print(f"Body output (action): {body_out.shape}")