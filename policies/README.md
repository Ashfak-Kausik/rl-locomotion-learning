# `policies/` — external model checkpoints

This directory is where the pretrained **walk-these-ways** policy lives.
Its contents are **deliberately not committed** (large binaries), which is why
you will only see this README on a fresh clone.

Expected layout:

```
policies/
└── walk-these-ways-go2/
    ├── body_latest.jit                 # actor: (1, 2102) -> (1, 12)
    └── adaptation_module_latest.jit    # RMA student: (1, 2100) -> (1, 2)
```

Anything that reads them resolves the location through
`stage2-go2-mujoco-inference/paths.py`, so you can keep the files elsewhere:

```bash
export GO2_POLICY_DIR=/absolute/path/to/checkpoints
```

Verify what the tooling can see:

```bash
python scripts/check_env.py            # reports policy status
python stage2-go2-mujoco-inference/paths.py
```

See `docs/DEPENDENCIES.md` for how to obtain the files and what still works
without them (spoiler: all of Stage 1, Stage 2 scripts 01-04, scene
generation, and every figure in the paper).
