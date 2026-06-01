# Quadruped pipeline smoke test

Generate the local synthetic LeRobot dataset:

```bash
uv run examples/quadruped/generate_synthetic_lerobot.py --overwrite
```

After training `pi0_quadruped_synthetic_base`, serve its latest checkpoint:

```bash
uv run examples/quadruped/serve_checkpoint.py
```

In a second terminal, query the policy:

```bash
uv run examples/quadruped/query_policy.py
```

For a real fine-tuning run, select the latest `pi0_quadruped` checkpoint:

```bash
uv run examples/quadruped/serve_checkpoint.py \
  --config-name pi0_quadruped \
  --exp-name m20_finetune
```
