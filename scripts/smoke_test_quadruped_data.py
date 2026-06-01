"""Load one synthetic quadruped batch through the production data transforms.

Example:
    uv run scripts/smoke_test_quadruped_data.py
"""

import dataclasses

import numpy as np
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _describe(name: str, value) -> None:
    array = np.asarray(value)
    print(f"{name}: shape={array.shape}, dtype={array.dtype}")


def main(config_name: str = "pi0_quadruped_synthetic", *, skip_norm_stats: bool = False) -> None:
    config = dataclasses.replace(_config.get_config(config_name), num_workers=0)
    loader = _data_loader.create_data_loader(
        config,
        num_batches=1,
        skip_norm_stats=skip_norm_stats,
    )
    observation, actions = next(iter(loader))

    for key, image in observation.images.items():
        _describe(f"observation.images[{key!r}]", image)
    for key, image_mask in observation.image_masks.items():
        _describe(f"observation.image_masks[{key!r}]", image_mask)
    _describe("observation.state", observation.state)
    _describe("observation.tokenized_prompt", observation.tokenized_prompt)
    _describe("observation.tokenized_prompt_mask", observation.tokenized_prompt_mask)
    _describe("actions", actions)

    if observation.state.shape[-1] != 58:
        raise ValueError(f"Expected 58-dimensional quadruped state, got {observation.state.shape}.")
    if actions.shape[-2:] != (32, 32):
        raise ValueError(f"Expected padded action chunks with shape (*, 32, 32), got {actions.shape}.")

    print("Quadruped data pipeline smoke test passed.")


if __name__ == "__main__":
    tyro.cli(main)
