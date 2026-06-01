import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_quadruped_example() -> dict:
    """Creates a random input example for the quadruped policy."""
    return {
        "observation/front_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/bottom_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/base_ang_vel": np.random.rand(3),
        "observation/projected_gravity": np.random.rand(3),
        "observation/joint_pos": np.random.rand(16),
        "observation/joint_vel": np.random.rand(16),
        "observation/last_action": np.random.rand(16),
        "observation/normal_force": np.random.rand(4),
        "prompt": "move toward the doorway",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class QuadrupedInputs(transforms.DataTransformFn):
    """Converts quadruped observations into the input format expected by Pi0.

    The 58-dimensional state is:
    [base_ang_vel(3), projected_gravity(3), normal_force(4), joint_pos(16),
     joint_vel(16), last_action(16)].

    A pre-concatenated ``observation/state`` may be provided instead.
    """

    model_type: _model.ModelType

    STATE_FIELDS: ClassVar[tuple[tuple[str, int], ...]] = (
        ("observation/base_ang_vel", 3),
        ("observation/projected_gravity", 3),
        ("observation/normal_force", 4),
        ("observation/joint_pos", 16),
        ("observation/joint_vel", 16),
        ("observation/last_action", 16),
    )
    STATE_DIM: ClassVar[int] = 58

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/front_image"])
        bottom_image = _parse_image(data["observation/bottom_image"])
        state = self._extract_state(data)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": bottom_image,
                "right_wrist_0_rgb": np.zeros_like(front_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

    def _extract_state(self, data: dict) -> np.ndarray:
        if "observation/state" in data:
            state = np.asarray(data["observation/state"])
        else:
            fields = []
            for key, expected_dim in self.STATE_FIELDS:
                value = np.asarray(data[key])
                if value.shape[-1] != expected_dim:
                    raise ValueError(f"Expected {key} to have last dimension {expected_dim}, got shape {value.shape}.")
                fields.append(value)
            state = np.concatenate(fields, axis=-1)

        if state.shape[-1] != self.STATE_DIM:
            raise ValueError(f"Expected quadruped state to have last dimension {self.STATE_DIM}, got shape {state.shape}.")
        return state


@dataclasses.dataclass(frozen=True)
class QuadrupedOutputs(transforms.DataTransformFn):
    """Returns the M20 action-manager inputs from a padded Pi0 action chunk."""

    ACTION_DIM: ClassVar[int] = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.ACTION_DIM])}
