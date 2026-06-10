"""Read and validate one trajectory recorded by the Isaac Sim VLN sampler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

STATE_DIM = 58
ACTION_DIM = 16
ACTION_CHUNK_SIZE = 32
EXPECTED_STATE_LAYOUT = [
    ["base_ang_vel", 3],
    ["projected_gravity", 3],
    ["normal_force", 4],
    ["joint_pos", 16],
    ["joint_vel", 16],
    ["last_action", 16],
]


class IsaacTrajectoryDataset:
    """A deliberately single-trajectory loader for collection-pipeline bring-up."""

    def __init__(self, trajectory_dir: str | Path, *, validate: bool = True) -> None:
        self.trajectory_dir = Path(trajectory_dir).expanduser().resolve()
        self.metadata = self._read_metadata()
        self.prompt = str(self.metadata["instruction"])

        trajectory_path = self.trajectory_dir / "trajectory.npz"
        if not trajectory_path.is_file():
            raise FileNotFoundError(f"Missing trajectory arrays: {trajectory_path}")

        trajectory = np.load(trajectory_path)
        self.simulation_step = trajectory["simulation_step"]
        self.simulation_time_s = trajectory["simulation_time_s"]
        self.state = trajectory["state"]
        self.action = trajectory["action"]
        self.last_action = trajectory["last_action"]

        self.action_chunks = np.load(self.trajectory_dir / "action_chunks.npy")
        self.action_chunk_valid_mask = np.load(self.trajectory_dir / "action_chunk_valid_mask.npy")
        self.image_shape = self._read_image(0, camera="front").shape

        if validate:
            self.validate()

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)

        return {
            "front_image": self._read_image(index, camera="front"),
            "bottom_image": self._read_image(index, camera="bottom_undistorted"),
            "state": np.asarray(self.state[index], dtype=np.float32),
            "action": np.asarray(self.action[index], dtype=np.float32),
            "action_chunk": np.asarray(self.action_chunks[index], dtype=np.float32),
            "action_chunk_valid_mask": np.asarray(self.action_chunk_valid_mask[index], dtype=np.bool_),
            "task": self.prompt,
        }

    @property
    def fps(self) -> float:
        return float(self.metadata["fps"])

    def validate(self) -> None:
        """Fail early if a partially written or misaligned trajectory is selected."""
        num_frames = len(self)
        expected_metadata = {
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_chunk_size": ACTION_CHUNK_SIZE,
        }
        for key, expected_value in expected_metadata.items():
            if int(self.metadata[key]) != expected_value:
                raise ValueError(f"Unexpected metadata {key}: {self.metadata[key]}")
        expected_dt = 1.0 / self.fps
        if not np.isclose(float(self.metadata["dt"]), expected_dt):
            raise ValueError(f"metadata dt does not match fps: dt={self.metadata['dt']}, fps={self.fps}")
        if num_frames != int(self.metadata["num_frames"]):
            raise ValueError(f"metadata num_frames does not match trajectory arrays: {num_frames}")
        if self.metadata["state_layout"] != EXPECTED_STATE_LAYOUT:
            raise ValueError(f"Unexpected state layout: {self.metadata['state_layout']}")
        if self.state.shape != (num_frames, STATE_DIM):
            raise ValueError(f"Unexpected state shape: {self.state.shape}")
        if self.action.shape != (num_frames, ACTION_DIM):
            raise ValueError(f"Unexpected action shape: {self.action.shape}")
        if self.last_action.shape != (num_frames, ACTION_DIM):
            raise ValueError(f"Unexpected last_action shape: {self.last_action.shape}")
        if self.action_chunks.shape != (num_frames, ACTION_CHUNK_SIZE, ACTION_DIM):
            raise ValueError(f"Unexpected action chunk shape: {self.action_chunks.shape}")
        if self.action_chunk_valid_mask.shape != (num_frames, ACTION_CHUNK_SIZE):
            raise ValueError(f"Unexpected action chunk mask shape: {self.action_chunk_valid_mask.shape}")

        arrays = (self.simulation_time_s, self.state, self.action, self.last_action, self.action_chunks)
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("Trajectory contains NaN or infinite values")
        if num_frames > 1 and not np.all(np.diff(self.simulation_step) == 1):
            raise ValueError("simulation_step is not contiguous")
        if num_frames > 1 and not np.allclose(np.diff(self.simulation_time_s), expected_dt):
            raise ValueError(f"simulation_time_s does not advance at the expected {expected_dt}s interval")
        if num_frames > 1 and not np.allclose(self.last_action[1:], self.action[:-1]):
            raise ValueError("last_action does not match the previous PPO output")
        if not np.allclose(self.last_action[0], 0.0):
            raise ValueError("The first last_action must be zero-filled")
        if not np.allclose(self.state[:, -ACTION_DIM:], self.last_action):
            raise ValueError("state does not contain last_action in its final 16 dimensions")

        for index in range(num_frames):
            valid_count = min(ACTION_CHUNK_SIZE, num_frames - index)
            if not self.action_chunk_valid_mask[index, :valid_count].all():
                raise ValueError(f"Missing valid action chunk entries at frame {index}")
            if self.action_chunk_valid_mask[index, valid_count:].any():
                raise ValueError(f"Unexpected valid action chunk padding at frame {index}")
            if not np.allclose(self.action_chunks[index, :valid_count], self.action[index : index + valid_count]):
                raise ValueError(f"Misaligned action chunk at frame {index}")
            if not np.allclose(self.action_chunks[index, valid_count:], 0.0):
                raise ValueError(f"Action chunk padding is not zero at frame {index}")

            front_shape = self._read_image(index, camera="front").shape
            bottom_shape = self._read_image(index, camera="bottom_undistorted").shape
            if front_shape != self.image_shape or bottom_shape != self.image_shape:
                raise ValueError(
                    f"Image shape mismatch at frame {index}: front={front_shape}, bottom={bottom_shape}, "
                    f"expected={self.image_shape}"
                )

    def _read_metadata(self) -> dict[str, Any]:
        metadata_path = self.trajectory_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")
        with metadata_path.open(encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def _read_image(self, index: int, *, camera: str) -> np.ndarray:
        relative_path = str(self.metadata["images"][camera]).format(frame_index=index)
        image_path = self.trajectory_dir / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing {camera} image for frame {index}: {image_path}")
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"))
