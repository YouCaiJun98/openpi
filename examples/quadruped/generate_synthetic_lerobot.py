"""Generate a small synthetic LeRobot dataset for the M20 quadruped pipeline.

Example:
    uv run examples/quadruped/generate_synthetic_lerobot.py --overwrite
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

REPO_ID = "openpi/m20_quadruped_synthetic"
FPS = 50
STATE_DIM = 58
ACTION_DIM = 16
IMAGE_SHAPE = (224, 224, 3)

TASKS = (
    "walk forward toward the doorway",
    "turn left and continue forward",
    "turn right around the obstacle",
    "stop in front of the target",
)


def _make_image(rng: np.random.Generator, episode_index: int, frame_index: int, *, bottom_camera: bool) -> np.ndarray:
    """Creates a structured image so the camera path is easy to inspect."""
    height, width, _ = IMAGE_SHAPE
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    phase = episode_index * 37 + frame_index * 3 + (89 if bottom_camera else 0)
    noise = rng.integers(0, 24, size=(height, width), dtype=np.uint8)
    return np.stack(
        [
            ((x + phase + noise) % 256).astype(np.uint8),
            ((y + 2 * phase + noise) % 256).astype(np.uint8),
            ((x + y + 3 * phase + noise) % 256).astype(np.uint8),
        ],
        axis=-1,
    )


def _make_frame(
    rng: np.random.Generator,
    episode_index: int,
    frame_index: int,
    previous_action: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Creates correlated random state and action values with the production dimensions."""
    t = frame_index / FPS
    phase = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)
    joint_pos = 0.35 * np.sin(2.0 * np.pi * 0.8 * t + phase)
    joint_vel = 0.35 * 2.0 * np.pi * 0.8 * np.cos(2.0 * np.pi * 0.8 * t + phase)
    normal_force = np.maximum(0.0, 65.0 + 20.0 * np.sin(2.0 * np.pi * 1.6 * t + phase[:4]))
    base_ang_vel = rng.normal(0.0, 0.08, size=3).astype(np.float32)
    projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    projected_gravity += rng.normal(0.0, 0.015, size=3).astype(np.float32)

    leg_target = joint_pos[:12] + rng.normal(0.0, 0.025, size=12).astype(np.float32)
    wheel_velocity = np.full(4, 1.5 + 0.2 * episode_index, dtype=np.float32)
    wheel_velocity += rng.normal(0.0, 0.08, size=4).astype(np.float32)
    action = np.concatenate([leg_target, wheel_velocity]).astype(np.float32)

    state = np.concatenate(
        [
            base_ang_vel,
            projected_gravity,
            normal_force.astype(np.float32),
            joint_pos.astype(np.float32),
            joint_vel.astype(np.float32),
            previous_action,
        ]
    ).astype(np.float32)
    assert state.shape == (STATE_DIM,)

    frame = {
        "front_image": _make_image(rng, episode_index, frame_index, bottom_camera=False),
        "bottom_image": _make_image(rng, episode_index, frame_index, bottom_camera=True),
        "state": state,
        "action": action,
        "task": TASKS[episode_index % len(TASKS)],
    }
    return frame, action


def main(
    repo_id: str = REPO_ID,
    *,
    num_episodes: int = 4,
    frames_per_episode: int = 96,
    seed: int = 0,
    overwrite: bool = False,
) -> None:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="m20",
        fps=FPS,
        features={
            "front_image": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "bottom_image": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["action"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    rng = np.random.default_rng(seed)
    for episode_index in range(num_episodes):
        previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        for frame_index in range(frames_per_episode):
            frame, previous_action = _make_frame(rng, episode_index, frame_index, previous_action)
            dataset.add_frame(frame)
        dataset.save_episode()

    print(f"Wrote {num_episodes} episodes x {frames_per_episode} frames to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
