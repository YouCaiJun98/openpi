"""Convert validated Isaac Sim trajectories into a local LeRobot dataset.

Example:
    uv run examples/quadruped/convert_isaac_trajectory_to_lerobot.py \
        --trajectories-dir /workspace/openpi/data/vln_sample \
        --overwrite
"""

from pathlib import Path
import shutil

from isaac_trajectory_dataset import ACTION_DIM
from isaac_trajectory_dataset import IsaacTrajectoryDataset
from isaac_trajectory_dataset import STATE_DIM
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

REPO_ID = "openpi/m20_quadruped_isaac"


def main(
    *,
    trajectory_dir: Path | None = None,
    trajectories_dir: Path | None = None,
    repo_id: str = REPO_ID,
    overwrite: bool = False,
) -> None:
    """Convert either one explicitly selected trajectory or every trajectory in one directory."""
    sources = _load_sources(trajectory_dir=trajectory_dir, trajectories_dir=trajectories_dir)
    reference = sources[0]
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="m20",
        fps=reference.fps,
        features={
            "front_image": {
                "dtype": "image",
                "shape": reference.image_shape,
                "names": ["height", "width", "channel"],
            },
            "bottom_image": {
                "dtype": "image",
                "shape": reference.image_shape,
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

    total_frames = 0
    for source in sources:
        for index in range(len(source)):
            frame = source[index]
            dataset.add_frame(
                {
                    "front_image": frame["front_image"],
                    "bottom_image": frame["bottom_image"],
                    "state": frame["state"],
                    "action": frame["action"],
                    "task": frame["task"],
                }
            )
        dataset.save_episode()
        total_frames += len(source)
        print(f"Converted episode {source.trajectory_dir.name}: {len(source)} frames")

    print(f"Wrote {len(sources)} episodes and {total_frames} frames to {output_path}")


def _load_sources(
    *,
    trajectory_dir: Path | None,
    trajectories_dir: Path | None,
) -> list[IsaacTrajectoryDataset]:
    if (trajectory_dir is None) == (trajectories_dir is None):
        raise ValueError("Pass exactly one of --trajectory-dir or --trajectories-dir.")

    if trajectory_dir is not None:
        trajectory_dirs = [trajectory_dir]
    else:
        assert trajectories_dir is not None
        if not trajectories_dir.is_dir():
            raise NotADirectoryError(trajectories_dir)
        trajectory_dirs = sorted(path for path in trajectories_dir.iterdir() if path.is_dir())
        if not trajectory_dirs:
            raise ValueError(f"No trajectory directories found in {trajectories_dir}")

    sources = []
    for path in trajectory_dirs:
        try:
            sources.append(IsaacTrajectoryDataset(path))
        except Exception as error:
            raise RuntimeError(f"Failed to validate Isaac trajectory: {path}") from error

    reference = sources[0]
    for source in sources[1:]:
        if source.fps != reference.fps:
            raise ValueError(f"Trajectory FPS mismatch: {source.trajectory_dir} has {source.fps}, expected {reference.fps}")
        if source.image_shape != reference.image_shape:
            raise ValueError(
                f"Trajectory image shape mismatch: {source.trajectory_dir} has {source.image_shape}, "
                f"expected {reference.image_shape}"
            )
    return sources


if __name__ == "__main__":
    tyro.cli(main)
