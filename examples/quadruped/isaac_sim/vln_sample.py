# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive play script for RSL-RL checkpoint in Isaac Lab."""

import argparse
import base64
import datetime
import json
import os
import random
import re
import socket
import sys
import threading
from collections import deque

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args


# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with VLN-driven interactive control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Rough-Deeprobotics-M20-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--lin_speed", type=float, default=0.5, help="Linear speed (m/s) for move/back/left/right commands.")
parser.add_argument("--ang_speed", type=float, default=1.0, help="Angular speed (rad/s) for turn commands.")
parser.add_argument(
    "--hospital",
    action="store_true",
    default=False,
    help="Use default NVIDIA hospital scene resolved from common asset locations.",
)
DEFAULT_SCENE_USD = os.environ.get(
    "VLN_DEFAULT_SCENE_USD",
    "/workspace/third_party/TataService/kujiale_0003/start_result_navigation.usd",
)
parser.add_argument(
    "--scene_usd",
    type=str,
    default=DEFAULT_SCENE_USD,
    help=(
        "Scene USD path to override terrain. Supports local file path or Omniverse URI. "
        "Default uses kujiale_0003 scene; can also be overridden by env VLN_DEFAULT_SCENE_USD."
    ),
)
parser.add_argument(
    "--disable_scene_override",
    action="store_true",
    default=False,
    help="Disable custom scene override and keep task default terrain.",
)
# Deprecated aliases kept for compatibility with old command lines.
parser.add_argument("--hospital_usd", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--disable_hospital_scene", action="store_true", default=False, help=argparse.SUPPRESS)
parser.add_argument("--photo_dir", type=str, default="photos", help="Directory to save first-person snapshots.")
parser.add_argument("--vln_photo_prefix", type=str, default="vln_obs", help="Prefix for VLN observation image filenames.")
# parser.add_argument("--vln_host", type=str, default="172.23.11.9", help="VLN server host (same as vln_ref).")
# parser.add_argument("--vln_host", type=str, default="172.23.53.75", help="VLN server host (same as vln_ref).")
parser.add_argument("--vln_host", type=str, default="172.23.64.202", help="VLN server host (same as vln_ref).")
parser.add_argument("--vln_port", type=int, default=54321, help="VLN server port (same as vln_ref).")
parser.add_argument(
    "--vln_max_images",
    type=int,
    default=8,
    help="Max number of images sent to VLN per query. First and last are always included.",
)
parser.add_argument(
    "--vln_max_iterations",
    type=int,
    default=200,
    help="Max VLN query iterations per goal.",
)
parser.add_argument(
    "--delta_exec_mode",
    type=str,
    default="twist",
    choices=["twist", "atomic"],
    help="How to execute delta output: twist=simultaneous vx/vy/wz, atomic=sequential move/left/turn.",
)
parser.add_argument(
    "--use_real_vln",
    action="store_true",
    default=False,
    help="Use real VLN server socket communication.",
)
parser.add_argument(
    "--dummy_vln",
    action="store_true",
    default=True,
    help="Use dummy sampled VLN output for local testing.",
)
parser.add_argument("--front_sensor_width", type=int, default=640, help="Front camera image width.")
parser.add_argument("--front_sensor_height", type=int, default=480, help="Front camera image height.")
parser.add_argument("--bottom_sensor_width", type=int, default=640, help="Bottom fisheye camera image width.")
parser.add_argument("--bottom_sensor_height", type=int, default=480, help="Bottom fisheye camera image height.")
parser.add_argument("--front_cam_x", type=float, default=-0.40, help="Front sensor offset x in robot frame.")
parser.add_argument("--front_cam_y", type=float, default=0.0, help="Front sensor offset y in robot frame.")
parser.add_argument("--front_cam_z", type=float, default=0.08, help="Front sensor offset z in robot frame.")
parser.add_argument("--bottom_cam_x", type=float, default=0.0, help="Bottom fisheye sensor offset x in robot frame.")
parser.add_argument("--bottom_cam_y", type=float, default=0.0, help="Bottom fisheye sensor offset y in robot frame.")
parser.add_argument("--bottom_cam_z", type=float, default=-0.085, help="Bottom fisheye sensor offset z in robot frame.")
parser.add_argument("--data_dir", type=str, default="data", help="Directory for VLN trajectory recordings.")
parser.add_argument("--action_chunk_size", type=int, default=32, help="Sliding-window PPO action chunk length.")
parser.add_argument("--stop_tail_seconds", type=float, default=0.75, help="Seconds of standing actions to record after stop.")
parser.add_argument(
    "--bottom_undistorted_fov",
    type=float,
    default=120.0,
    help="Rectilinear field of view for bottom fisheye post-processing.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# enable camera pipeline for interactive snapshots / videos
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rl_utils import camera_follow

import gymnasium as gym
import numpy as np
import omni.client
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.dict import print_dict
import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


FRONT_CAMERA_NAME = "front_camera"
BOTTOM_CAMERA_NAME = "bottom_fisheye_camera"
FISHEYE_NOMINAL_WIDTH = 1936.0
FISHEYE_NOMINAL_HEIGHT = 1216.0
FISHEYE_OPTICAL_CENTRE_X = 970.94244
FISHEYE_OPTICAL_CENTRE_Y = 600.37482
FISHEYE_MAX_FOV = 200.0
FISHEYE_POLYNOMIAL = (0.0, 0.00245, 0.0, 0.0, 0.0)
LEG_JOINT_NAMES = (
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",
    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",
    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",
    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",
)
WHEEL_JOINT_NAMES = ("fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint")
JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
WHEEL_BODY_NAMES = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


def _infer_step_dt(env_cfg) -> float:
    sim_dt = float(getattr(env_cfg.sim, "dt", 1.0 / 60.0))
    decimation = int(getattr(env_cfg, "decimation", 1))
    return sim_dt * decimation


def _configure_mounted_cameras(env_cfg):
    """Attach front pinhole and bottom fisheye sensors to the M20 base link."""
    fisheye_cfg_cls = getattr(sim_utils, "FisheyeCameraCfg", None)
    if fisheye_cfg_cls is None:
        raise RuntimeError("This Isaac Lab build does not expose sim_utils.FisheyeCameraCfg.")

    update_period = _infer_step_dt(env_cfg)
    env_cfg.scene.front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera",
        update_period=update_period,
        height=args_cli.front_sensor_height,
        width=args_cli.front_sensor_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 30.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(args_cli.front_cam_x, args_cli.front_cam_y, args_cli.front_cam_z),
            rot=(0.5, -0.5, -0.5, 0.5),
            convention="ros",
        ),
    )
    env_cfg.scene.bottom_fisheye_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/bottom_fisheye_camera",
        update_period=update_period,
        height=args_cli.bottom_sensor_height,
        width=args_cli.bottom_sensor_width,
        data_types=["rgb"],
        spawn=fisheye_cfg_cls(
            clipping_range=(0.05, 30.0),
            fisheye_nominal_width=FISHEYE_NOMINAL_WIDTH,
            fisheye_nominal_height=FISHEYE_NOMINAL_HEIGHT,
            fisheye_optical_centre_x=FISHEYE_OPTICAL_CENTRE_X,
            fisheye_optical_centre_y=FISHEYE_OPTICAL_CENTRE_Y,
            fisheye_max_fov=FISHEYE_MAX_FOV,
            fisheye_polynomial_a=FISHEYE_POLYNOMIAL[0],
            fisheye_polynomial_b=FISHEYE_POLYNOMIAL[1],
            fisheye_polynomial_c=FISHEYE_POLYNOMIAL[2],
            fisheye_polynomial_d=FISHEYE_POLYNOMIAL[3],
            fisheye_polynomial_e=FISHEYE_POLYNOMIAL[4],
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(args_cli.bottom_cam_x, args_cli.bottom_cam_y, args_cli.bottom_cam_z),
            rot=(0.0, -0.70710678, 0.70710678, 0.0),
            convention="ros",
        ),
    )


def _parse_action_script(script: str):
    actions = []
    for token in script.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid action token '{token}'. Expected 'name:value'.")
        name, value = token.split(":", 1)
        actions.append((name.strip().lower(), float(value.strip())))
    if not actions:
        raise ValueError("Action script is empty.")
    return actions


def _resolve_scene_usd_path(explicit_path: str | None) -> str | None:
    if not explicit_path:
        return None

    # Accept local file path directly.
    local_path = os.path.expanduser(explicit_path)
    if os.path.isfile(local_path):
        return os.path.abspath(local_path)

    # Also support Omniverse URI (e.g. omniverse://..., file://...).
    result, _ = omni.client.stat(explicit_path)
    if result == omni.client.Result.OK:
        return explicit_path

    return None


def _resolve_hospital_usd_path() -> str | None:
    candidates = [
        f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd",
        f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/Hospital.usd",
        f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/warehouse_hospital.usd",
    ]
    for path in candidates:
        result, _ = omni.client.stat(path)
        if result == omni.client.Result.OK:
            return path
    return None


def _get_robot_pose(env):
    """Return robot world pose as (x, y, z, qw, qx, qy, qz, yaw_deg)."""
    robot = env.unwrapped.scene["robot"]
    pos = robot.data.root_pos_w[0].detach().cpu()
    quat = robot.data.root_quat_w[0].detach().cpu().unsqueeze(0)
    yaw_tensor = math_utils.euler_xyz_from_quat(quat)[2]
    yaw_deg = float(torch.rad2deg(yaw_tensor.reshape(-1)[0]).item())
    return (
        float(pos[0].item()),
        float(pos[1].item()),
        float(pos[2].item()),
        float(quat[0, 0].item()),
        float(quat[0, 1].item()),
        float(quat[0, 2].item()),
        float(quat[0, 3].item()),
        yaw_deg,
    )


def _set_robot_pose(env, x: float, y: float, z: float, yaw_deg: float | None):
    """Set robot world pose with best-effort API compatibility."""
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    env_ids = torch.tensor([0], dtype=torch.long, device=device)

    if yaw_deg is None:
        # Convert via CPU to avoid getting an inference tensor.
        current_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        target_quat = torch.tensor(current_quat, dtype=torch.float32, device=device)
    else:
        yaw_rad = torch.tensor([float(yaw_deg) * float(torch.pi) / 180.0], dtype=torch.float32, device=device)
        zeros = torch.zeros_like(yaw_rad)
        target_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_rad)[0]

    target_pos = torch.tensor([x, y, z], dtype=torch.float32, device=device)

    last_err = None

    if hasattr(robot, "write_root_pose_to_sim"):
        try:
            root_pose = torch.cat((target_pos.view(1, 3), target_quat.view(1, 4)), dim=1)
            robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(
                    torch.zeros((1, 6), dtype=torch.float32, device=device), env_ids=env_ids
                )
            return
        except Exception as err:
            last_err = err

    if hasattr(robot, "write_root_state_to_sim"):
        try:
            root_state = torch.cat(
                (
                    target_pos.view(1, 3),
                    target_quat.view(1, 4),
                    torch.zeros((1, 6), dtype=torch.float32, device=device),
                ),
                dim=1,
            )
            robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            return
        except Exception as err:
            last_err = err

    if last_err is not None:
        raise RuntimeError(f"Failed to set pose via simulator APIs: {last_err}") from last_err
    raise RuntimeError("Robot object does not expose write_root_pose_to_sim/write_root_state_to_sim in this IsaacLab build.")


def _extract_rgb_array(frame) -> np.ndarray | None:
    """Extract an HxWx3 uint8 RGB image from render output."""
    if frame is None:
        return None
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    elif isinstance(frame, dict):
        for key in ("rgb", "image", "color", "render"):
            if key in frame:
                return _extract_rgb_array(frame[key])
        for value in frame.values():
            arr = _extract_rgb_array(value)
            if arr is not None:
                return arr
        return None
    elif isinstance(frame, (list, tuple)) and len(frame) > 0:
        return _extract_rgb_array(frame[0])

    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        max_value = float(np.max(arr)) if arr.size > 0 else 0.0
        if max_value <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _save_rgb_image(rgb: np.ndarray, path: str):
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, rgb)
        return
    except Exception:
        pass
    from PIL import Image

    Image.fromarray(rgb).save(path)


def _is_mostly_black(rgb: np.ndarray, threshold: int = 5, ratio: float = 0.98) -> bool:
    """Check if image is almost entirely black."""
    gray = rgb.mean(axis=2)
    return float((gray < threshold).mean()) >= ratio


def _extract_mounted_camera_rgb(env, camera_name: str) -> np.ndarray:
    """Read the first environment's current RGB frame from a mounted sensor."""
    camera = env.unwrapped.scene[camera_name]
    frame = camera.data.output.get("rgb")
    if frame is not None and getattr(frame, "ndim", 0) == 4:
        frame = frame[0]
    rgb = _extract_rgb_array(frame)
    if rgb is None:
        raise RuntimeError(f"Mounted camera '{camera_name}' did not produce an RGB frame.")
    return rgb


def _capture_front_camera_image(env, photo_dir: str, photo_name: str) -> str:
    """Save one RGB frame from the robot-mounted front camera."""
    rgb = _extract_mounted_camera_rgb(env, FRONT_CAMERA_NAME)
    if _is_mostly_black(rgb):
        print("[VLN][WARN] Mounted front camera image is mostly black.")

    os.makedirs(photo_dir, exist_ok=True)
    if not photo_name.lower().endswith(".png"):
        photo_name = f"{photo_name}.png"
    photo_path = os.path.abspath(os.path.join(photo_dir, photo_name))
    _save_rgb_image(rgb, photo_path)
    return photo_path


def _undistort_bottom_fisheye(rgb: np.ndarray, rectilinear_fov_deg: float) -> np.ndarray:
    """Convert the polynomial fisheye render into a rectilinear image."""
    try:
        import cv2
    except ImportError as err:
        raise RuntimeError("Bottom fisheye undistortion requires OpenCV (cv2).") from err

    if not 1.0 < rectilinear_fov_deg < 179.0:
        raise ValueError("--bottom_undistorted_fov must be between 1 and 179 degrees.")

    height, width = rgb.shape[:2]
    centre_x = (width - 1) * 0.5
    centre_y = (height - 1) * 0.5
    focal_length = 0.5 * width / np.tan(np.deg2rad(rectilinear_fov_deg) * 0.5)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    ray_x = (grid_x - centre_x) / focal_length
    ray_y = (grid_y - centre_y) / focal_length
    ray_radius = np.sqrt(ray_x**2 + ray_y**2)
    theta = np.arctan(ray_radius)
    phi = np.arctan2(ray_y, ray_x)

    a, b, c, d, e = FISHEYE_POLYNOMIAL
    radius = np.maximum((theta - a) / b, 0.0)
    for _ in range(5):
        polynomial = a + b * radius + c * radius**2 + d * radius**3 + e * radius**4
        derivative = b + 2 * c * radius + 3 * d * radius**2 + 4 * e * radius**3
        radius -= (polynomial - theta) / np.maximum(derivative, 1e-12)

    source_x = (FISHEYE_OPTICAL_CENTRE_X + radius * np.cos(phi)) * width / FISHEYE_NOMINAL_WIDTH
    source_y = (FISHEYE_OPTICAL_CENTRE_Y + radius * np.sin(phi)) * height / FISHEYE_NOMINAL_HEIGHT
    return cv2.remap(
        rgb,
        source_x.astype(np.float32),
        source_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _capture_mounted_camera_images(env, photo_dir: str, photo_name: str):
    """Save front RGB, bottom raw fisheye, and bottom undistorted images."""
    front_rgb = _extract_mounted_camera_rgb(env, FRONT_CAMERA_NAME)
    bottom_raw_rgb = _extract_mounted_camera_rgb(env, BOTTOM_CAMERA_NAME)
    bottom_undistorted_rgb = _undistort_bottom_fisheye(bottom_raw_rgb, args_cli.bottom_undistorted_fov)

    os.makedirs(photo_dir, exist_ok=True)
    stem = os.path.splitext(photo_name)[0]
    paths = {
        "front": os.path.abspath(os.path.join(photo_dir, f"{stem}_front.png")),
        "bottom_fisheye_raw": os.path.abspath(os.path.join(photo_dir, f"{stem}_bottom_fisheye_raw.png")),
        "bottom_undistorted": os.path.abspath(os.path.join(photo_dir, f"{stem}_bottom_undistorted.png")),
    }
    _save_rgb_image(front_rgb, paths["front"])
    _save_rgb_image(bottom_raw_rgb, paths["bottom_fisheye_raw"])
    _save_rgb_image(bottom_undistorted_rgb, paths["bottom_undistorted"])
    return paths


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _find_ids(asset, names: tuple[str, ...], method_name: str) -> list[int]:
    method = getattr(asset, method_name)
    try:
        ids, _ = method(list(names), preserve_order=True)
    except TypeError:
        ids, _ = method(list(names))
    ids = [int(index) for index in ids]
    if len(ids) != len(names):
        raise RuntimeError(f"Expected to resolve {len(names)} names with {method_name}, got {len(ids)}: {names}")
    return ids


class VlnTrajectoryRecorder:
    """Records aligned observations and raw PPO outputs for one goal command."""

    STATE_DIM = 58
    ACTION_DIM = 16

    def __init__(self, data_dir: str, dt: float, action_chunk_size: int, stop_tail_seconds: float):
        self._data_dir = os.path.abspath(data_dir)
        self._dt = float(dt)
        self._action_chunk_size = int(action_chunk_size)
        self._tail_frames = max(1, int(round(float(stop_tail_seconds) / self._dt)))
        if self._action_chunk_size <= 0:
            raise ValueError("--action_chunk_size must be positive.")

        self._trajectory_dir = None
        self._instruction = None
        self._frames = []
        self._previous_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self._tail_frames_remaining = None
        self._stop_reason = None
        self._joint_ids = None
        self._wheel_body_ids = None

    @property
    def is_recording(self) -> bool:
        return self._trajectory_dir is not None

    def start(self, instruction: str):
        if self.is_recording:
            self.finalize("interrupted_by_new_goal")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._trajectory_dir = os.path.join(self._data_dir, timestamp)
        self._instruction = instruction
        self._frames = []
        self._previous_action = np.zeros(self.ACTION_DIM, dtype=np.float32)
        self._tail_frames_remaining = None
        self._stop_reason = None
        for dirname in ("front", "bottom_undistorted", "bottom_fisheye_raw"):
            os.makedirs(os.path.join(self._trajectory_dir, dirname), exist_ok=True)
        with open(os.path.join(self._trajectory_dir, "prompt.txt"), "w", encoding="utf-8") as file:
            file.write(f"{instruction}\n")
        print(f"[DATA] Started trajectory: {self._trajectory_dir}")

    def request_stop(self, reason: str):
        if not self.is_recording or self._tail_frames_remaining is not None:
            return
        self._stop_reason = reason
        self._tail_frames_remaining = self._tail_frames
        print(f"[DATA] Recording {self._tail_frames} standing tail frames after {reason}.")

    def record_frame(self, env, ppo_action, simulation_step: int):
        if not self.is_recording:
            return

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene["contact_forces"]
        if self._joint_ids is None:
            self._joint_ids = _find_ids(robot, JOINT_NAMES, "find_joints")
            self._wheel_body_ids = _find_ids(contact_sensor, WHEEL_BODY_NAMES, "find_bodies")

        action = _to_numpy(ppo_action)[0].astype(np.float32, copy=True)
        if action.shape != (self.ACTION_DIM,):
            raise RuntimeError(f"Expected PPO action shape {(self.ACTION_DIM,)}, got {action.shape}.")

        base_ang_vel = _to_numpy(robot.data.root_ang_vel_b[0]).astype(np.float32, copy=True)
        projected_gravity = _to_numpy(robot.data.projected_gravity_b[0]).astype(np.float32, copy=True)
        joint_pos = _to_numpy(robot.data.joint_pos[0, self._joint_ids]).astype(np.float32, copy=True)
        joint_vel = _to_numpy(robot.data.joint_vel[0, self._joint_ids]).astype(np.float32, copy=True)
        contact_force_w = _to_numpy(contact_sensor.data.net_forces_w[0, self._wheel_body_ids]).astype(
            np.float32, copy=True
        )
        normal_force = np.maximum(contact_force_w[:, 2], 0.0)
        state = np.concatenate(
            [base_ang_vel, projected_gravity, normal_force, joint_pos, joint_vel, self._previous_action]
        ).astype(np.float32)
        if state.shape != (self.STATE_DIM,):
            raise RuntimeError(f"Expected state shape {(self.STATE_DIM,)}, got {state.shape}.")

        frame_index = len(self._frames)
        filename = f"{frame_index:06d}.png"
        front_rgb = _extract_mounted_camera_rgb(env, FRONT_CAMERA_NAME)
        bottom_raw_rgb = _extract_mounted_camera_rgb(env, BOTTOM_CAMERA_NAME)
        bottom_undistorted_rgb = _undistort_bottom_fisheye(bottom_raw_rgb, args_cli.bottom_undistorted_fov)
        _save_rgb_image(front_rgb, os.path.join(self._trajectory_dir, "front", filename))
        _save_rgb_image(bottom_raw_rgb, os.path.join(self._trajectory_dir, "bottom_fisheye_raw", filename))
        _save_rgb_image(bottom_undistorted_rgb, os.path.join(self._trajectory_dir, "bottom_undistorted", filename))

        self._frames.append(
            {
                "simulation_step": int(simulation_step),
                "simulation_time_s": float(simulation_step * self._dt),
                "wall_time_ns": int(time.time_ns()),
                "state": state,
                "base_ang_vel": base_ang_vel,
                "projected_gravity": projected_gravity,
                "normal_force": normal_force,
                "contact_force_w": contact_force_w,
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "last_action": self._previous_action.copy(),
                "action": action,
            }
        )
        self._previous_action = action

        if self._tail_frames_remaining is not None:
            self._tail_frames_remaining -= 1
            if self._tail_frames_remaining <= 0:
                self.finalize(self._stop_reason or "stop")

    def finalize(self, reason: str):
        if not self.is_recording:
            return

        trajectory_dir = self._trajectory_dir
        frames = self._frames
        arrays = {}
        if frames:
            for key in frames[0]:
                arrays[key] = np.asarray([frame[key] for frame in frames])
        else:
            arrays = {
                "simulation_step": np.empty((0,), dtype=np.int64),
                "simulation_time_s": np.empty((0,), dtype=np.float64),
                "wall_time_ns": np.empty((0,), dtype=np.int64),
                "state": np.empty((0, self.STATE_DIM), dtype=np.float32),
                "action": np.empty((0, self.ACTION_DIM), dtype=np.float32),
            }
        np.savez_compressed(os.path.join(trajectory_dir, "trajectory.npz"), **arrays)

        actions = arrays["action"]
        chunks = np.zeros((len(actions), self._action_chunk_size, self.ACTION_DIM), dtype=np.float32)
        valid_mask = np.zeros((len(actions), self._action_chunk_size), dtype=np.bool_)
        for frame_index in range(len(actions)):
            valid_count = min(self._action_chunk_size, len(actions) - frame_index)
            chunks[frame_index, :valid_count] = actions[frame_index : frame_index + valid_count]
            valid_mask[frame_index, :valid_count] = True
        np.save(os.path.join(trajectory_dir, "action_chunks.npy"), chunks)
        np.save(os.path.join(trajectory_dir, "action_chunk_valid_mask.npy"), valid_mask)

        metadata = {
            "instruction": self._instruction,
            "stop_reason": reason,
            "fps": 1.0 / self._dt,
            "dt": self._dt,
            "num_frames": len(frames),
            "state_dim": self.STATE_DIM,
            "state_layout": [
                ["base_ang_vel", 3],
                ["projected_gravity", 3],
                ["normal_force", 4],
                ["joint_pos", 16],
                ["joint_vel", 16],
                ["last_action", 16],
            ],
            "action_dim": self.ACTION_DIM,
            "action_semantics": "raw PPO policy output before action-manager scaling and offsets",
            "action_chunk_size": self._action_chunk_size,
            "joint_names": JOINT_NAMES,
            "wheel_body_names": WHEEL_BODY_NAMES,
            "images": {
                "front": "front/{frame_index:06d}.png",
                "bottom_undistorted": "bottom_undistorted/{frame_index:06d}.png",
                "bottom_fisheye_raw": "bottom_fisheye_raw/{frame_index:06d}.png",
            },
        }
        with open(os.path.join(trajectory_dir, "metadata.json"), "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print(f"[DATA] Saved trajectory with {len(frames)} frames: {trajectory_dir}")

        self._trajectory_dir = None
        self._instruction = None
        self._frames = []
        self._tail_frames_remaining = None
        self._stop_reason = None


def dummy_vln_api(user_instruction: str, image_paths: list[str]) -> str:
    """Dummy VLN output sampler with atomic command strings."""
    _ = user_instruction
    _ = image_paths
    candidates = [
        "move 1",
        "back 1",
        "turn 90",
        "turn -90",
        "right 1",
        "left 1",
        "stop",
    ]
    # Keep stop less frequent so one round can execute several actions.
    weights = [0.22, 0.12, 0.16, 0.16, 0.12, 0.12, 0.10]
    return random.choices(candidates, weights=weights, k=1)[0]


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        packet = sock.recv(min(4096, size - len(chunks)))
        if not packet:
            raise ConnectionError(f"VLN server closed the connection after {len(chunks)} / {size} bytes")
        chunks.extend(packet)
    return bytes(chunks)


def _image_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _select_images_for_vln(history_paths: list[str], max_images: int) -> list[str]:
    """Select image subset for VLN query with first/last guaranteed and uniform middle sampling."""
    if not history_paths:
        return []
    if max_images <= 1:
        return [history_paths[-1]]
    if len(history_paths) <= max_images:
        return history_paths[:]

    first = history_paths[0]
    last = history_paths[-1]
    middle = history_paths[1:-1]
    k = max_images - 2
    if k <= 0:
        return [first, last]
    if len(middle) <= k:
        return [first] + middle + [last]
    if k == 1:
        return [first, middle[len(middle) // 2], last]

    indices = [int(i * (len(middle) - 1) / (k - 1)) for i in range(k)]
    sampled_middle = [middle[i] for i in indices]
    return [first] + sampled_middle + [last]


def query_vln_server(
    user_instruction: str,
    image_paths: list[str],
    host: str,
    port: int,
    timeout: float = 30.0,
) -> str:
    """VLN server query with same transport protocol as vln_ref.py."""
    if not image_paths:
        raise ValueError("At least one image is required for VLN query.")
    image_b64_list = [_image_file_to_base64(p) for p in image_paths]
    request_data = {"images": image_b64_list, "query": user_instruction}
    data_bytes = json.dumps(request_data).encode()

    # Real communication logic (same pattern as vln_ref.py).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.sendall(len(data_bytes).to_bytes(8, "big"))
        sock.sendall(data_bytes)
        size_data = _recv_exact(sock, 8)
        response_size = int.from_bytes(size_data, "big")
        response_data = _recv_exact(sock, response_size)

    response = json.loads(response_data.decode())
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("response", "result", "text", "message"):
            value = response.get(key)
            if isinstance(value, str):
                return value
    raise ValueError(f"Unexpected VLN response type: {type(response)!r}")


def _parse_vln_output_to_atomic_actions(action_text: str):
    """Parse VLN output to atomic actions.

    Supported examples:
    - move 1 / back 1 / right 1 / left 1 / turn 90 / turn -90 / stop
    - The next action is move forward 75 cm.
    - The next action is turn left 45 degrees.
    """
    raw = action_text.strip()
    text = raw.lower().strip()
    text = text[:-1] if text.endswith(".") else text
    text = re.sub(r"\s+", " ", text)

    def _delta_to_atomic(dx: float, dy: float, dyaw_deg: float):
        # Isaac local convention:
        # dx > 0: forward, dy > 0: left, dyaw_deg > 0: turn left.
        # Sequential execution (legacy behavior): translation then yaw.
        linear_eps = 1e-3
        yaw_eps = 1e-2
        acts = []
        if abs(dx) > linear_eps:
            if dx >= 0:
                acts.append(("move", abs(dx)))
            else:
                acts.append(("back", abs(dx)))
        if abs(dy) > linear_eps:
            if dy >= 0:
                acts.append(("left", abs(dy)))
            else:
                acts.append(("right", abs(dy)))
        if abs(dyaw_deg) > yaw_eps:
            if dyaw_deg >= 0:
                acts.append(("turn_left", abs(dyaw_deg)))
            else:
                acts.append(("turn_right", abs(dyaw_deg)))
        return acts

    def _delta_to_twist(dx: float, dy: float, dyaw_deg: float):
        # Isaac local convention:
        # dx > 0: forward, dy > 0: left, dyaw_deg > 0: turn left.
        # Convert one delta into one simultaneous twist segment (vx, vy, wz, duration).
        linear_eps = 1e-3
        yaw_eps = 1e-2
        if abs(dx) <= linear_eps and abs(dy) <= linear_eps and abs(dyaw_deg) <= yaw_eps:
            return []

        max_lin = max(float(args_cli.lin_speed), 1e-6)      # m/s
        max_ang = max(float(args_cli.ang_speed), 1e-6)      # rad/s
        dyaw_rad = float(dyaw_deg) * float(np.pi) / 180.0

        tx = abs(dx) / max_lin if abs(dx) > linear_eps else 0.0
        ty = abs(dy) / max_lin if abs(dy) > linear_eps else 0.0
        tyaw = abs(dyaw_rad) / max_ang if abs(dyaw_rad) > 1e-6 else 0.0
        duration = max(tx, ty, tyaw, 0.05)

        vx = float(np.clip(dx / duration, -max_lin, max_lin))
        vy = float(np.clip(dy / duration, -max_lin, max_lin))
        wz = float(np.clip(dyaw_rad / duration, -max_ang, max_ang))

        # Pack original delta for debug in controller.
        return [("twist", (vx, vy, wz, duration, dx, dy, dyaw_deg))]

    def _delta_to_actions(dx: float, dy: float, dyaw_deg: float):
        mode = str(getattr(args_cli, "delta_exec_mode", "twist")).lower().strip()
        if mode == "atomic":
            return _delta_to_atomic(dx, dy, dyaw_deg)
        return _delta_to_twist(dx, dy, dyaw_deg)

    # JSON dict mode:
    # {"delta_x": 0.12, "delta_y": -0.01, "delta_yaw_deg": 1.5}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            if bool(parsed.get("stop", False)):
                return True, []
            if all(k in parsed for k in ("delta_x", "delta_y", "delta_yaw_deg")):
                dx = float(parsed["delta_x"])
                dy = float(parsed["delta_y"])
                dyaw = float(parsed["delta_yaw_deg"])
                actions = _delta_to_actions(dx, dy, dyaw)
                if len(actions) == 0:
                    return True, []
                return False, actions
    except Exception:
        pass

    # delta plain text mode:
    # "delta 0.123 -0.045 2.500"
    # "deltax=0.123 deltay=-0.045 deltayaw=2.5"
    m = re.fullmatch(
        r"delta\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        text,
    )
    if m:
        dx = float(m.group(1))
        dy = float(m.group(2))
        dyaw = float(m.group(3))
        actions = _delta_to_actions(dx, dy, dyaw)
        if len(actions) == 0:
            return True, []
        return False, actions

    m = re.search(
        r"deltax\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\D+"
        r"deltay\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\D+"
        r"deltayaw(?:_deg)?\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        text,
    )
    if m:
        dx = float(m.group(1))
        dy = float(m.group(2))
        dyaw = float(m.group(3))
        actions = _delta_to_actions(dx, dy, dyaw)
        if len(actions) == 0:
            return True, []
        return False, actions

    # Atomic command format
    m = re.fullmatch(r"(move|back|right|left)\s+(-?\d+(?:\.\d+)?)", text)
    if m:
        cmd = m.group(1)
        val = float(m.group(2))
        if cmd in {"move", "back", "right", "left"}:
            return False, [(cmd, abs(val))]
    m = re.fullmatch(r"turn\s+(-?\d+(?:\.\d+)?)", text)
    if m:
        deg = float(m.group(1))
        if deg >= 0:
            return False, [("turn_left", abs(deg))]
        return False, [("turn_right", abs(deg))]
    if text == "stop":
        return True, []

    # Natural language style
    m = re.search(r"(?:the next action is )?move forward (\d+(?:\.\d+)?)\s*cm", text)
    if m:
        return False, [("move", float(m.group(1)) / 100.0)]
    m = re.search(r"(?:the next action is )?move backward (\d+(?:\.\d+)?)\s*cm", text)
    if m:
        return False, [("back", float(m.group(1)) / 100.0)]
    m = re.search(r"(?:the next action is )?turn left (\d+(?:\.\d+)?)\s*degrees?", text)
    if m:
        return False, [("turn_left", float(m.group(1)))]
    m = re.search(r"(?:the next action is )?turn right (\d+(?:\.\d+)?)\s*degrees?", text)
    if m:
        return False, [("turn_right", float(m.group(1)))]
    if re.search(r"(?:the next action is )?stop", text):
        return True, []

    raise ValueError(f"Unsupported VLN output: '{raw}'")


class VlnCommandController:
    """Controller that executes atomic actions generated by a VLN policy output."""

    def __init__(self, dt: float, lin_speed: float, ang_speed: float):
        self._dt = dt
        self._lin_speed = lin_speed
        self._ang_speed = ang_speed
        self._queue = deque()
        self._photo_requests = deque()
        self._goal_requests = deque()
        self._pose_requests = deque()
        self._set_pose_requests = deque()
        self._rescue_requests = deque()
        self._current = torch.zeros(3, dtype=torch.float32)
        self._remaining_steps = 0
        self._camera_follow_enabled = True
        self._cancel_vln_requested = False
        self._lock = threading.Lock()

        self._print_help()
        self._thread = threading.Thread(target=self._repl, daemon=True)
        self._thread.start()

    def _print_help(self):
        print(
            "[INTERACTIVE] Commands:\n"
            "  help\n"
            "  stop                            (clear actions and stop current VLN round)\n"
            "  goal <instruction>             (send instruction + image to VLN)\n"
            "  vln <instruction>              (alias of goal)\n"
            "  move <m> | back <m> | left <m> | right <m>\n"
            "  turn <deg>                     (+left, -right)\n"
            "  wait <s>\n"
            "  pose                            (print current robot world pose)\n"
            "  set_pose <x> <y> <z> [yaw_deg] (teleport robot; keep yaw if omitted)\n"
            "  rescue [up_m] [back_m]         (lift + move backward to escape clipping)\n"
            "  script move:1.0,turn_left:90,move:0.5\n"
            "  photo [filename]               (save mounted front and bottom camera images)\n"
            "  cam follow | cam free          (toggle viewport follow mode)"
        )

    def _steps_for_time(self, seconds: float) -> int:
        return max(1, int(round(seconds / self._dt)))

    def _enqueue_atomic(self, name: str, value: float):
        # Note: in this task setup, forward corresponds to negative x command.
        if name == "move":
            duration = abs(value) / max(self._lin_speed, 1e-6)
            cmd = torch.tensor([-self._lin_speed if value >= 0 else self._lin_speed, 0.0, 0.0])
        elif name == "back":
            duration = abs(value) / max(self._lin_speed, 1e-6)
            cmd = torch.tensor([self._lin_speed if value >= 0 else -self._lin_speed, 0.0, 0.0])
        elif name == "left":
            duration = abs(value) / max(self._lin_speed, 1e-6)
            cmd = torch.tensor([0.0, self._lin_speed if value >= 0 else -self._lin_speed, 0.0])
        elif name == "right":
            duration = abs(value) / max(self._lin_speed, 1e-6)
            cmd = torch.tensor([0.0, -self._lin_speed if value >= 0 else self._lin_speed, 0.0])
        elif name == "turn_left":
            angle_rad = abs(value) * float(torch.pi) / 180.0
            duration = angle_rad / max(self._ang_speed, 1e-6)
            cmd = torch.tensor([0.0, 0.0, self._ang_speed if value >= 0 else -self._ang_speed])
        elif name == "turn_right":
            angle_rad = abs(value) * float(torch.pi) / 180.0
            duration = angle_rad / max(self._ang_speed, 1e-6)
            cmd = torch.tensor([0.0, 0.0, -self._ang_speed if value >= 0 else self._ang_speed])
        elif name == "wait":
            duration = abs(value)
            cmd = torch.tensor([0.0, 0.0, 0.0])
        elif name == "twist":
            if not isinstance(value, (list, tuple)) or len(value) < 4:
                raise ValueError("twist value must be (vx, vy, wz, duration[, ...])")
            vx_local = float(value[0])  # Isaac local forward(+)
            vy_local = float(value[1])  # Isaac local left(+)
            wz_local = float(value[2])  # left turn(+), rad/s
            duration = max(abs(float(value[3])), self._dt)

            # Convert Isaac local velocity semantics to this task's command frame.
            # In this task setup, forward corresponds to negative x command.
            cmd_x = -vx_local
            cmd_y = vy_local
            cmd_z = wz_local
            cmd = torch.tensor([cmd_x, cmd_y, cmd_z], dtype=torch.float32)

            if len(value) >= 7:
                dx, dy, dyaw = float(value[4]), float(value[5]), float(value[6])
                print(
                    "[VLN][TWIST]"
                    f" target_delta=({dx:.4f},{dy:.4f},{dyaw:.2f}deg)"
                    f" cmd_local=({vx_local:.4f},{vy_local:.4f},{wz_local:.4f}rad/s)"
                    f" duration={duration:.3f}s"
                )
        else:
            raise ValueError(
                f"Unsupported action '{name}'. "
                "Supported: move, back, left, right, turn_left, turn_right, wait, twist."
            )

        self._queue.append((cmd, self._steps_for_time(duration)))

    def enqueue_atomic_actions(self, atomic_actions):
        for name, value in atomic_actions:
            self._enqueue_atomic(name, value)

    def _repl(self):
        while True:
            try:
                line = input("[INTERACTIVE] > ").strip()
            except EOFError:
                break

            if not line:
                continue

            try:
                parts = line.split()
                cmd = parts[0].lower()

                with self._lock:
                    if cmd == "help":
                        self._print_help()
                    elif cmd == "stop":
                        self._queue.clear()
                        self._remaining_steps = 0
                        self._goal_requests.clear()
                        self._cancel_vln_requested = True
                    elif cmd in {"goal", "vln"}:
                        instruction = line[len(parts[0]) :].strip()
                        if not instruction:
                            raise ValueError(f"Usage: {cmd} <instruction>")
                        self._cancel_vln_requested = False
                        self._goal_requests.append((instruction, cmd == "goal"))
                        print(f"[VLN] Queued instruction: {instruction}")
                    elif cmd in {"move", "back", "left", "right", "wait"}:
                        if len(parts) != 2:
                            raise ValueError(f"Usage: {cmd} <value>")
                        value = float(parts[1])
                        self._enqueue_atomic(cmd, value)
                    elif cmd == "turn":
                        if len(parts) != 2:
                            raise ValueError("Usage: turn <degrees>")
                        degrees = float(parts[1])
                        self._enqueue_atomic("turn_left" if degrees >= 0 else "turn_right", abs(degrees))
                    elif cmd == "script":
                        script = line[len("script") :].strip()
                        if not script:
                            raise ValueError("Usage: script <action_script>")
                        for name, value in _parse_action_script(script):
                            self._enqueue_atomic(name, value)
                    elif cmd == "photo":
                        if len(parts) > 2:
                            raise ValueError("Usage: photo [filename]")
                        filename = parts[1] if len(parts) == 2 else ""
                        self._photo_requests.append(filename)
                    elif cmd == "pose":
                        self._pose_requests.append(True)
                    elif cmd == "set_pose":
                        if len(parts) not in {4, 5}:
                            raise ValueError("Usage: set_pose <x> <y> <z> [yaw_deg]")
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        yaw_deg = float(parts[4]) if len(parts) == 5 else None
                        # Stop current queued motion before teleporting.
                        self._queue.clear()
                        self._remaining_steps = 0
                        self._set_pose_requests.append((x, y, z, yaw_deg))
                    elif cmd == "rescue":
                        if len(parts) > 3:
                            raise ValueError("Usage: rescue [up_m] [back_m]")
                        up_m = abs(float(parts[1])) if len(parts) >= 2 else 0.20
                        back_m = abs(float(parts[2])) if len(parts) >= 3 else 0.35
                        # Stop current queued motion/VLN before rescue teleport.
                        self._queue.clear()
                        self._remaining_steps = 0
                        self._goal_requests.clear()
                        self._cancel_vln_requested = True
                        self._rescue_requests.append((up_m, back_m))
                    elif cmd == "cam":
                        if len(parts) != 2 or parts[1].lower() not in {"follow", "free"}:
                            raise ValueError("Usage: cam follow|free")
                        self._camera_follow_enabled = parts[1].lower() == "follow"
                        mode = "follow" if self._camera_follow_enabled else "free"
                        print(f"[INTERACTIVE] Camera mode -> {mode}")
                    else:
                        raise ValueError(f"Unknown command '{cmd}'. Type 'help' for supported commands.")
            except Exception as err:
                print(f"[INTERACTIVE][ERROR] {err}")

    def advance(self):
        with self._lock:
            if self._remaining_steps <= 0:
                if self._queue:
                    self._current, self._remaining_steps = self._queue.popleft()
                else:
                    self._current = torch.zeros(3, dtype=torch.float32)
                    self._remaining_steps = 1
            self._remaining_steps -= 1
            return self._current.tolist()

    def pop_photo_request(self):
        with self._lock:
            if len(self._photo_requests) == 0:
                return None
            return self._photo_requests.popleft()

    def is_camera_follow_enabled(self):
        with self._lock:
            return self._camera_follow_enabled

    def pop_goal_request(self):
        with self._lock:
            if len(self._goal_requests) == 0:
                return None
            return self._goal_requests.popleft()

    def pop_pose_request(self):
        with self._lock:
            if len(self._pose_requests) == 0:
                return None
            return self._pose_requests.popleft()

    def pop_set_pose_request(self):
        with self._lock:
            if len(self._set_pose_requests) == 0:
                return None
            return self._set_pose_requests.popleft()

    def pop_rescue_request(self):
        with self._lock:
            if len(self._rescue_requests) == 0:
                return None
            return self._rescue_requests.popleft()

    def has_pending_actions(self):
        with self._lock:
            return self._remaining_steps > 0 or len(self._queue) > 0

    def consume_vln_cancel_request(self) -> bool:
        with self._lock:
            if not self._cancel_vln_requested:
                return False
            self._cancel_vln_requested = False
            return True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent using interactive controls."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    disable_scene_override = args_cli.disable_scene_override or args_cli.disable_hospital_scene
    scene_usd = None

    if not disable_scene_override:
        if args_cli.hospital:
            scene_usd = _resolve_hospital_usd_path()
            if scene_usd is None:
                raise RuntimeError(
                    "Hospital scene USD not found from default NVIDIA locations. "
                    "Check asset availability or use --scene_usd <omniverse_or_local_usd>."
                )
            print(f"[INFO] Using hospital scene: {scene_usd}")
        else:
            scene_usd_arg = args_cli.scene_usd
            if args_cli.hospital_usd:
                scene_usd_arg = args_cli.hospital_usd
            scene_usd = _resolve_scene_usd_path(scene_usd_arg)
            if scene_usd is None:
                raise RuntimeError(
                    "Scene USD not found. "
                    "Pass --scene_usd <omniverse_or_local_usd> or use --disable_scene_override."
                )
            print(f"[INFO] Using scene USD: {scene_usd}")

    if scene_usd is not None:
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.usd_path = scene_usd
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.max_init_terrain_level = None
        # Disable terrain-generator specific termination for USD scene.
        env_cfg.terminations.terrain_out_of_bounds = None
    else:
        env_cfg.scene.terrain.max_init_terrain_level = None
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels = None
    env_cfg.terminations.time_out = None
    env_cfg.commands.base_velocity.debug_vis = False

    controller = VlnCommandController(
        dt=_infer_step_dt(env_cfg),
        lin_speed=args_cli.lin_speed,
        ang_speed=args_cli.ang_speed,
    )
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
    )
    _configure_mounted_cameras(env_cfg)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(policy=policy_nn, normalizer=None, path=export_model_dir, filename="policy.onnx")
    export_policy_as_jit(policy=policy_nn, normalizer=None, path=export_model_dir, filename="policy.pt")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    recorder = VlnTrajectoryRecorder(
        data_dir=args_cli.data_dir,
        dt=dt,
        action_chunk_size=args_cli.action_chunk_size,
        stop_tail_seconds=args_cli.stop_tail_seconds,
    )
    active_goal_instruction = None
    active_goal_image_history = []
    active_goal_iterations = 0
    use_dummy_vln = args_cli.dummy_vln and not args_cli.use_real_vln
    print(f"[INFO] VLN mode: {'real' if not use_dummy_vln else 'dummy'}")
    print(f"[INFO] Delta execution mode: {args_cli.delta_exec_mode}")

    timestep = 0
    simulation_step = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            try:
                recorder.record_frame(env, actions, simulation_step)
            except Exception as err:
                print(f"[DATA][ERROR] Failed to record aligned frame: {err}")
                recorder.finalize("recording_error")
            obs, _, _, _ = env.step(actions)
        simulation_step += 1

        if controller.consume_vln_cancel_request():
            if active_goal_instruction is not None:
                print("[VLN] Round interrupted by user stop.")
            active_goal_instruction = None
            active_goal_image_history = []
            active_goal_iterations = 0
            recorder.request_stop("manual_stop")

        goal_request = controller.pop_goal_request()
        if goal_request is not None:
            active_goal_instruction, should_record = goal_request
            active_goal_image_history = []
            active_goal_iterations = 0
            print(f"[VLN] Start round with goal: {active_goal_instruction}")
            if should_record:
                recorder.start(active_goal_instruction)

        # VLN closed-loop: keep querying and acting until output is stop.
        if active_goal_instruction is not None and not controller.has_pending_actions():
            if active_goal_iterations >= args_cli.vln_max_iterations:
                print(f"[VLN] Round finished (reached max iterations: {args_cli.vln_max_iterations}).")
                active_goal_instruction = None
                recorder.request_stop("max_iterations")
                continue
            obs_name = f"{args_cli.vln_photo_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            try:
                obs_path = _capture_front_camera_image(env, args_cli.photo_dir, obs_name)
                print(f"[VLN] Observation image saved: {obs_path}")
                active_goal_image_history.append(obs_path)
                selected_paths = _select_images_for_vln(active_goal_image_history, args_cli.vln_max_images)
                print(f'Selected paths : {selected_paths}.')
                print(f"[VLN] Sending {len(selected_paths)} image(s) with instruction.")
                if use_dummy_vln:
                    vln_output = dummy_vln_api(active_goal_instruction, selected_paths)
                else:
                    # Real VLN call (same protocol as vln_ref.py):
                    # base64 image + query text over length-prefixed TCP.
                    vln_output = query_vln_server(
                        user_instruction=active_goal_instruction,
                        image_paths=selected_paths,
                        host=args_cli.vln_host,
                        port=args_cli.vln_port,
                    )
                active_goal_iterations += 1
                print(f"[VLN] Output: {vln_output}")
                is_stop, atomic_actions = _parse_vln_output_to_atomic_actions(vln_output)
                if is_stop:
                    print("[VLN] Round finished (stop). Waiting for next goal.")
                    active_goal_instruction = None
                    recorder.request_stop("vln_stop")
                else:
                    controller.enqueue_atomic_actions(atomic_actions)
            except Exception as err:
                print(f"[VLN][ERROR] Failed VLN round for goal '{active_goal_instruction}': {err}")
                active_goal_instruction = None
                recorder.request_stop("vln_error")

        photo_request = controller.pop_photo_request()
        if photo_request is not None:
            photo_name = photo_request.strip()
            if not photo_name:
                photo_name = datetime.datetime.now().strftime("photo_%Y%m%d_%H%M%S_%f")
            try:
                photo_paths = _capture_mounted_camera_images(env, args_cli.photo_dir, photo_name)
                print("[PHOTO] Saved mounted camera images:")
                for label, photo_path in photo_paths.items():
                    print(f"  {label}: {photo_path}")
            except Exception as err:
                print(f"[PHOTO][ERROR] Failed to save image: {err}")

        pose_request = controller.pop_pose_request()
        if pose_request is not None:
            try:
                x, y, z, qw, qx, qy, qz, yaw_deg = _get_robot_pose(env)
                print(
                    f"[POSE] position=({x:.3f}, {y:.3f}, {z:.3f}), "
                    f"quat_wxyz=({qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f}), yaw={yaw_deg:.2f} deg"
                )
            except Exception as err:
                print(f"[POSE][ERROR] Failed to read robot pose: {err}")

        set_pose_request = controller.pop_set_pose_request()
        if set_pose_request is not None:
            x, y, z, yaw_deg = set_pose_request
            try:
                # Robot buffers may be inference tensors created during env.step() in inference mode.
                # Perform teleport in inference mode to allow in-place updates in simulator internals.
                with torch.inference_mode():
                    _set_robot_pose(env, x=x, y=y, z=z, yaw_deg=yaw_deg)
                x2, y2, z2, _, _, _, _, yaw2 = _get_robot_pose(env)
                if yaw_deg is None:
                    print(
                        f"[POSE] Robot teleported to ({x2:.3f}, {y2:.3f}, {z2:.3f}), "
                        f"yaw kept at {yaw2:.2f} deg"
                    )
                else:
                    print(
                        f"[POSE] Robot teleported to ({x2:.3f}, {y2:.3f}, {z2:.3f}), "
                        f"yaw set to {yaw2:.2f} deg"
                    )
            except Exception as err:
                print(f"[POSE][ERROR] Failed to set robot pose: {err}")

        rescue_request = controller.pop_rescue_request()
        if rescue_request is not None:
            up_m, back_m = rescue_request
            try:
                x, y, z, _, _, _, _, yaw_deg = _get_robot_pose(env)
                yaw_rad = float(yaw_deg) * float(torch.pi) / 180.0
                # In this task setup, robot forward aligns with local -x.
                # Therefore backward offset in world XY is +[cos(yaw), sin(yaw)].
                target_x = x + back_m * float(np.cos(yaw_rad))
                target_y = y + back_m * float(np.sin(yaw_rad))
                target_z = z + up_m
                with torch.inference_mode():
                    _set_robot_pose(env, x=target_x, y=target_y, z=target_z, yaw_deg=yaw_deg)
                print(
                    f"[POSE] Rescue applied: ({x:.3f}, {y:.3f}, {z:.3f}) -> "
                    f"({target_x:.3f}, {target_y:.3f}, {target_z:.3f}), yaw={yaw_deg:.2f} deg"
                )
            except Exception as err:
                print(f"[POSE][ERROR] Failed to rescue robot pose: {err}")

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        if controller.is_camera_follow_enabled():
            camera_follow(env)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    recorder.finalize("simulation_closed")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
