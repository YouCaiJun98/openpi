"""Serve the latest quadruped checkpoint over WebSocket.

Example:
    uv run examples/quadruped/serve_checkpoint.py
"""

import dataclasses
import logging
import pathlib

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for serving a trained quadruped policy."""

    config_name: str = "pi0_quadruped_synthetic_base"
    exp_name: str = "synthetic_base"
    checkpoint_base_dir: pathlib.Path = pathlib.Path("checkpoints")
    checkpoint_dir: pathlib.Path | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str | None = None
    record: bool = False


def find_latest_checkpoint(args: Args) -> pathlib.Path:
    """Returns an explicit checkpoint or the latest numeric step for an experiment."""
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return checkpoint_dir

    experiment_dir = args.checkpoint_base_dir / args.config_name / args.exp_name
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment checkpoint directory does not exist: {experiment_dir}")

    checkpoints = [path for path in experiment_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    if not checkpoints:
        raise FileNotFoundError(f"No numeric checkpoint steps found in: {experiment_dir}")
    return max(checkpoints, key=lambda path: int(path.name))


def main(args: Args) -> None:
    checkpoint_dir = find_latest_checkpoint(args)
    logging.info("Loading quadruped checkpoint: %s", checkpoint_dir)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config_name),
        checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records/quadruped")

    logging.info("Serving quadruped policy on ws://%s:%s", args.host, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
