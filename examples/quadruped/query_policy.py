"""Query a served quadruped policy with synthetic observations.

Example:
    uv run examples/quadruped/query_policy.py
"""

import dataclasses
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy
import tyro

ACTION_HORIZON = 32
ACTION_DIM = 16


@dataclasses.dataclass
class Args:
    """Arguments for querying a served quadruped policy."""

    host: str = "127.0.0.1"
    port: int = 8000
    num_queries: int = 3
    seed: int = 0
    prompt: str = "walk forward toward the doorway"


def make_observation(rng: np.random.Generator, prompt: str) -> dict:
    """Creates one inference request with the same fields expected from Isaac Sim."""
    return {
        "observation/front_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/bottom_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/base_ang_vel": rng.normal(0.0, 0.08, size=3).astype(np.float32),
        "observation/projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "observation/normal_force": rng.uniform(35.0, 95.0, size=4).astype(np.float32),
        "observation/joint_pos": rng.normal(0.0, 0.3, size=16).astype(np.float32),
        "observation/joint_vel": rng.normal(0.0, 1.0, size=16).astype(np.float32),
        "observation/last_action": np.zeros(16, dtype=np.float32),
        "prompt": prompt,
    }


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", client.get_server_metadata())
    rng = np.random.default_rng(args.seed)

    for query_index in range(args.num_queries):
        start_time = time.monotonic()
        response = client.infer(make_observation(rng, args.prompt))
        elapsed_ms = (time.monotonic() - start_time) * 1000
        actions = np.asarray(response["actions"])

        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"Expected actions with shape {(ACTION_HORIZON, ACTION_DIM)}, got {actions.shape}.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("Policy returned non-finite actions.")

        print(f"query={query_index} round_trip_ms={elapsed_ms:.1f} action_shape={actions.shape}")
        print(f"first_action={np.array2string(actions[0], precision=4, suppress_small=True)}")
        print(f"server_timing={response.get('server_timing', {})}")
        print(f"policy_timing={response.get('policy_timing', {})}")

    print("Quadruped serve/query smoke test passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
