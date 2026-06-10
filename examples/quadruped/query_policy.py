"""Standalone query client for a served quadruped policy.

Example:
    python examples/quadruped/query_policy.py
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

ACTION_HORIZON = 32
ACTION_DIM = 16


def _require_numpy():
    if np is None:
        raise RuntimeError(
            "Standalone query requires lightweight dependencies: "
            "python -m pip install numpy msgpack websockets"
        )
    return np


def _pack_array(obj):
    np_mod = _require_numpy()
    if isinstance(obj, np_mod.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np_mod.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    np_mod = _require_numpy()
    if b"__ndarray__" in obj:
        return np_mod.ndarray(buffer=obj[b"data"], dtype=np_mod.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np_mod.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class StandalonePolicyClient:
    """Minimal openpi-compatible WebSocket/msgpack client.

    This intentionally avoids importing ``openpi_client`` so it can run in a
    small generic Python environment.
    """

    def __init__(self, host: str, port: int, *, api_key: str | None = None, retry_interval_s: float = 5.0):
        self._uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._api_key = api_key
        self._retry_interval_s = retry_interval_s
        try:
            import msgpack
            import websockets.sync.client
        except ImportError as err:
            raise RuntimeError(
                "Standalone query requires lightweight dependencies: "
                "python -m pip install numpy msgpack websockets"
            ) from err
        self._msgpack = msgpack
        self._websockets_client = websockets.sync.client
        self._packer = msgpack.Packer(default=_pack_array)
        self._ws = None
        self._metadata = None
        self._connect()

    @property
    def metadata(self) -> dict:
        return self._metadata

    def _connect_once(self):
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        try:
            return self._websockets_client.connect(
                self._uri,
                compression=None,
                max_size=None,
                additional_headers=headers,
            )
        except TypeError:
            return self._websockets_client.connect(
                self._uri,
                compression=None,
                max_size=None,
                extra_headers=headers,
            )

    def _connect(self):
        logging.info("Waiting for policy server at %s ...", self._uri)
        while True:
            try:
                self._ws = self._connect_once()
                self._metadata = self._msgpack.unpackb(self._ws.recv(), object_hook=_unpack_array)
                return
            except ConnectionRefusedError:
                logging.info("Server is not ready; retrying in %.1fs ...", self._retry_interval_s)
                time.sleep(self._retry_interval_s)

    def infer(self, observation: dict) -> dict:
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._msgpack.unpackb(response, object_hook=_unpack_array)


def _get_mapping_value(mapping: dict, key: str) -> Any:
    if key in mapping:
        return mapping[key]
    byte_key = key.encode()
    if byte_key in mapping:
        return mapping[byte_key]
    raise KeyError(key)


@dataclasses.dataclass
class Args:
    """Arguments for querying a served quadruped policy."""

    host: str = "127.0.0.1"
    port: int = 8000
    num_queries: int = 3
    seed: int = 0
    prompt: str = "walk forward toward the doorway"
    api_key: str | None = None
    retry_interval_s: float = 5.0


def make_observation(rng, prompt: str) -> dict:
    """Creates one inference request with the same fields expected from Isaac Sim."""
    np_mod = _require_numpy()
    return {
        "observation/front_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np_mod.uint8),
        "observation/bottom_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np_mod.uint8),
        "observation/base_ang_vel": rng.normal(0.0, 0.08, size=3).astype(np_mod.float32),
        "observation/projected_gravity": np_mod.array([0.0, 0.0, -1.0], dtype=np_mod.float32),
        "observation/normal_force": rng.uniform(35.0, 95.0, size=4).astype(np_mod.float32),
        "observation/joint_pos": rng.normal(0.0, 0.3, size=16).astype(np_mod.float32),
        "observation/joint_vel": rng.normal(0.0, 1.0, size=16).astype(np_mod.float32),
        "observation/last_action": np_mod.zeros(16, dtype=np_mod.float32),
        "prompt": prompt,
    }


def main(args: Args) -> None:
    client = StandalonePolicyClient(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        retry_interval_s=args.retry_interval_s,
    )
    logging.info("Server metadata: %s", client.metadata)
    np_mod = _require_numpy()
    rng = np_mod.random.default_rng(args.seed)

    for query_index in range(args.num_queries):
        start_time = time.monotonic()
        response = client.infer(make_observation(rng, args.prompt))
        elapsed_ms = (time.monotonic() - start_time) * 1000
        actions = np_mod.asarray(_get_mapping_value(response, "actions"))

        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"Expected actions with shape {(ACTION_HORIZON, ACTION_DIM)}, got {actions.shape}.")
        if not np_mod.all(np_mod.isfinite(actions)):
            raise ValueError("Policy returned non-finite actions.")

        print(f"query={query_index} round_trip_ms={elapsed_ms:.1f} action_shape={actions.shape}")
        print(f"first_action={np_mod.array2string(actions[0], precision=4, suppress_small=True)}")
        print(f"server_timing={response.get('server_timing', response.get(b'server_timing', {}))}")
        print(f"policy_timing={response.get('policy_timing', response.get(b'policy_timing', {}))}")

    print("Quadruped serve/query smoke test passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--num-queries", type=int, default=Args.num_queries)
    parser.add_argument("--seed", type=int, default=Args.seed)
    parser.add_argument("--prompt", default=Args.prompt)
    parser.add_argument("--api-key", default=Args.api_key)
    parser.add_argument("--retry-interval-s", type=float, default=Args.retry_interval_s)
    main(Args(**vars(parser.parse_args())))
