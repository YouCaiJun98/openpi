import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at


class QuadrupedStateAdapter(nnx.Module):
    """Projects quadruped proprioception into the state space used by Pi0."""

    def __init__(self, input_dim: int = 58, hidden_dim: int = 128, output_dim: int = 32, *, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = nnx.Param(jnp.ones((input_dim,)))
        self.bias = nnx.Param(jnp.zeros((input_dim,)))
        self.linear_in = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.linear_out = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, state: at.Array) -> at.Array:
        if state.shape[-1] != self.input_dim:
            raise ValueError(f"Expected quadruped state dimension {self.input_dim}, got shape {state.shape}.")
        mean = jnp.mean(state, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(state - mean), axis=-1, keepdims=True)
        state = (state - mean) * jax.lax.rsqrt(variance + 1e-6)
        state = state * self.scale + self.bias
        state = self.linear_in(state)
        state = nnx.swish(state)
        return self.linear_out(state)


class QuadrupedPi0(_pi0.Pi0):
    """Pi0 variant that accepts the 58-dimensional M20 proprioceptive state."""

    def __init__(
        self,
        config: pi0_config.Pi0Config,
        rngs: nnx.Rngs,
        *,
        state_dim: int = 58,
        adapter_hidden_dim: int = 128,
    ):
        if config.pi05:
            raise ValueError("QuadrupedPi0 currently supports Pi0 only, not Pi0.5.")
        super().__init__(config, rngs)
        self.state_adapter = QuadrupedStateAdapter(
            input_dim=state_dim,
            hidden_dim=adapter_hidden_dim,
            output_dim=config.action_dim,
            rngs=rngs,
        )

    @override
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        adapted_obs = dataclasses.replace(obs, state=self.state_adapter(obs.state))
        return super().embed_suffix(adapted_obs, noisy_actions, timestep)
