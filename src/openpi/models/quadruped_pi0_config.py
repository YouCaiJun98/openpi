import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.quadruped_pi0 import QuadrupedPi0


@dataclasses.dataclass(frozen=True)
class QuadrupedPi0Config(pi0_config.Pi0Config):
    """Pi0 configuration for the M20 quadruped state and action spaces."""

    state_dim: int = 58
    adapter_hidden_dim: int = 128
    action_horizon: int = 32

    @override
    def create(self, rng: at.KeyArrayLike) -> "QuadrupedPi0":
        from openpi.models.quadruped_pi0 import QuadrupedPi0

        return QuadrupedPi0(
            self,
            rngs=nnx.Rngs(rng),
            state_dim=self.state_dim,
            adapter_hidden_dim=self.adapter_hidden_dim,
        )

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
            )
        return observation_spec, action_spec

    def get_quadruped_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freezes the backbone while keeping the quadruped adaptation layers trainable."""
        trainable_paths = (
            "state_adapter/.*|"
            "state_proj/.*|"
            "action_in_proj/.*|"
            "action_out_proj/.*|"
            "action_time_mlp_in/.*|"
            "action_time_mlp_out/.*"
        )
        return nnx.Not(nnx_utils.PathRegex(trainable_paths))
