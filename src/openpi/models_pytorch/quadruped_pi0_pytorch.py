import torch
from torch import nn

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class QuadrupedStateAdapter(nn.Module):
    """Projects quadruped proprioception into the state space used by Pi0."""

    def __init__(self, input_dim: int = 58, hidden_dim: int = 128, output_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.input_dim:
            raise ValueError(f"Expected quadruped state dimension {self.input_dim}, got shape {tuple(state.shape)}.")
        return self.proj(state)


class QuadrupedPI0Pytorch(PI0Pytorch):
    """Pi0 variant that accepts the 58-dimensional M20 proprioceptive state."""

    def __init__(self, config, *, state_dim: int = 58, adapter_hidden_dim: int = 128) -> None:
        if config.pi05:
            raise ValueError("QuadrupedPI0Pytorch currently supports Pi0 only, not Pi0.5.")
        super().__init__(config)
        self.state_adapter = QuadrupedStateAdapter(
            input_dim=state_dim,
            hidden_dim=adapter_hidden_dim,
            output_dim=config.action_dim,
        )

    def embed_suffix(self, state, noisy_actions, timestep):
        return super().embed_suffix(self.state_adapter(state), noisy_actions, timestep)
