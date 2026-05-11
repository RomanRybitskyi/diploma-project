from __future__ import annotations

import torch
from torch import nn

from ugv_swarm_expert.constants import STATE_FEATURE_COUNT, STATE_WINDOW_SIZE
from ugv_swarm_expert.models.weight_initializer import orthogonal_init

STATE_SHAPE = (STATE_WINDOW_SIZE, STATE_FEATURE_COUNT)


class CriticNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        _flat_dim = STATE_WINDOW_SIZE * STATE_FEATURE_COUNT
        self.value_net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(_flat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        orthogonal_init(self)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or tuple(states.shape[1:]) != STATE_SHAPE:
            raise ValueError(
                f"Critic input must have shape (Batch, {STATE_WINDOW_SIZE}, {STATE_FEATURE_COUNT}); "
                f"got {tuple(states.shape)}."
            )
        return self.value_net(states).squeeze(-1)
