from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from ugv_swarm_expert.weight_initializer import init_ppo_action_head, orthogonal_init

STATE_WINDOW_SIZE = 4
STATE_FEATURE_COUNT = 41
KINEMATIC_FEATURE_COUNT = 5
LIDAR_FEATURE_COUNT = 36
ACTION_FEATURE_COUNT = 2


class ActorNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=LIDAR_FEATURE_COUNT,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * 2, 32),
            nn.ReLU(),
        )

        self.kinematic_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(STATE_WINDOW_SIZE * KINEMATIC_FEATURE_COUNT, 32),
            nn.ReLU(),
        )

        self.core_mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        self.action_mean = nn.Sequential(
            nn.Linear(64, ACTION_FEATURE_COUNT),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(ACTION_FEATURE_COUNT))

        self._initialize_weights()

    def forward(self, state: torch.Tensor) -> Normal:
        self._validate_state_shape(state)

        kinematic_state = state[:, :, :KINEMATIC_FEATURE_COUNT]
        lidar_state = state[:, :, KINEMATIC_FEATURE_COUNT:]

        lidar_embedding = self.lidar_encoder(lidar_state.permute(0, 2, 1).contiguous())
        kinematic_embedding = self.kinematic_encoder(kinematic_state)

        fused = torch.cat((lidar_embedding, kinematic_embedding), dim=-1)
        latent = self.core_mlp(fused)
        mu = self.action_mean(latent)
        std = torch.exp(self.log_std).expand_as(mu)
        return Normal(mu, std)

    def _initialize_weights(self) -> None:
        orthogonal_init(self)
        init_ppo_action_head(self.action_mean)

    @staticmethod
    def _validate_state_shape(state: torch.Tensor) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a torch.Tensor.")
        expected_tail = (STATE_WINDOW_SIZE, STATE_FEATURE_COUNT)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected_tail:
            raise ValueError(
                f"state must have shape (Batch, {STATE_WINDOW_SIZE}, {STATE_FEATURE_COUNT}); "
                f"got {tuple(state.shape)}."
            )
