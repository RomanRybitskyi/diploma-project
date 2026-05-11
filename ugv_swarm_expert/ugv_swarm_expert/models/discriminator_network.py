from __future__ import annotations

import torch
from torch import nn

from ugv_swarm_expert.constants import DISCRIMINATOR_FEATURE_COUNT
from ugv_swarm_expert.models.weight_initializer import orthogonal_init

AGENT_STATE_ACTION_DIM: int = DISCRIMINATOR_FEATURE_COUNT
LOCAL_EMBED_DIM = 64
ATTENTION_HEADS = 4


class DiscriminatorNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Linear(AGENT_STATE_ACTION_DIM, LOCAL_EMBED_DIM),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=LOCAL_EMBED_DIM,
            num_heads=ATTENTION_HEADS,
            batch_first=True,
        )
        self.evaluator = nn.Sequential(
            nn.Linear(LOCAL_EMBED_DIM, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self._initialize_weights()

    def forward(self, joint_state_action: torch.Tensor) -> torch.Tensor:
        self._validate_input_shape(joint_state_action)

        local_features = self.local_encoder(joint_state_action)
        attention_output, _ = self.attention(
            local_features,
            local_features,
            local_features,
            need_weights=False,
        )
        aggregated_tensor = torch.max(attention_output, dim=1)[0]
        return self.evaluator(aggregated_tensor)

    def _initialize_weights(self) -> None:
        orthogonal_init(self)

    @staticmethod
    def _validate_input_shape(joint_state_action: torch.Tensor) -> None:
        if not isinstance(joint_state_action, torch.Tensor):
            raise TypeError("joint_state_action must be a torch.Tensor.")
        if joint_state_action.ndim != 3 or joint_state_action.shape[-1] != AGENT_STATE_ACTION_DIM:
            raise ValueError(
                f"joint_state_action must have shape (Batch, N, {AGENT_STATE_ACTION_DIM}); "
                f"got {tuple(joint_state_action.shape)}."
            )
        if joint_state_action.shape[1] == 0:
            raise ValueError("joint_state_action must contain at least one agent in dimension 1.")
