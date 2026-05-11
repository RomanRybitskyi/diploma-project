from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ugv_swarm_expert.constants import (
    ACTION_DIM,
    STATE_FEATURE_COUNT,
)

DEFAULT_REWARD_EPS = 1e-6


@torch.no_grad()
def compute_gail_reward(
    discriminator: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    device: Any = None,
    eps: float = DEFAULT_REWARD_EPS,
) -> torch.Tensor:
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in the open interval (0, 0.5).")

    target_device = torch.device(device) if device is not None else states.device
    states = states.to(target_device)
    actions = actions.to(target_device)
    _validate_reward_inputs(states, actions)

    batch_size, n_agents, _ = states.shape

    joint_state_action = torch.cat((states, actions), dim=2)
    disc_output = discriminator(joint_state_action)

    if disc_output.ndim == 1 and disc_output.shape[0] == batch_size:
        disc_output = disc_output.unsqueeze(-1)

    if disc_output.shape != (batch_size, 1):
        raise ValueError(
            f"Discriminator output must have shape ({batch_size}, 1); " f"got {tuple(disc_output.shape)}."
        )

    probability = torch.clamp(disc_output, min=eps, max=1.0 - eps)
    global_reward = -torch.log1p(-probability)
    return global_reward.expand(batch_size, n_agents).contiguous()


class GAILRewardComputer:
    def __init__(self, discriminator: nn.Module, device: Any = None, eps: float = DEFAULT_REWARD_EPS):
        self.discriminator = discriminator
        self.device = torch.device(device) if device is not None else None
        self.eps = float(eps)
        if self.device is not None:
            self.discriminator.to(self.device)

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return compute_gail_reward(
            discriminator=self.discriminator,
            states=states,
            actions=actions,
            device=self.device,
            eps=self.eps,
        )


def _validate_reward_inputs(states: torch.Tensor, actions: torch.Tensor) -> None:
    if not isinstance(states, torch.Tensor):
        raise TypeError("states must be a torch.Tensor.")
    if not isinstance(actions, torch.Tensor):
        raise TypeError("actions must be a torch.Tensor.")
    if states.ndim != 3 or states.shape[-1] != STATE_FEATURE_COUNT:
        raise ValueError(
            f"states must have shape (Batch, N, {STATE_FEATURE_COUNT}); got {tuple(states.shape)}."
        )
    if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(f"actions must have shape (Batch, N, {ACTION_DIM}); got {tuple(actions.shape)}.")
    if states.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "states and actions must have matching Batch and N dimensions; "
            f"got {tuple(states.shape[:2])} and {tuple(actions.shape[:2])}."
        )
