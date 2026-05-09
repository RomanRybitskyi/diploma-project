from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.distributions import Distribution

STATE_WINDOW_SIZE = 4
STATE_FEATURE_COUNT = 41
ACTION_FEATURE_COUNT = 2
DISCRIMINATOR_AGENT_FEATURE_COUNT = 43


class SharedActorPlaceholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(STATE_WINDOW_SIZE * STATE_FEATURE_COUNT, 64),
            nn.ReLU(),
            nn.Linear(64, ACTION_FEATURE_COUNT),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or states.shape[1:] != (STATE_WINDOW_SIZE, STATE_FEATURE_COUNT):
            raise ValueError(
                f"Actor input must have shape (N, {STATE_WINDOW_SIZE}, {STATE_FEATURE_COUNT}); "
                f"got {tuple(states.shape)}."
            )
        return self.policy(states)


class SharedDiscriminatorPlaceholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Linear(DISCRIMINATOR_AGENT_FEATURE_COUNT, 64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.evaluator = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, joint_state_action: torch.Tensor) -> torch.Tensor:
        if joint_state_action.ndim != 3 or joint_state_action.shape[-1] != DISCRIMINATOR_AGENT_FEATURE_COUNT:
            raise ValueError(
                f"Discriminator input must have shape (Batch, N, {DISCRIMINATOR_AGENT_FEATURE_COUNT}); "
                f"got {tuple(joint_state_action.shape)}."
            )
        encoded_agents = self.local_encoder(joint_state_action)
        swarm_embedding = encoded_agents.max(dim=1).values
        return self.evaluator(swarm_embedding)


class SwarmManager:
    def __init__(
        self,
        actor=None,
        discriminator=None,
        device=None,
        inference_mode: bool = True,
    ):
        self.device = torch.device(device) if device is not None else self._default_device()
        self.actor = (actor if actor is not None else SharedActorPlaceholder()).to(self.device)
        self.discriminator = (
            discriminator if discriminator is not None else SharedDiscriminatorPlaceholder()
        ).to(self.device)
        self.inference_mode = bool(inference_mode)

    def get_actions(self, agent_states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not agent_states:
            raise ValueError("agent_states must contain at least one agent state.")

        ordered_states = OrderedDict(agent_states)
        batched_states = self._stack_agent_states(ordered_states)

        if self.inference_mode:
            self.actor.eval()
            with torch.no_grad():
                output = self.actor(batched_states)
        else:
            output = self.actor(batched_states)

        if isinstance(output, Distribution):
            batched_actions = output.mean if self.inference_mode else output.rsample()
        else:
            batched_actions = output

        self._validate_batched_actions(batched_actions, expected_agents=len(ordered_states))
        return dict(zip(ordered_states.keys(), batched_actions.unbind(dim=0), strict=False))

    def evaluate_swarm(self, joint_state_action: torch.Tensor) -> torch.Tensor:
        joint_batch = joint_state_action.to(self.device)
        if self.inference_mode:
            self.discriminator.eval()
            with torch.no_grad():
                return self.discriminator(joint_batch)
        return self.discriminator(joint_batch)

    def train(self, mode: bool = True) -> SwarmManager:
        self.actor.train(mode)
        self.discriminator.train(mode)
        return self

    def eval(self) -> SwarmManager:
        return self.train(False)

    def to(self, device) -> SwarmManager:
        self.device = torch.device(device)
        self.actor.to(self.device)
        self.discriminator.to(self.device)
        return self

    def _stack_agent_states(self, agent_states: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        prepared_states = []
        for agent_id, state in agent_states.items():
            if not isinstance(state, torch.Tensor):
                raise TypeError(f"State for agent '{agent_id}' must be a torch.Tensor.")
            if state.shape != (STATE_WINDOW_SIZE, STATE_FEATURE_COUNT):
                raise ValueError(
                    f"State for agent '{agent_id}' must have shape "
                    f"({STATE_WINDOW_SIZE}, {STATE_FEATURE_COUNT}); got {tuple(state.shape)}."
                )
            prepared_states.append(state.to(self.device))
        return torch.stack(prepared_states, dim=0)

    @staticmethod
    def _validate_batched_actions(actions: torch.Tensor, expected_agents: int) -> None:
        if actions.ndim != 2 or actions.shape != (expected_agents, ACTION_FEATURE_COUNT):
            raise ValueError(
                f"Actor output must have shape ({expected_agents}, {ACTION_FEATURE_COUNT}); "
                f"got {tuple(actions.shape)}."
            )

    @staticmethod
    def _default_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
