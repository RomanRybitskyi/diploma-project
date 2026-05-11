from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch


class PPORolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_agents: int,
        state_shape: Sequence[int],
        action_dim: int,
        device: Any = None,
        dtype: torch.dtype = torch.float32,
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be a positive integer.")
        if num_agents <= 0:
            raise ValueError("num_agents must be a positive integer.")
        if action_dim <= 0:
            raise ValueError("action_dim must be a positive integer.")
        if not state_shape:
            raise ValueError("state_shape must contain at least one dimension.")

        self.num_steps = int(num_steps)
        self.num_agents = int(num_agents)
        self.state_shape = tuple(int(dim) for dim in state_shape)
        self.action_dim = int(action_dim)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.step_index = 0

        self.states = torch.zeros(
            (self.num_steps, self.num_agents, *self.state_shape),
            dtype=self.dtype,
            device=self.device,
        )
        self.actions = torch.zeros(
            (self.num_steps, self.num_agents, self.action_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.log_probs = torch.zeros((self.num_steps, self.num_agents), dtype=self.dtype, device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_agents), dtype=self.dtype, device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_agents), dtype=self.dtype, device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_agents), dtype=self.dtype, device=self.device)
        self.advantages = torch.zeros((self.num_steps, self.num_agents), dtype=self.dtype, device=self.device)
        self.returns = torch.zeros((self.num_steps, self.num_agents), dtype=self.dtype, device=self.device)

    @property
    def is_full(self) -> bool:
        return self.step_index >= self.num_steps

    def store(
        self,
        state: Any,
        action: Any,
        log_prob: Any,
        value: Any,
        reward: Any,
        done: Any,
    ) -> None:
        if self.step_index >= self.num_steps:
            raise IndexError("Rollout buffer is full. Call clear() before storing more transitions.")

        index = self.step_index
        self.states[index].copy_(self._as_tensor(state, expected_shape=(self.num_agents, *self.state_shape)))
        self.actions[index].copy_(self._as_tensor(action, expected_shape=(self.num_agents, self.action_dim)))
        self.log_probs[index].copy_(self._as_tensor(log_prob, expected_shape=(self.num_agents,)))
        self.values[index].copy_(self._as_tensor(value, expected_shape=(self.num_agents,)))
        self.rewards[index].copy_(self._as_tensor(reward, expected_shape=(self.num_agents,)))
        self.dones[index].copy_(self._as_tensor(done, expected_shape=(self.num_agents,)))
        self.step_index += 1

    def compute_returns_and_advantages(
        self,
        last_value: Any,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        if self.step_index != self.num_steps:
            raise RuntimeError("Cannot compute GAE until the rollout buffer is full.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")

        next_value = self._as_tensor(last_value, expected_shape=(self.num_agents,))
        next_advantage = torch.zeros(self.num_agents, dtype=self.dtype, device=self.device)

        for step in reversed(range(self.num_steps)):
            not_done = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * not_done - self.values[step]
            next_advantage = delta + gamma * gae_lambda * not_done * next_advantage
            self.advantages[step] = next_advantage
            self.returns[step] = self.advantages[step] + self.values[step]
            next_value = self.values[step]

    def get_batches(self, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.step_index != self.num_steps:
            raise RuntimeError("Cannot create PPO batches until the rollout buffer is full.")

        total_samples = self.num_steps * self.num_agents
        permutation = torch.randperm(total_samples, device=self.device)
        flat = self._flatten()

        for start in range(0, total_samples, batch_size):
            indices = permutation[start : start + batch_size]
            yield {name: tensor.index_select(0, indices) for name, tensor in flat.items()}

    def clear(self, zero_tensors: bool = False) -> None:
        self.step_index = 0
        if zero_tensors:
            for tensor in (
                self.states,
                self.actions,
                self.log_probs,
                self.values,
                self.rewards,
                self.dones,
                self.advantages,
                self.returns,
            ):
                tensor.zero_()

    def _flatten(self) -> dict[str, torch.Tensor]:
        return {
            "states": self.states.reshape(self.num_steps * self.num_agents, *self.state_shape),
            "actions": self.actions.reshape(self.num_steps * self.num_agents, self.action_dim),
            "log_probs": self.log_probs.reshape(-1),
            "returns": self.returns.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "values": self.values.reshape(-1),
        }

    def _as_tensor(self, data: Any, expected_shape: tuple[int, ...]) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=self.device, dtype=self.dtype)
        else:
            tensor = torch.as_tensor(data, dtype=self.dtype, device=self.device)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Expected tensor shape {expected_shape}, got {tuple(tensor.shape)}.")
        return tensor
