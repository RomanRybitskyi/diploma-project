from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

STATE_FEATURE_COUNT = 41
ACTION_FEATURE_COUNT = 2
DISCRIMINATOR_FEATURE_COUNT = STATE_FEATURE_COUNT + ACTION_FEATURE_COUNT


class MAGAILTrainer:
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        discriminator: nn.Module,
        rollout_buffer=None,
        expert_dataloader: Any = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        disc_lr: float = 3e-4,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 10,
        disc_epochs: int = 3,
        max_grad_norm: float = 0.5,
        device: Any = None,
    ):
        if ppo_epochs <= 0:
            raise ValueError("ppo_epochs must be positive.")
        if disc_epochs <= 0:
            raise ValueError("disc_epochs must be positive.")
        if clip_ratio <= 0.0:
            raise ValueError("clip_ratio must be positive.")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive.")

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.rollout_buffer = rollout_buffer
        self.expert_dataloader = expert_dataloader

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=disc_lr)

        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.ppo_epochs = int(ppo_epochs)
        self.disc_epochs = int(disc_epochs)
        self.max_grad_norm = float(max_grad_norm)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def update_discriminator(self, expert_batch: Any, generated_batch: Any) -> dict[str, float]:
        expert_states, expert_actions = self._unpack_state_action_batch(expert_batch)
        gen_states, gen_actions = self._unpack_state_action_batch(generated_batch)
        expert_joint = self._prepare_joint_state_action(expert_states, expert_actions).detach()
        gen_joint = self._prepare_joint_state_action(gen_states, gen_actions).detach()

        self.discriminator.train()
        loss_value = expert_loss_value = gen_loss_value = 0.0
        for _ in range(self.disc_epochs):
            expert_preds = self.discriminator(expert_joint)
            gen_preds = self.discriminator(gen_joint)
            expert_targets = torch.ones_like(expert_preds)
            gen_targets = torch.zeros_like(gen_preds)
            expert_loss = self.bce_loss(expert_preds, expert_targets)
            gen_loss = self.bce_loss(gen_preds, gen_targets)
            disc_loss = 0.5 * (expert_loss + gen_loss)

            self.disc_optimizer.zero_grad(set_to_none=True)
            disc_loss.backward()
            self.disc_optimizer.step()

            loss_value = float(disc_loss.detach().cpu())
            expert_loss_value = float(expert_loss.detach().cpu())
            gen_loss_value = float(gen_loss.detach().cpu())

        return {
            "disc_loss": loss_value,
            "expert_loss": expert_loss_value,
            "generated_loss": gen_loss_value,
        }

    def update_ppo(self, rollout_buffer=None, batch_size: int = 128) -> dict[str, float]:
        buffer = rollout_buffer or self.rollout_buffer
        if buffer is None:
            raise ValueError("A PPORolloutBuffer must be provided.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        with torch.no_grad():
            advantages = buffer.advantages
            buffer.advantages.copy_(
                (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            )

        self.actor.train()
        self.critic.train()
        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        update_count = 0

        for _ in range(self.ppo_epochs):
            for batch in buffer.get_batches(batch_size):
                states = batch["states"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device).detach()
                returns = batch["returns"].to(self.device).detach()
                advantages = batch["advantages"].to(self.device).detach()

                distribution = self.actor(states)
                new_log_probs = self._sum_action_dimension(distribution.log_prob(actions))
                entropy = self._sum_action_dimension(distribution.entropy()).mean()
                ratio = torch.exp(new_log_probs - old_log_probs)
                surrogate_1 = ratio * advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                actor_loss = -torch.min(surrogate_1, surrogate_2).mean() - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                values = self._critic_values(states)
                critic_loss = self.mse_loss(values, returns)

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean()
                metrics = {
                    "actor_loss": float(actor_loss.detach().cpu()),
                    "critic_loss": float(critic_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "approx_kl": float(approx_kl.detach().cpu()),
                }
                update_count += 1

        metrics["num_updates"] = float(update_count)
        return metrics

    def _critic_values(self, states: torch.Tensor) -> torch.Tensor:
        values = self.critic(states)
        if isinstance(values, tuple):
            values = values[0]
        if values.ndim == 2 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.ndim != 1:
            raise ValueError(f"critic must return shape (Batch,) or (Batch, 1); got {tuple(values.shape)}.")
        return values

    @staticmethod
    def _sum_action_dimension(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=-1) if tensor.ndim > 1 else tensor

    def _prepare_joint_state_action(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = states.to(self.device)
        actions = actions.to(self.device)
        states = self._final_state_frame(states)
        actions = self._ensure_joint_actions(actions)
        if states.shape[:2] != actions.shape[:2]:
            raise ValueError(
                "states and actions must share Batch and N dimensions; "
                f"got {tuple(states.shape[:2])} and {tuple(actions.shape[:2])}."
            )
        return torch.cat((states, actions), dim=-1)

    @staticmethod
    def _final_state_frame(states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 4 and states.shape[-2:] == (4, STATE_FEATURE_COUNT):
            return states[:, :, -1, :]
        if states.ndim == 3 and states.shape[-2:] == (4, STATE_FEATURE_COUNT):
            return states[:, -1, :].unsqueeze(1)
        if states.ndim == 3 and states.shape[-1] == STATE_FEATURE_COUNT:
            return states
        if states.ndim == 2 and states.shape[-1] == STATE_FEATURE_COUNT:
            return states.unsqueeze(1)
        raise ValueError("states must have shape (B,N,41), (B,N,4,41), (B,4,41), or (B,41).")

    @staticmethod
    def _ensure_joint_actions(actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3 and actions.shape[-1] == ACTION_FEATURE_COUNT:
            return actions
        if actions.ndim == 2 and actions.shape[-1] == ACTION_FEATURE_COUNT:
            return actions.unsqueeze(1)
        raise ValueError("actions must have shape (B,N,2) or (B,2).")

    @staticmethod
    def _unpack_state_action_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, Mapping):
            state = batch.get("states", batch.get("state"))
            action = batch.get("actions", batch.get("action"))
            if state is None or action is None:
                raise KeyError("Batch dictionary must contain states/state and actions/action keys.")
            return state, action
        if isinstance(batch, tuple | list) and len(batch) >= 2:
            return batch[0], batch[1]
        raise TypeError("Batch must be a mapping or a tuple/list of (states, actions).")
