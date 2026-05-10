from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from geometry_msgs.msg import Twist
from torch.distributions import Distribution

STATE_WINDOW_SIZE = 4
STATE_FEATURE_COUNT = 41
ACTION_DIM = 2
TB3_MAX_LINEAR_MPS = 0.22
TB3_MAX_ANGULAR_RADPS = 2.84
DEFAULT_ANGULAR_SMOOTHING_ALPHA = 0.8


@torch.no_grad()
def get_deterministic_action(
    model: torch.nn.Module, state_tensor: torch.Tensor, device: Any = None
) -> torch.Tensor:
    if not isinstance(state_tensor, torch.Tensor):
        raise TypeError("state_tensor must be a torch.Tensor.")
    if tuple(state_tensor.shape) != (1, STATE_WINDOW_SIZE, STATE_FEATURE_COUNT):
        raise ValueError(
            f"state_tensor must have shape (1, {STATE_WINDOW_SIZE}, {STATE_FEATURE_COUNT}); "
            f"got {tuple(state_tensor.shape)}."
        )

    target_device = torch.device(device) if device is not None else state_tensor.device
    model.eval()
    output = model(state_tensor.to(target_device, dtype=torch.float32))
    action = _extract_mean_action(output)
    if action.shape == (1, ACTION_DIM):
        action = action.squeeze(0)
    if action.shape != (ACTION_DIM,):
        raise ValueError(
            f"deterministic action must flatten to shape ({ACTION_DIM},); got {tuple(action.shape)}."
        )
    return action.detach().to(dtype=torch.float32)


def scale_action_to_velocity(action: Any) -> tuple[float, float]:
    raw = torch.as_tensor(action, dtype=torch.float32).flatten()
    if raw.shape != (ACTION_DIM,):
        raise ValueError(f"action must contain exactly {ACTION_DIM} values; got shape {tuple(raw.shape)}.")
    raw = torch.clamp(raw, -1.0, 1.0)
    linear = (raw[0] + 1.0) * 0.5 * TB3_MAX_LINEAR_MPS
    angular = raw[1] * TB3_MAX_ANGULAR_RADPS
    return float(linear), float(angular)


def is_placeholder_state(state_tensor: torch.Tensor) -> bool:
    if not isinstance(state_tensor, torch.Tensor):
        return True
    if tuple(state_tensor.shape) != (1, STATE_WINDOW_SIZE, STATE_FEATURE_COUNT):
        return True
    return (not torch.isfinite(state_tensor).all()) or bool(torch.all(state_tensor == 0.0).item())


def make_twist(linear_velocity: float, angular_velocity: float) -> Twist:
    msg = Twist()
    msg.linear.x = float(linear_velocity)
    msg.angular.z = float(angular_velocity)
    return msg


@dataclass
class DeterministicDecisionMaker:
    model: torch.nn.Module
    device: Any = None
    angular_smoothing_alpha: float = DEFAULT_ANGULAR_SMOOTHING_ALPHA

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.angular_smoothing_alpha) < 1.0:
            raise ValueError("angular_smoothing_alpha must be in [0.0, 1.0).")
        self.device = torch.device(self.device) if self.device is not None else None
        if self.device is not None:
            self.model.to(self.device)
        self.model.eval()
        self._previous_angular_velocity = 0.0
        self._has_previous_angular_velocity = False

    def build_command(self, state_tensor: torch.Tensor) -> Twist:
        if is_placeholder_state(state_tensor):
            self.reset_smoothing()
            return make_twist(0.0, 0.0)

        action = get_deterministic_action(self.model, state_tensor, device=self.device)
        linear, angular = scale_action_to_velocity(action)
        angular = self._smooth_angular_velocity(angular)
        return make_twist(linear, angular)

    def reset_smoothing(self) -> None:
        self._previous_angular_velocity = 0.0
        self._has_previous_angular_velocity = False

    def _smooth_angular_velocity(self, angular_velocity: float) -> float:
        if not self._has_previous_angular_velocity:
            self._previous_angular_velocity = float(angular_velocity)
            self._has_previous_angular_velocity = True
            return float(angular_velocity)

        alpha = float(self.angular_smoothing_alpha)
        smoothed = alpha * self._previous_angular_velocity + (1.0 - alpha) * float(angular_velocity)
        self._previous_angular_velocity = smoothed
        return smoothed


def _extract_mean_action(model_output: Any) -> torch.Tensor:
    if (
        isinstance(model_output, Distribution)
        or hasattr(model_output, "mean")
        and isinstance(model_output.mean, torch.Tensor)
    ):
        action = model_output.mean
    else:
        action = model_output
    if not isinstance(action, torch.Tensor):
        action = torch.as_tensor(action, dtype=torch.float32)
    return action
