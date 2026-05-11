from __future__ import annotations

import math

from torch import nn

DEFAULT_ORTHOGONAL_GAIN = float(math.sqrt(2.0))
PPO_ACTION_HEAD_GAIN = 0.01


def orthogonal_init(module: nn.Module, gain: float = DEFAULT_ORTHOGONAL_GAIN) -> nn.Module:
    for layer in module.modules():
        if isinstance(layer, nn.Linear | nn.Conv1d):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    return module


def orthogonal_init_layer(layer: nn.Module, gain: float = DEFAULT_ORTHOGONAL_GAIN) -> nn.Module:
    if not isinstance(layer, nn.Linear | nn.Conv1d):
        raise TypeError("orthogonal_init_layer supports only nn.Linear and nn.Conv1d layers.")
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)
    return layer


def init_ppo_action_head(action_head: nn.Module, gain: float = PPO_ACTION_HEAD_GAIN) -> nn.Module:
    layer = _resolve_action_linear(action_head)
    orthogonal_init_layer(layer, gain=gain)
    return action_head


def _resolve_action_linear(action_head: nn.Module) -> nn.Linear:
    if isinstance(action_head, nn.Linear):
        return action_head
    if (
        isinstance(action_head, nn.Sequential)
        and len(action_head) > 0
        and isinstance(action_head[0], nn.Linear)
    ):
        return action_head[0]
    raise TypeError("action_head must be an nn.Linear or nn.Sequential beginning with nn.Linear.")
