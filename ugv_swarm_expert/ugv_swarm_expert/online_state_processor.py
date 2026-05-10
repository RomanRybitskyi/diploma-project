from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
import torch


class OnlineStateProcessor:
    WINDOW_SIZE = 4
    STATE_FEATURE_COUNT = 41
    LIDAR_SECTOR_COUNT = 36

    LINEAR_V_MIN = 0.0
    LINEAR_V_MAX = 0.22
    ANGULAR_W_MIN = -2.84
    ANGULAR_W_MAX = 2.84
    THETA_MIN = -math.pi
    THETA_MAX = math.pi
    FORMATION_ERROR_MIN = -2.0
    FORMATION_ERROR_MAX = 2.0
    LIDAR_MIN = 0.12
    LIDAR_MAX = 3.5

    def __init__(self, device: Any = None, dtype: torch.dtype = torch.float32) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self._buffer: deque[np.ndarray] = deque(maxlen=self.WINDOW_SIZE)

    def process_step(
        self,
        follower_v: float,
        follower_w: float,
        follower_theta: float,
        follower_x: float,
        follower_y: float,
        leader_x: float,
        leader_y: float,
        leader_theta: float,
        target_offset_x: float,
        target_offset_y: float,
        lidar_ranges: Any,
    ) -> torch.Tensor:
        kinematics = np.asarray(
            [
                self._normalize_scalar(follower_v, self.LINEAR_V_MIN, self.LINEAR_V_MAX),
                self._normalize_scalar(follower_w, self.ANGULAR_W_MIN, self.ANGULAR_W_MAX),
                self._normalize_scalar(self._wrap_angle(follower_theta), self.THETA_MIN, self.THETA_MAX),
            ],
            dtype=np.float32,
        )
        formation_error = self._compute_normalized_formation_error(
            follower_x=follower_x,
            follower_y=follower_y,
            leader_x=leader_x,
            leader_y=leader_y,
            leader_theta=leader_theta,
            target_offset_x=target_offset_x,
            target_offset_y=target_offset_y,
        )
        lidar = self.process_lidar(lidar_ranges)

        state = np.concatenate((kinematics, formation_error, lidar), axis=0).astype(np.float32, copy=False)
        if state.shape != (self.STATE_FEATURE_COUNT,):
            raise RuntimeError(
                f"Internal state must have shape ({self.STATE_FEATURE_COUNT},); got {state.shape}."
            )

        self._append_state(state)
        return self.as_tensor()

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) == self.WINDOW_SIZE

    def as_tensor(self) -> torch.Tensor:
        if not self._buffer:
            raise RuntimeError("State buffer is empty. Call process_step() before as_tensor().")
        stacked = np.stack(tuple(self._buffer), axis=0).astype(np.float32, copy=False)
        return torch.as_tensor(stacked, dtype=self.dtype, device=self.device).unsqueeze(0)

    @classmethod
    def process_lidar(cls, lidar_ranges: Any) -> np.ndarray:
        ranges = np.asarray(lidar_ranges, dtype=np.float32)
        if ranges.shape != (360,):
            raise ValueError(f"lidar_ranges must contain exactly 360 rays; got shape {ranges.shape}.")

        filtered = np.nan_to_num(ranges, nan=cls.LIDAR_MAX, posinf=cls.LIDAR_MAX, neginf=cls.LIDAR_MAX)
        filtered = np.clip(filtered, cls.LIDAR_MIN, cls.LIDAR_MAX)
        sectors = filtered.reshape(cls.LIDAR_SECTOR_COUNT, -1).min(axis=1)
        return cls._normalize_array(sectors, cls.LIDAR_MIN, cls.LIDAR_MAX)

    def _append_state(self, state: np.ndarray) -> None:
        if not self._buffer:
            for _ in range(self.WINDOW_SIZE):
                self._buffer.append(state.copy())
        else:
            self._buffer.append(state.copy())

    @classmethod
    def _compute_normalized_formation_error(
        cls,
        follower_x: float,
        follower_y: float,
        leader_x: float,
        leader_y: float,
        leader_theta: float,
        target_offset_x: float,
        target_offset_y: float,
    ) -> np.ndarray:
        dx_global = float(follower_x) - float(leader_x)
        dy_global = float(follower_y) - float(leader_y)
        theta = cls._wrap_angle(float(leader_theta))
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        local_x = cos_theta * dx_global + sin_theta * dy_global
        local_y = -sin_theta * dx_global + cos_theta * dy_global
        error = np.asarray(
            [local_x - float(target_offset_x), local_y - float(target_offset_y)],
            dtype=np.float32,
        )
        return cls._normalize_array(error, cls.FORMATION_ERROR_MIN, cls.FORMATION_ERROR_MAX)

    @classmethod
    def _normalize_scalar(cls, value: float, lower: float, upper: float) -> float:
        return float(cls._normalize_array(np.asarray([value], dtype=np.float32), lower, upper)[0])

    @staticmethod
    def _normalize_array(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
        if upper <= lower:
            raise ValueError("upper normalization bound must be greater than lower bound.")
        finite = np.nan_to_num(values.astype(np.float32, copy=False), nan=lower, posinf=upper, neginf=lower)
        clipped = np.clip(finite, lower, upper)
        normalized = (clipped - lower) / (upper - lower)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))
