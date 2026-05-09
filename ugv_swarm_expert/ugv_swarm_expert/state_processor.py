from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

STATE_WINDOW_SIZE = 4
STATE_FEATURE_COUNT = 41
EGO_FEATURE_COUNT = 3
FORMATION_FEATURE_COUNT = 2
LIDAR_SECTOR_COUNT = 36

TB3_MIN_LINEAR_MPS = 0.0
TB3_MAX_LINEAR_MPS = 0.22
TB3_MIN_ANGULAR_RADPS = -2.84
TB3_MAX_ANGULAR_RADPS = 2.84
MIN_HEADING_RAD = -math.pi
MAX_HEADING_RAD = math.pi
FORMATION_ERROR_MIN_M = -2.0
FORMATION_ERROR_MAX_M = 2.0
LIDAR_MIN_RANGE_M = 0.12
LIDAR_MAX_RANGE_M = 3.5


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Velocity2D:
    linear: float
    angular: float


DeviceLike = Any


class StateProcessor:
    def __init__(
        self,
        target_offset: tuple[float, float],
        device: DeviceLike = None,
        window_size: int = STATE_WINDOW_SIZE,
        formation_error_limit: float = FORMATION_ERROR_MAX_M,
        dtype: torch.dtype = torch.float32,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if formation_error_limit <= 0.0:
            raise ValueError("formation_error_limit must be positive.")
        if len(target_offset) != 2:
            raise ValueError("target_offset must contain exactly two values: (x, y).")

        self.target_offset = np.asarray(target_offset, dtype=np.float32)
        self.window_size = int(window_size)
        self.formation_error_limit = float(formation_error_limit)
        self.device = torch.device(device) if device is not None else self._default_device()
        self.dtype = dtype
        self._buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

    def process(self, follower_odom: Any, leader_odom: Any, scan: Any) -> torch.Tensor:
        follower_pose = self.pose_from_odom(follower_odom)
        leader_pose = self.pose_from_odom(leader_odom)
        follower_velocity = self.velocity_from_odom(follower_odom)
        lidar_sectors = self.process_lidar(scan)

        normalized_state = self.build_state(
            follower_pose=follower_pose,
            leader_pose=leader_pose,
            follower_velocity=follower_velocity,
            lidar_sectors=lidar_sectors,
        )
        self._append_state(normalized_state)
        return self.as_tensor()

    def build_state(
        self,
        follower_pose: Pose2D,
        leader_pose: Pose2D,
        follower_velocity: Velocity2D,
        lidar_sectors: Sequence[float],
    ) -> np.ndarray:
        if len(lidar_sectors) != LIDAR_SECTOR_COUNT:
            raise ValueError(f"lidar_sectors must contain {LIDAR_SECTOR_COUNT} values.")

        formation_error = self.compute_formation_error(follower_pose, leader_pose)
        raw_state = np.concatenate(
            [
                np.asarray(
                    [follower_velocity.linear, follower_velocity.angular, follower_pose.yaw],
                    dtype=np.float32,
                ),
                formation_error.astype(np.float32),
                np.asarray(lidar_sectors, dtype=np.float32),
            ]
        )
        return self.normalize(raw_state)

    def compute_formation_error(self, follower_pose: Pose2D, leader_pose: Pose2D) -> np.ndarray:
        dx_global = follower_pose.x - leader_pose.x
        dy_global = follower_pose.y - leader_pose.y
        cos_yaw = math.cos(leader_pose.yaw)
        sin_yaw = math.sin(leader_pose.yaw)

        local_x = cos_yaw * dx_global + sin_yaw * dy_global
        local_y = -sin_yaw * dx_global + cos_yaw * dy_global
        return np.asarray([local_x, local_y], dtype=np.float32) - self.target_offset

    @staticmethod
    def process_lidar(scan: Any) -> np.ndarray:
        ranges = np.asarray(getattr(scan, "ranges", scan), dtype=np.float32)
        if ranges.size == 0:
            return np.full(LIDAR_SECTOR_COUNT, LIDAR_MAX_RANGE_M, dtype=np.float32)

        filtered = np.nan_to_num(
            ranges,
            nan=LIDAR_MAX_RANGE_M,
            posinf=LIDAR_MAX_RANGE_M,
            neginf=LIDAR_MIN_RANGE_M,
        )
        filtered = np.clip(filtered, LIDAR_MIN_RANGE_M, LIDAR_MAX_RANGE_M)

        if filtered.size == LIDAR_SECTOR_COUNT:
            return filtered.astype(np.float32, copy=False)

        if filtered.size % LIDAR_SECTOR_COUNT == 0:
            return filtered.reshape(LIDAR_SECTOR_COUNT, -1).min(axis=1).astype(np.float32)

        sectors = np.empty(LIDAR_SECTOR_COUNT, dtype=np.float32)
        split_indices = np.linspace(0, filtered.size, LIDAR_SECTOR_COUNT + 1, dtype=np.int64)
        for index in range(LIDAR_SECTOR_COUNT):
            start = split_indices[index]
            end = split_indices[index + 1]
            if end <= start:
                end = min(start + 1, filtered.size)
            sectors[index] = filtered[start:end].min()
        return sectors

    def normalize(self, raw_state: Sequence[float]) -> np.ndarray:
        state = np.asarray(raw_state, dtype=np.float32)
        if state.shape != (STATE_FEATURE_COUNT,):
            raise ValueError(f"raw_state must have shape ({STATE_FEATURE_COUNT},).")

        lower_bounds, upper_bounds = self._normalization_bounds()
        clipped = np.clip(state, lower_bounds, upper_bounds)
        normalized = (clipped - lower_bounds) / (upper_bounds - lower_bounds)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def as_tensor(self) -> torch.Tensor:
        if not self._buffer:
            raise RuntimeError("State buffer is empty. Call process() before as_tensor().")
        stacked = np.stack(tuple(self._buffer), axis=0).astype(np.float32, copy=False)
        return torch.as_tensor(stacked, dtype=self.dtype, device=self.device)

    def reset(self) -> None:
        self._buffer.clear()

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) == self.window_size

    @staticmethod
    def pose_from_odom(odom: Any) -> Pose2D:
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        yaw = StateProcessor.yaw_from_quaternion(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )
        return Pose2D(x=float(position.x), y=float(position.y), yaw=yaw)

    @staticmethod
    def velocity_from_odom(odom: Any) -> Velocity2D:
        twist = odom.twist.twist
        return Velocity2D(linear=float(twist.linear.x), angular=float(twist.angular.z))

    @staticmethod
    def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _append_state(self, normalized_state: np.ndarray) -> None:
        if normalized_state.shape != (STATE_FEATURE_COUNT,):
            raise ValueError(f"normalized_state must have shape ({STATE_FEATURE_COUNT},).")
        if not self._buffer:
            for _ in range(self.window_size):
                self._buffer.append(normalized_state.copy())
        else:
            self._buffer.append(normalized_state.copy())

    def _normalization_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower_bounds = np.asarray(
            [
                TB3_MIN_LINEAR_MPS,
                TB3_MIN_ANGULAR_RADPS,
                MIN_HEADING_RAD,
                -self.formation_error_limit,
                -self.formation_error_limit,
                *([LIDAR_MIN_RANGE_M] * LIDAR_SECTOR_COUNT),
            ],
            dtype=np.float32,
        )
        upper_bounds = np.asarray(
            [
                TB3_MAX_LINEAR_MPS,
                TB3_MAX_ANGULAR_RADPS,
                MAX_HEADING_RAD,
                self.formation_error_limit,
                self.formation_error_limit,
                *([LIDAR_MAX_RANGE_M] * LIDAR_SECTOR_COUNT),
            ],
            dtype=np.float32,
        )
        return lower_bounds, upper_bounds

    @staticmethod
    def _default_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
