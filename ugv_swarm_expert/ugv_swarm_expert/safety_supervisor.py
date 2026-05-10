from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from geometry_msgs.msg import Twist

COLLISION_THRESHOLD_M = 0.15
SLOWDOWN_THRESHOLD_M = 0.50
SLOWDOWN_SCALE = 0.50
FORMATION_LOST_DISTANCE_M = 2.0
RECOVERY_LIDAR_FOCUS_RANGE_M = 1.5
LIDAR_MIN_RANGE_M = 0.12
LIDAR_MAX_RANGE_M = 3.5
FRONT_ARC_DEG = 30.0


@dataclass(frozen=True)
class FormationMetrics:
    distance_error: float
    heading_error: float
    mean_formation_error: float
    sample_count: int
    formation_lost: bool


class FormationMonitor:
    def __init__(
        self,
        target_offset: tuple[float, float],
        formation_lost_distance_m: float = FORMATION_LOST_DISTANCE_M,
    ):
        if len(target_offset) != 2:
            raise ValueError("target_offset must contain exactly two values.")
        if formation_lost_distance_m <= 0.0:
            raise ValueError("formation_lost_distance_m must be positive.")
        self.target_offset = np.asarray(target_offset, dtype=np.float32)
        self.formation_lost_distance_m = float(formation_lost_distance_m)
        self._error_sum = 0.0
        self._sample_count = 0
        self._last_metrics = FormationMetrics(0.0, 0.0, 0.0, 0, False)

    def update(
        self,
        follower_x: float,
        follower_y: float,
        follower_theta: float,
        leader_x: float,
        leader_y: float,
        leader_theta: float,
    ) -> FormationMetrics:
        error_xy = self.compute_error_vector(
            follower_x=follower_x,
            follower_y=follower_y,
            leader_x=leader_x,
            leader_y=leader_y,
            leader_theta=leader_theta,
        )
        distance_error = float(np.linalg.norm(error_xy))
        heading_error = self.wrap_angle(float(follower_theta) - float(leader_theta))
        self._sample_count += 1
        self._error_sum += distance_error
        mean_error = self._error_sum / self._sample_count
        metrics = FormationMetrics(
            distance_error=distance_error,
            heading_error=heading_error,
            mean_formation_error=mean_error,
            sample_count=self._sample_count,
            formation_lost=distance_error > self.formation_lost_distance_m,
        )
        self._last_metrics = metrics
        return metrics

    def compute_error_vector(
        self,
        follower_x: float,
        follower_y: float,
        leader_x: float,
        leader_y: float,
        leader_theta: float,
    ) -> np.ndarray:
        dx_global = float(follower_x) - float(leader_x)
        dy_global = float(follower_y) - float(leader_y)
        theta = self.wrap_angle(float(leader_theta))
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        local_x = cos_theta * dx_global + sin_theta * dy_global
        local_y = -sin_theta * dx_global + cos_theta * dy_global
        return np.asarray([local_x, local_y], dtype=np.float32) - self.target_offset

    def reset(self) -> None:
        self._error_sum = 0.0
        self._sample_count = 0
        self._last_metrics = FormationMetrics(0.0, 0.0, 0.0, 0, False)

    @property
    def mean_formation_error(self) -> float:
        return self._error_sum / self._sample_count if self._sample_count else 0.0

    @property
    def last_metrics(self) -> FormationMetrics:
        return self._last_metrics

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


class SafetySupervisor:
    def __init__(
        self,
        collision_threshold_m: float = COLLISION_THRESHOLD_M,
        slowdown_threshold_m: float = SLOWDOWN_THRESHOLD_M,
        slowdown_scale: float = SLOWDOWN_SCALE,
        front_arc_deg: float = FRONT_ARC_DEG,
        logger: Any = None,
    ):
        if collision_threshold_m <= 0.0:
            raise ValueError("collision_threshold_m must be positive.")
        if slowdown_threshold_m < collision_threshold_m:
            raise ValueError("slowdown_threshold_m must be >= collision_threshold_m.")
        if not 0.0 <= slowdown_scale <= 1.0:
            raise ValueError("slowdown_scale must be in [0, 1].")
        if not 0.0 < front_arc_deg <= 180.0:
            raise ValueError("front_arc_deg must be in (0, 180].")
        self.collision_threshold_m = float(collision_threshold_m)
        self.slowdown_threshold_m = float(slowdown_threshold_m)
        self.slowdown_scale = float(slowdown_scale)
        self.front_arc_deg = float(front_arc_deg)
        self.logger = logger
        self.last_override_reason = "none"

    def apply_safety_policy(self, actor_v: float, actor_w: float, raw_scan: Any) -> Twist:
        ranges = sanitize_lidar(raw_scan)
        front_min = min_front_distance(ranges, front_arc_deg=self.front_arc_deg)
        safe_v = float(actor_v)
        safe_w = float(actor_w)
        self.last_override_reason = "none"

        if front_min < self.collision_threshold_m:
            safe_v = 0.0
            self.last_override_reason = "emergency_stop"
            self._warn("Safety Override: Emergency Stop Active")
        elif front_min < self.slowdown_threshold_m:
            safe_v *= self.slowdown_scale
            self.last_override_reason = "slowdown"
            self._warn("Safety Override: Slowdown Active")

        return make_twist(safe_v, safe_w)

    def mask_lidar_for_recovery(
        self,
        raw_scan: Any,
        formation_lost: bool,
        focus_range_m: float = RECOVERY_LIDAR_FOCUS_RANGE_M,
    ) -> np.ndarray:
        ranges = sanitize_lidar(raw_scan)
        if not formation_lost:
            return ranges
        masked = ranges.copy()
        masked[masked > focus_range_m] = LIDAR_MAX_RANGE_M
        return masked

    def _warn(self, message: str) -> None:
        if self.logger is None:
            return
        if hasattr(self.logger, "warning"):
            self.logger.warning(message)
        elif hasattr(self.logger, "warn"):
            self.logger.warn(message)


_DEFAULT_SUPERVISOR = SafetySupervisor()


def apply_safety_policy(actor_v: float, actor_w: float, raw_scan: Any, logger: Any = None) -> Twist:
    supervisor = SafetySupervisor(logger=logger) if logger is not None else _DEFAULT_SUPERVISOR
    return supervisor.apply_safety_policy(actor_v, actor_w, raw_scan)


def sanitize_lidar(raw_scan: Any) -> np.ndarray:
    ranges = np.asarray(getattr(raw_scan, "ranges", raw_scan), dtype=np.float32)
    if ranges.size == 0:
        return np.full(360, LIDAR_MAX_RANGE_M, dtype=np.float32)
    ranges = np.nan_to_num(ranges, nan=LIDAR_MAX_RANGE_M, posinf=LIDAR_MAX_RANGE_M, neginf=LIDAR_MAX_RANGE_M)
    return np.clip(ranges, LIDAR_MIN_RANGE_M, LIDAR_MAX_RANGE_M).astype(np.float32, copy=False)


def front_arc_ranges(ranges: np.ndarray, front_arc_deg: float = FRONT_ARC_DEG) -> np.ndarray:
    if ranges.ndim != 1:
        raise ValueError("LiDAR ranges must be one-dimensional.")
    if ranges.size == 0:
        return np.full(1, LIDAR_MAX_RANGE_M, dtype=np.float32)
    rays_per_degree = ranges.size / 360.0
    half_width = max(1, int(round(float(front_arc_deg) * rays_per_degree)))
    return np.concatenate((ranges[-half_width:], ranges[: half_width + 1]))


def min_front_distance(raw_scan: Any, front_arc_deg: float = FRONT_ARC_DEG) -> float:
    ranges = sanitize_lidar(raw_scan)
    return float(np.min(front_arc_ranges(ranges, front_arc_deg=front_arc_deg)))


def make_twist(linear_velocity: float, angular_velocity: float) -> Twist:
    msg = Twist()
    msg.linear.x = float(linear_velocity)
    msg.angular.z = float(angular_velocity)
    return msg
