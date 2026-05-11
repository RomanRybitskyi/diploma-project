from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ugv_swarm_expert.constants import CONTROL_PERIOD_SEC, RECOVERY_LIDAR_FOCUS_RANGE_M

LOGGER = logging.getLogger(__name__)

FORMATION_LOST_THRESHOLD_M: float = 0.5
FORMATION_LOST_DURATION_SEC: float = 3.0
RECOVERY_THRESHOLD_M: float = 0.1
OBSTACLE_DETECT_RANGE_M: float = RECOVERY_LIDAR_FOCUS_RANGE_M


@dataclass
class StepData:
    timestamp: float
    formation_errors: list[float]
    angular_velocities: list[float]
    collision: bool
    obstacle_detected: bool = False


@dataclass
class EpisodeResult:
    mean_formation_error: float
    smoothness_factor: float
    recovery_time: float | None
    success: bool
    steps: int
    total_collisions: int
    formation_lost_events: int


class MetricsTracker:
    def __init__(
        self,
        num_followers: int,
        control_period_sec: float = CONTROL_PERIOD_SEC,
        formation_lost_threshold_m: float = FORMATION_LOST_THRESHOLD_M,
        formation_lost_duration_sec: float = FORMATION_LOST_DURATION_SEC,
        recovery_threshold_m: float = RECOVERY_THRESHOLD_M,
    ) -> None:
        if num_followers <= 0:
            raise ValueError("num_followers must be a positive integer.")
        self.num_followers = int(num_followers)
        self.dt = float(control_period_sec)
        self._lost_thr = float(formation_lost_threshold_m)
        self._lost_dur = float(formation_lost_duration_sec)
        self._rec_thr = float(recovery_threshold_m)

        self._step_count: int = 0
        self._ef_sum: float = 0.0
        self._smoothness: float = 0.0
        self._prev_omega: list[float] | None = None
        self._collision_count: int = 0

        self._current_streak: int = 0
        self._max_streak: int = 0
        self._lost_events: int = 0
        self._in_lost: bool = False

        self._obstacle_step: int | None = None
        self._recovery_step: int | None = None

    def update(self, step: StepData) -> None:
        idx = self._step_count
        self._step_count += 1

        ef = float(np.mean(step.formation_errors)) if step.formation_errors else 0.0
        self._ef_sum += ef

        if step.collision:
            self._collision_count += 1

        if self._prev_omega is not None:
            for prev_w, curr_w in zip(self._prev_omega, step.angular_velocities, strict=False):
                self._smoothness += (curr_w - prev_w) ** 2
        self._prev_omega = list(step.angular_velocities)

        if ef > self._lost_thr:
            self._current_streak += 1
            self._max_streak = max(self._max_streak, self._current_streak)
            if not self._in_lost:
                self._in_lost = True
                self._lost_events += 1
        else:
            self._current_streak = 0
            self._in_lost = False

        if step.obstacle_detected and self._obstacle_step is None:
            self._obstacle_step = idx
        if (
            self._obstacle_step is not None
            and self._recovery_step is None
            and ef < self._rec_thr
            and idx > self._obstacle_step
        ):
            self._recovery_step = idx

    @property
    def mean_formation_error(self) -> float:
        return self._ef_sum / self._step_count if self._step_count else 0.0

    @property
    def smoothness_factor(self) -> float:
        return self._smoothness / max(self.num_followers, 1)

    @property
    def recovery_time(self) -> float | None:
        if self._obstacle_step is None or self._recovery_step is None:
            return None
        return (self._recovery_step - self._obstacle_step) * self.dt

    @property
    def success(self) -> bool:
        if self._collision_count > 0:
            return False
        max_lost_duration = self._max_streak * self.dt
        return max_lost_duration < self._lost_dur

    def finalize(self) -> EpisodeResult:
        return EpisodeResult(
            mean_formation_error=self.mean_formation_error,
            smoothness_factor=self.smoothness_factor,
            recovery_time=self.recovery_time,
            success=self.success,
            steps=self._step_count,
            total_collisions=self._collision_count,
            formation_lost_events=self._lost_events,
        )


class EvaluationSuite:
    def __init__(self, results: list[EpisodeResult]) -> None:
        self.results = list(results)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    @property
    def mean_formation_error(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.mean_formation_error for r in self.results]))

    @property
    def mean_smoothness(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.smoothness_factor for r in self.results]))

    @property
    def mean_recovery_time(self) -> float | None:
        times = [r.recovery_time for r in self.results if r.recovery_time is not None]
        return float(np.mean(times)) if times else None

    def report(self) -> dict[str, Any]:
        return {
            "num_episodes": len(self.results),
            "success_rate": round(self.success_rate, 4),
            "mean_formation_error_m": round(self.mean_formation_error, 4),
            "smoothness_factor": round(self.mean_smoothness, 4),
            "mean_recovery_time_s": (
                round(self.mean_recovery_time, 3) if self.mean_recovery_time is not None else None
            ),
            "successful_episodes": sum(1 for r in self.results if r.success),
            "total_collisions": sum(r.total_collisions for r in self.results),
            "formation_lost_events": sum(r.formation_lost_events for r in self.results),
            "episodes": [
                {
                    "mean_formation_error_m": round(r.mean_formation_error, 4),
                    "smoothness_factor": round(r.smoothness_factor, 4),
                    "recovery_time_s": (round(r.recovery_time, 3) if r.recovery_time is not None else None),
                    "success": r.success,
                    "steps": r.steps,
                    "collisions": r.total_collisions,
                }
                for r in self.results
            ],
        }

    def print_summary(self) -> None:
        n = len(self.results)
        ok = sum(1 for r in self.results if r.success)
        print("\n" + "=" * 58)
        print(" MA-GAIL Evaluation Results")
        print("=" * 58)
        print(f"  Episodes        : {n}  (successful: {ok})")
        print(f"  SR  Success Rate: {self.success_rate * 100:.1f} %")
        print(f"  E_f Form. Error : {self.mean_formation_error * 100:.2f} cm")
        print(f"  S_ω Smoothness  : {self.mean_smoothness:.4f}")
        t_rec = self.mean_recovery_time
        print(f"  T_rec Recovery  : {'N/A' if t_rec is None else f'{t_rec:.2f} s'}")
        print("=" * 58 + "\n")


def _step_data_from_info(
    info: dict[str, Any],
    num_followers: int,
    timestamp: float,
    obstacle_range_m: float = OBSTACLE_DETECT_RANGE_M,
) -> StepData:
    formation_errors = list(info.get("formation_errors", np.zeros(num_followers + 1, dtype=np.float32))[1:])
    angular_velocities = list(
        info.get("angular_velocities", np.zeros(num_followers + 1, dtype=np.float32))[1:]
    )
    collision = bool(np.any(info.get("collision", False)))
    min_lidar = float(np.min(info.get("min_lidar", np.array([obstacle_range_m + 1.0]))))
    obstacle_detected = min_lidar < obstacle_range_m
    return StepData(
        timestamp=timestamp,
        formation_errors=formation_errors,
        angular_velocities=angular_velocities,
        collision=collision,
        obstacle_detected=obstacle_detected,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MA-GAIL Actor in Gazebo/ROS 2 and report SR, E_f, S_ω, T_rec."
    )
    parser.add_argument("--actor", required=True, help="Path to actor .pth checkpoint.")
    parser.add_argument("--num-agents", type=int, default=3, help="Total agents (leader + followers).")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Maximum steps per episode (50 s at 10 Hz)."
    )
    parser.add_argument("--output", default=None, help="Optional path for JSON results file.")
    parser.add_argument("--device", default=None, help="Torch device override (e.g. 'cpu' or 'cuda:0').")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv=None) -> None:
    import rclpy
    import torch

    from ugv_swarm_expert.constants import STATE_FEATURE_COUNT, STATE_WINDOW_SIZE
    from ugv_swarm_expert.env.ugv_swarm_env import UGVSwarmEnv
    from ugv_swarm_expert.models.actor_network import ActorNetwork

    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Evaluation device: %s", device)

    state_shape = (STATE_WINDOW_SIZE, STATE_FEATURE_COUNT)
    num_agents = args.num_agents
    num_followers = num_agents - 1

    actor = ActorNetwork().to(device)
    checkpoint_path = Path(args.actor).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Actor checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and not all(isinstance(k, str) and "." in k for k in state_dict):
        for key in ("actor_state_dict", "model_state_dict", "state_dict", "actor"):
            if key in state_dict:
                state_dict = state_dict[key]
                break
    actor.load_state_dict(state_dict)
    actor.eval()
    LOGGER.info("Loaded actor from %s", checkpoint_path)

    rclpy.init()
    results: list[EpisodeResult] = []

    try:
        with UGVSwarmEnv(num_agents=num_agents, device=device) as env:
            for ep in range(1, args.episodes + 1):
                tracker = MetricsTracker(num_followers=num_followers)
                states = env.reset()
                t0 = time.monotonic()

                for step_idx in range(args.max_steps):
                    with torch.no_grad():
                        flat_states = states.reshape(num_agents, *state_shape).to(device)
                        dist = actor(flat_states)
                        actions = dist.mean.clamp(-1.0, 1.0)

                    states, dones, info = env.step(actions.cpu().numpy())
                    timestamp = t0 + step_idx * CONTROL_PERIOD_SEC

                    step_data = _step_data_from_info(info, num_followers, timestamp)
                    tracker.update(step_data)

                    if np.any(dones):
                        break

                result = tracker.finalize()
                results.append(result)
                LOGGER.info(
                    "Episode %d/%d | steps=%d | E_f=%.3f m | S_ω=%.4f | " "T_rec=%s | success=%s",
                    ep,
                    args.episodes,
                    result.steps,
                    result.mean_formation_error,
                    result.smoothness_factor,
                    f"{result.recovery_time:.2f}s" if result.recovery_time is not None else "N/A",
                    result.success,
                )
    finally:
        rclpy.shutdown()

    suite = EvaluationSuite(results)
    suite.print_summary()

    if args.output is not None:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(suite.report(), indent=2))
        LOGGER.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
