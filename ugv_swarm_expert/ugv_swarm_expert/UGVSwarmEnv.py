from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from ugv_swarm_expert.state_processor import (
    LIDAR_MIN_RANGE_M,
    TB3_MAX_ANGULAR_RADPS,
    TB3_MAX_LINEAR_MPS,
    StateProcessor,
)

CONTROL_PERIOD_SEC = 0.1
DEFAULT_COLLISION_THRESHOLD_M = 0.15
DEFAULT_WORKSPACE_BOUNDARY_M = 10.0
DEFAULT_RESET_SERVICE = "/reset_simulation"


class UGVSwarmEnv:
    def __init__(
        self,
        num_agents: int = 3,
        agent_names=None,
        leader_name=None,
        target_offsets=None,
        device=None,
        node_name: str = "ugv_swarm_env",
        odom_topic_template: str = "/{agent}/odom",
        scan_topic_template: str = "/{agent}/scan",
        cmd_vel_topic_template: str = "/{agent}/cmd_vel",
        reset_service_name: str = DEFAULT_RESET_SERVICE,
        collision_threshold_m: float = DEFAULT_COLLISION_THRESHOLD_M,
        workspace_boundary_m: float = DEFAULT_WORKSPACE_BOUNDARY_M,
        control_period_sec: float = CONTROL_PERIOD_SEC,
        observation_timeout_sec: float = 5.0,
        reset_timeout_sec: float = 5.0,
        reset_settle_sec: float = 0.5,
        use_background_spin: bool = True,
    ):
        if num_agents <= 0:
            raise ValueError("num_agents must be a positive integer.")
        if control_period_sec <= 0.0:
            raise ValueError("control_period_sec must be positive.")
        if collision_threshold_m < LIDAR_MIN_RANGE_M:
            collision_threshold_m = LIDAR_MIN_RANGE_M

        self.agent_names = (
            list(agent_names) if agent_names is not None else [f"ugv_{i}" for i in range(num_agents)]
        )
        if len(self.agent_names) != num_agents:
            raise ValueError("agent_names length must match num_agents.")
        self.num_agents = num_agents
        self.leader_name = leader_name or self.agent_names[0]
        if self.leader_name not in self.agent_names:
            raise ValueError("leader_name must be one of agent_names.")

        self.control_period_sec = float(control_period_sec)
        self.observation_timeout_sec = float(observation_timeout_sec)
        self.reset_timeout_sec = float(reset_timeout_sec)
        self.reset_settle_sec = float(reset_settle_sec)
        self.collision_threshold_m = float(collision_threshold_m)
        self.workspace_boundary_m = float(workspace_boundary_m)
        self.use_background_spin = bool(use_background_spin)
        self._closed = False
        self._owns_rclpy = False

        state_device = device if device is not None else "cpu"
        self._target_offsets = self._build_target_offsets(target_offsets or {})
        self._state_processors = {
            agent: StateProcessor(target_offset=self._target_offsets[agent], device=state_device)
            for agent in self.agent_names
        }

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        self.node = Node(node_name)
        self._lock = threading.RLock()
        self._latest_odom: dict[str, Odometry] = {}
        self._latest_scan: dict[str, LaserScan] = {}
        self._subscriptions = []
        self._cmd_publishers = {}

        self._create_ros_interfaces(
            odom_topic_template=odom_topic_template,
            scan_topic_template=scan_topic_template,
            cmd_vel_topic_template=cmd_vel_topic_template,
        )
        self._reset_client = self.node.create_client(Empty, reset_service_name)

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = None
        if self.use_background_spin:
            self._spin_thread = threading.Thread(
                target=self._executor.spin, name="ugv_swarm_env_ros_spin", daemon=True
            )
            self._spin_thread.start()

    def step(self, actions: Any):
        self._ensure_open()
        normalized_actions = self._coerce_actions(actions, self.num_agents)
        physical_actions = self.unnormalize_actions(normalized_actions)

        start_time = time.monotonic()
        self._publish_actions(physical_actions)
        self._sleep_control_period(start_time)

        self._wait_for_observations(self.observation_timeout_sec)
        next_states = self._collect_states()
        dones, info = self._compute_dones_and_info(physical_actions)
        return next_states, dones, info

    def reset(self):
        self._ensure_open()
        self._publish_zero_commands()
        self._call_reset_service()

        with self._lock:
            self._latest_odom.clear()
            self._latest_scan.clear()
        for processor in self._state_processors.values():
            processor.reset()

        self._spin_or_sleep(self.reset_settle_sec)
        self._wait_for_observations(self.observation_timeout_sec)
        return self._collect_states()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._publish_zero_commands()
        self._executor.shutdown()
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self.node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def __enter__(self) -> UGVSwarmEnv:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    @staticmethod
    def unnormalize_actions(actions: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        physical = np.empty_like(clipped, dtype=np.float32)
        physical[:, 0] = (clipped[:, 0] + 1.0) * 0.5 * TB3_MAX_LINEAR_MPS
        physical[:, 1] = clipped[:, 1] * TB3_MAX_ANGULAR_RADPS
        return physical

    @staticmethod
    def _coerce_actions(actions: Any, expected_agents: int) -> np.ndarray:
        if hasattr(actions, "detach") and hasattr(actions, "cpu"):
            actions = actions.detach().cpu().numpy()
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (expected_agents, 2):
            raise ValueError(f"actions must have shape ({expected_agents}, 2); got {action_array.shape}.")
        return action_array

    def _create_ros_interfaces(
        self, odom_topic_template: str, scan_topic_template: str, cmd_vel_topic_template: str
    ) -> None:
        odom_qos = QoSProfile(depth=10)
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        for agent in self.agent_names:
            odom_topic = self._format_topic(odom_topic_template, agent)
            scan_topic = self._format_topic(scan_topic_template, agent)
            cmd_topic = self._format_topic(cmd_vel_topic_template, agent)
            self._subscriptions.append(
                self.node.create_subscription(
                    Odometry, odom_topic, lambda msg, name=agent: self._odom_callback(name, msg), odom_qos
                )
            )
            self._subscriptions.append(
                self.node.create_subscription(
                    LaserScan, scan_topic, lambda msg, name=agent: self._scan_callback(name, msg), scan_qos
                )
            )
            self._cmd_publishers[agent] = self.node.create_publisher(Twist, cmd_topic, 10)

    @staticmethod
    def _format_topic(template: str, agent: str) -> str:
        return template.format(agent=agent).replace("//", "/")

    def _odom_callback(self, agent: str, msg: Odometry) -> None:
        with self._lock:
            self._latest_odom[agent] = msg

    def _scan_callback(self, agent: str, msg: LaserScan) -> None:
        with self._lock:
            self._latest_scan[agent] = msg

    def _publish_actions(self, physical_actions: np.ndarray) -> None:
        for agent, action in zip(self.agent_names, physical_actions, strict=False):
            msg = Twist()
            msg.linear.x = float(action[0])
            msg.angular.z = float(action[1])
            self._cmd_publishers[agent].publish(msg)

    def _publish_zero_commands(self) -> None:
        zero_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self._publish_actions(zero_actions)

    def _sleep_control_period(self, start_time: float) -> None:
        remaining = self.control_period_sec - (time.monotonic() - start_time)
        if remaining > 0.0:
            self._spin_or_sleep(remaining)

    def _spin_or_sleep(self, duration_sec: float) -> None:
        if self.use_background_spin:
            time.sleep(duration_sec)
            return
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=min(0.01, max(0.0, deadline - time.monotonic())))

    def _wait_for_observations(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            with self._lock:
                ready = all(
                    agent in self._latest_odom and agent in self._latest_scan for agent in self.agent_names
                )
            if ready:
                return
            self._spin_or_sleep(0.01)
        missing = self._missing_observations()
        raise TimeoutError(f"Timed out waiting for observations. Missing: {missing}.")

    def _missing_observations(self) -> dict[str, list[str]]:
        with self._lock:
            return {
                "odom": [agent for agent in self.agent_names if agent not in self._latest_odom],
                "scan": [agent for agent in self.agent_names if agent not in self._latest_scan],
            }

    def _collect_states(self):
        with self._lock:
            leader_odom = self._latest_odom[self.leader_name]
            odom_by_agent = {agent: self._latest_odom[agent] for agent in self.agent_names}
            scan_by_agent = {agent: self._latest_scan[agent] for agent in self.agent_names}

        state_tensors = [
            self._state_processors[agent].process(odom_by_agent[agent], leader_odom, scan_by_agent[agent])
            for agent in self.agent_names
        ]
        return self._stack_state_tensors(state_tensors)

    @staticmethod
    def _stack_state_tensors(state_tensors: Sequence[Any]):
        first = state_tensors[0]
        if hasattr(first, "new_empty"):
            return torch.stack(list(state_tensors), dim=0)
        return np.stack(state_tensors, axis=0)

    def _compute_dones_and_info(self, physical_actions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        with self._lock:
            odom_by_agent = {agent: self._latest_odom[agent] for agent in self.agent_names}
            scan_by_agent = {agent: self._latest_scan[agent] for agent in self.agent_names}

        min_lidar = np.asarray(
            [float(StateProcessor.process_lidar(scan_by_agent[agent]).min()) for agent in self.agent_names],
            dtype=np.float32,
        )
        collision = min_lidar <= self.collision_threshold_m
        out_of_bounds = np.asarray(
            [self._is_out_of_bounds(odom_by_agent[agent]) for agent in self.agent_names],
            dtype=bool,
        )
        dones = collision | out_of_bounds
        info = {
            "agent_ids": list(self.agent_names),
            "physical_actions": physical_actions.copy(),
            "min_lidar": min_lidar,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
        }
        return dones, info

    def _is_out_of_bounds(self, odom: Odometry) -> bool:
        pose = StateProcessor.pose_from_odom(odom)
        return abs(pose.x) > self.workspace_boundary_m or abs(pose.y) > self.workspace_boundary_m

    def _call_reset_service(self) -> None:
        if not self._reset_client.wait_for_service(timeout_sec=self.reset_timeout_sec):
            self.node.get_logger().warning("Reset service is unavailable; continuing without Gazebo reset.")
            return
        future = self._reset_client.call_async(Empty.Request())
        deadline = time.monotonic() + self.reset_timeout_sec
        while time.monotonic() < deadline and not future.done():
            self._spin_or_sleep(0.01)
        if not future.done():
            raise TimeoutError("Timed out waiting for Gazebo reset service response.")
        future.result()

    def _build_target_offsets(
        self, target_offsets: Mapping[str, tuple[float, float]]
    ) -> dict[str, tuple[float, float]]:
        offsets: dict[str, tuple[float, float]] = {}
        follower_index = 1
        for agent in self.agent_names:
            if agent in target_offsets:
                offsets[agent] = tuple(target_offsets[agent])
            elif agent == self.leader_name:
                offsets[agent] = (0.0, 0.0)
            else:
                offsets[agent] = (-0.7 * follower_index, 0.0)
                follower_index += 1
        return offsets

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("UGVSwarmEnv is closed.")
