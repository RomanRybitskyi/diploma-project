from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from torch import nn
from torch.distributions import Distribution

from ugv_swarm_expert.actor_network import ActorNetwork
from ugv_swarm_expert.state_processor import (
    TB3_MAX_ANGULAR_RADPS,
    TB3_MAX_LINEAR_MPS,
    StateProcessor,
)

CONTROL_PERIOD_SEC = 0.1
DEFAULT_ROBOT_NAMESPACE = "ugv_1"
DEFAULT_LEADER_NAME = "leader"
DEFAULT_MODEL_PATH = "checkpoints/actor_ep500.pth"


class UGVInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("ugv_inference_node")

        self.robot_namespace = self._normalize_namespace(
            self._declare_parameter_value("robot_namespace", DEFAULT_ROBOT_NAMESPACE)
        )
        namespace_override = self._normalize_namespace(self._declare_parameter_value("namespace", ""))
        if namespace_override:
            self.robot_namespace = namespace_override

        self.leader_name = self._normalize_namespace(
            self._declare_parameter_value("leader_name", DEFAULT_LEADER_NAME)
        )
        self.model_path = Path(
            os.path.expanduser(str(self._declare_parameter_value("model_path", DEFAULT_MODEL_PATH)))
        )
        self.device = self._resolve_device(str(self._declare_parameter_value("device", "auto")))
        self.target_offset = self._parse_target_offset(
            self._declare_parameter_value("target_offset", [-0.7, 0.0])
        )

        self.odom_topic_template = str(self._declare_parameter_value("odom_topic_template", "/{agent}/odom"))
        self.scan_topic_template = str(self._declare_parameter_value("scan_topic_template", "/{agent}/scan"))
        self.cmd_vel_topic_template = str(
            self._declare_parameter_value("cmd_vel_topic_template", "/{agent}/cmd_vel")
        )
        self.leader_odom_topic = str(
            self._declare_parameter_value("leader_odom_topic", f"/{self.leader_name}/odom")
        )
        self.control_period_sec = float(
            self._declare_parameter_value("control_period_sec", CONTROL_PERIOD_SEC)
        )
        if self.control_period_sec <= 0.0:
            raise ValueError("control_period_sec must be positive.")

        self._latest_local_odom: Odometry | None = None
        self._latest_leader_odom: Odometry | None = None
        self._latest_scan: LaserScan | None = None
        self._last_wait_log_time = 0.0

        self.state_processor = StateProcessor(target_offset=self.target_offset, device=self.device)
        self.model = ActorNetwork().to(self.device)
        self._load_model_weights(self.model, self.model_path, self.device)
        self.model.eval()

        self._subscriptions = []
        self._create_ros_interfaces()
        self._timer = self.create_timer(self.control_period_sec, self.timer_callback)

        self.get_logger().info(
            f"Loaded Actor policy from '{self.model_path}' on {self.device}. "
            f"Running decentralized inference for '/{self.robot_namespace}' at "
            f"{1.0 / self.control_period_sec:.1f} Hz."
        )

    def timer_callback(self) -> None:
        if not self._messages_ready():
            self._publish_zero_command()
            self._log_waiting_throttled()
            return

        if self._latest_local_odom is None or self._latest_leader_odom is None or self._latest_scan is None:
            self._publish_zero_command()
            return

        try:
            state = self.state_processor.process(
                self._latest_local_odom,
                self._latest_leader_odom,
                self._latest_scan,
            )
            if not self.state_processor.is_ready:
                self._publish_zero_command()
                return

            batched_state = state.unsqueeze(0).to(self.device)
            with torch.no_grad():
                model_output = self.model(batched_state)
                action = self._deterministic_action(model_output)

            velocity = self.unnormalize_action(action.squeeze(0))
            self._publish_velocity(float(velocity[0]), float(velocity[1]))
        except Exception as exc:
            self.get_logger().error(f"Inference step failed; publishing zero command. Error: {exc}")
            self._publish_zero_command()

    def destroy_node(self) -> bool:
        with contextlib.suppress(Exception):
            self._publish_zero_command()
        return super().destroy_node()

    @staticmethod
    def unnormalize_action(action: torch.Tensor) -> torch.Tensor:
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if action.shape != (2,):
            raise ValueError(f"action must have shape (2,), got {tuple(action.shape)}.")
        clipped = torch.clamp(action.to(dtype=torch.float32), -1.0, 1.0)
        linear = (clipped[0] + 1.0) * 0.5 * TB3_MAX_LINEAR_MPS
        angular = clipped[1] * TB3_MAX_ANGULAR_RADPS
        return torch.stack((linear, angular), dim=0)

    def _create_ros_interfaces(self) -> None:
        odom_qos = QoSProfile(depth=10)
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        local_odom_topic = self._format_topic(self.odom_topic_template, self.robot_namespace)
        local_scan_topic = self._format_topic(self.scan_topic_template, self.robot_namespace)
        cmd_vel_topic = self._format_topic(self.cmd_vel_topic_template, self.robot_namespace)

        self._subscriptions.append(
            self.create_subscription(Odometry, local_odom_topic, self._local_odom_callback, odom_qos)
        )
        self._subscriptions.append(
            self.create_subscription(LaserScan, local_scan_topic, self._scan_callback, scan_qos)
        )
        self._subscriptions.append(
            self.create_subscription(Odometry, self.leader_odom_topic, self._leader_odom_callback, odom_qos)
        )
        self._cmd_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)

    @staticmethod
    def _format_topic(template: str, agent: str) -> str:
        return template.format(agent=agent).replace("//", "/")

    def _local_odom_callback(self, msg: Odometry) -> None:
        self._latest_local_odom = msg

    def _leader_odom_callback(self, msg: Odometry) -> None:
        self._latest_leader_odom = msg

    def _scan_callback(self, msg: LaserScan) -> None:
        self._latest_scan = msg

    def _messages_ready(self) -> bool:
        return (
            self._latest_local_odom is not None
            and self._latest_leader_odom is not None
            and self._latest_scan is not None
        )

    def _publish_velocity(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._cmd_publisher.publish(msg)

    def _publish_zero_command(self) -> None:
        self._publish_velocity(0.0, 0.0)

    def _log_waiting_throttled(self) -> None:
        now = time.monotonic()
        if now - self._last_wait_log_time >= 2.0:
            self.get_logger().warning("Waiting for local odom, local scan, and leader odom before inference.")
            self._last_wait_log_time = now

    @staticmethod
    def _deterministic_action(model_output: Any) -> torch.Tensor:
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
        if action.ndim != 2 or action.shape != (1, 2):
            raise ValueError(f"Actor deterministic action must have shape (1, 2); got {tuple(action.shape)}.")
        return action

    @staticmethod
    def _load_model_weights(model: nn.Module, model_path: Path, device: torch.device) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Actor checkpoint does not exist: {model_path}")

        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = UGVInferenceNode._extract_state_dict(checkpoint)
        model.load_state_dict(state_dict)

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
        if isinstance(checkpoint, Mapping):
            for key in ("actor_state_dict", "model_state_dict", "state_dict", "actor"):
                value = checkpoint.get(key)
                if isinstance(value, Mapping):
                    return UGVInferenceNode._strip_module_prefix(value)
            if all(isinstance(key, str) for key in checkpoint):
                return UGVInferenceNode._strip_module_prefix(checkpoint)
        raise TypeError(
            "Unsupported checkpoint format. Expected a state_dict or a dict containing actor weights."
        )

    @staticmethod
    def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {(key[7:] if key.startswith("module.") else key): value for key, value in state_dict.items()}

    @staticmethod
    def _parse_target_offset(value: Any) -> tuple[float, float]:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
        elif isinstance(value, Sequence):
            parts = list(value)
        else:
            raise TypeError("target_offset must be a sequence of two floats or a 'x,y' string.")
        if len(parts) != 2:
            raise ValueError("target_offset must contain exactly two values.")
        return float(parts[0]), float(parts[1])

    @staticmethod
    def _normalize_namespace(value: Any) -> str:
        return str(value).strip().strip("/")

    @staticmethod
    def _resolve_device(device_text: str) -> torch.device:
        normalized = device_text.strip().lower()
        if normalized in ("", "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_text)

    def _declare_parameter_value(self, name: str, default: Any) -> Any:
        return self.declare_parameter(name, default).value


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UGVInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
