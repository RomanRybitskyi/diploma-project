from __future__ import annotations

import math
import random
import time
from enum import Enum
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from ugv_swarm_expert.constants import (
    CONTROL_PERIOD_SEC,
    TB3_MAX_ANGULAR_RADPS,
    TB3_MAX_LINEAR_MPS,
)

try:
    from nav2_msgs.action import NavigateToPose  # type: ignore[import]
    from rclpy.action import ActionClient  # noqa: F401

    _NAV2_AVAILABLE = True
except ImportError:
    _NAV2_AVAILABLE = False


class NavigatorMode(str, Enum):
    MANUAL = "manual"
    WAYPOINT = "waypoint"
    NAV2 = "nav2"


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def _min_lidar_front(scan: LaserScan, arc_deg: float = 30.0) -> float:
    if not scan.ranges:
        return float("inf")
    half = math.radians(arc_deg / 2.0)
    results: list[float] = []
    for i, r in enumerate(scan.ranges):
        if math.isnan(r) or math.isinf(r) or r <= 0.0:
            continue
        angle = scan.angle_min + i * scan.angle_increment
        if abs(angle) <= half:
            results.append(r)
    return min(results) if results else float("inf")


class LeaderNavigator(Node):
    def __init__(self) -> None:
        super().__init__("leader_navigator")

        mode_str = str(self.declare_parameter("mode", "manual").value).strip().lower()
        try:
            self._mode = NavigatorMode(mode_str)
        except ValueError:
            self.get_logger().error(
                f"Unknown mode '{mode_str}'. Falling back to 'manual'. "
                "Valid options: manual, waypoint, nav2."
            )
            self._mode = NavigatorMode.MANUAL

        self._leader_name: str = str(self.declare_parameter("leader_name", "leader").value).strip("/")
        self._odom_topic: str = (
            str(self.declare_parameter("odom_topic_template", "/{agent}/odom").value)
            .format(agent=self._leader_name)
            .replace("//", "/")
        )
        self._scan_topic: str = (
            str(self.declare_parameter("scan_topic_template", "/{agent}/scan").value)
            .format(agent=self._leader_name)
            .replace("//", "/")
        )
        self._cmd_topic: str = (
            str(self.declare_parameter("cmd_vel_topic_template", "/{agent}/cmd_vel").value)
            .format(agent=self._leader_name)
            .replace("//", "/")
        )

        self._workspace_m: float = float(self.declare_parameter("workspace_boundary_m", 8.0).value)
        self._waypoint_margin_m: float = float(self.declare_parameter("waypoint_margin_m", 1.0).value)
        self._waypoint_tolerance_m: float = float(self.declare_parameter("waypoint_tolerance_m", 0.35).value)
        self._min_wp_dist_m: float = float(self.declare_parameter("min_waypoint_distance_m", 1.5).value)
        seed_val: int = int(self.declare_parameter("seed", -1).value)
        if seed_val >= 0:
            random.seed(seed_val)

        self._k_linear: float = float(self.declare_parameter("k_linear", 0.8).value)
        self._k_angular: float = float(self.declare_parameter("k_angular", 2.2).value)
        self._slowdown_threshold_m: float = float(self.declare_parameter("slowdown_threshold_m", 0.5).value)
        self._slowdown_scale: float = float(self.declare_parameter("slowdown_scale", 0.5).value)
        self._control_period_sec: float = float(
            self.declare_parameter("control_period_sec", CONTROL_PERIOD_SEC).value
        )

        self._teleop_topic: str = str(self.declare_parameter("teleop_topic", "/teleop_cmd_vel").value).strip()

        self._nav2_action: str = str(
            self.declare_parameter("nav2_action_name", "navigate_to_pose").value
        ).strip()
        self._nav2_frame_id: str = str(self.declare_parameter("nav2_frame_id", "map").value).strip()
        self._nav2_goal_timeout_sec: float = float(
            self.declare_parameter("nav2_goal_timeout_sec", 60.0).value
        )

        self._latest_odom: Odometry | None = None
        self._latest_scan: LaserScan | None = None
        self._current_waypoint: tuple[float, float] | None = None
        self._last_log_time: float = 0.0

        self._nav2_client: Any = None
        self._nav2_goal_handle: Any = None
        self._nav2_goal_active: bool = False
        self._nav2_goal_sent_time: float = 0.0

        odom_qos = QoSProfile(depth=10)
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._subscriptions: list = []

        if self._mode != NavigatorMode.MANUAL:
            self._subscriptions.append(
                self.create_subscription(Odometry, self._odom_topic, self._odom_cb, odom_qos)
            )
            self._subscriptions.append(
                self.create_subscription(LaserScan, self._scan_topic, self._scan_cb, scan_qos)
            )

        self._cmd_pub = self.create_publisher(Twist, self._cmd_topic, 10)

        if self._mode == NavigatorMode.MANUAL and self._teleop_topic:
            self._subscriptions.append(
                self.create_subscription(Twist, self._teleop_topic, self._teleop_relay_cb, 10)
            )
            self.get_logger().info(
                f"[LeaderNavigator] Mode: MANUAL  |  Relaying '{self._teleop_topic}' → '{self._cmd_topic}'"
            )

        elif self._mode == NavigatorMode.WAYPOINT:
            self._timer = self.create_timer(self._control_period_sec, self._waypoint_tick)
            self.get_logger().info(
                f"[LeaderNavigator] Mode: WAYPOINT  |  workspace=±{self._workspace_m} m  "
                f"cmd→'{self._cmd_topic}'"
            )

        elif self._mode == NavigatorMode.NAV2:
            if not _NAV2_AVAILABLE:
                raise RuntimeError(
                    "nav2_msgs is not installed. "
                    "Install it with: sudo apt install ros-$ROS_DISTRO-nav2-msgs"
                )
            from rclpy.action import ActionClient as _ActionClient

            self._nav2_client = _ActionClient(self, NavigateToPose, self._nav2_action)
            self._timer = self.create_timer(1.0, self._nav2_tick)
            self.get_logger().info(
                f"[LeaderNavigator] Mode: NAV2  |  action='{self._nav2_action}'  "
                f"frame='{self._nav2_frame_id}'  workspace=±{self._workspace_m} m"
            )

        else:
            self.get_logger().info(
                f"[LeaderNavigator] Mode: MANUAL  |  No teleop_topic set. "
                f"Publish directly to '{self._cmd_topic}'."
            )

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg

    def _teleop_relay_cb(self, msg: Twist) -> None:
        self._cmd_pub.publish(msg)

    def _waypoint_tick(self) -> None:
        if self._latest_odom is None:
            self._log_throttled("Waiting for leader odometry…")
            return

        leader_x, leader_y, leader_yaw = self._pose_from_odom(self._latest_odom)

        if self._current_waypoint is None:
            self._current_waypoint = self._generate_waypoint(leader_x, leader_y)
            self.get_logger().info(
                f"[LeaderNavigator] New waypoint: ({self._current_waypoint[0]:.2f}, "
                f"{self._current_waypoint[1]:.2f})"
            )

        wx, wy = self._current_waypoint
        dx = wx - leader_x
        dy = wy - leader_y
        distance = math.hypot(dx, dy)

        if distance <= self._waypoint_tolerance_m:
            self.get_logger().info(
                f"[LeaderNavigator] Waypoint reached  ({wx:.2f}, {wy:.2f}). Generating next…"
            )
            self._current_waypoint = self._generate_waypoint(leader_x, leader_y)
            self.get_logger().info(
                f"[LeaderNavigator] New waypoint: ({self._current_waypoint[0]:.2f}, "
                f"{self._current_waypoint[1]:.2f})"
            )
            self._publish_velocity(0.0, 0.0)
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = _normalize_angle(desired_heading - leader_yaw)

        linear = self._k_linear * distance * max(0.0, math.cos(heading_error))
        angular = self._k_angular * heading_error

        if self._latest_scan is not None:
            front_dist = _min_lidar_front(self._latest_scan)
            if front_dist < self._slowdown_threshold_m:
                linear *= self._slowdown_scale

        linear = _clip(linear, 0.0, TB3_MAX_LINEAR_MPS)
        angular = _clip(angular, -TB3_MAX_ANGULAR_RADPS, TB3_MAX_ANGULAR_RADPS)
        self._publish_velocity(linear, angular)

    def _nav2_tick(self) -> None:
        if not self._nav2_client.server_is_ready():
            self._log_throttled(f"Waiting for Nav2 action server '{self._nav2_action}'…")
            return

        if self._nav2_goal_active:
            elapsed = time.monotonic() - self._nav2_goal_sent_time
            if elapsed > self._nav2_goal_timeout_sec:
                self.get_logger().warn(
                    f"[LeaderNavigator] Nav2 goal timed out after {elapsed:.1f} s. " "Sending next waypoint."
                )
                self._nav2_goal_active = False
                if self._nav2_goal_handle is not None:
                    self._nav2_goal_handle.cancel_goal_async()
                    self._nav2_goal_handle = None
            else:
                return

        if self._latest_odom is None:
            self._log_throttled("Waiting for leader odometry (nav2 mode)…")
            return

        leader_x, leader_y, _ = self._pose_from_odom(self._latest_odom)
        waypoint = self._generate_waypoint(leader_x, leader_y)
        self._send_nav2_goal(waypoint)

    def _send_nav2_goal(self, waypoint: tuple[float, float]) -> None:
        wx, wy = waypoint
        dx = wx - (self._pose_from_odom(self._latest_odom)[0] if self._latest_odom else 0.0)
        dy = wy - (self._pose_from_odom(self._latest_odom)[1] if self._latest_odom else 0.0)
        goal_yaw = math.atan2(dy, dx)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self._nav2_frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = wx
        goal_msg.pose.pose.position.y = wy
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation = _yaw_to_quaternion(goal_yaw)

        self.get_logger().info(
            f"[LeaderNavigator] Sending Nav2 goal: ({wx:.2f}, {wy:.2f})  " f"frame='{self._nav2_frame_id}'"
        )

        send_future = self._nav2_client.send_goal_async(
            goal_msg,
            feedback_callback=self._nav2_feedback_cb,
        )
        send_future.add_done_callback(self._nav2_goal_response_cb)
        self._nav2_goal_active = True
        self._nav2_goal_sent_time = time.monotonic()

    def _nav2_goal_response_cb(self, future: Any) -> None:
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn("[LeaderNavigator] Nav2 rejected the goal.")
            self._nav2_goal_active = False
            self._nav2_goal_handle = None
            return
        self._nav2_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav2_result_cb)

    def _nav2_result_cb(self, future: Any) -> None:
        self._nav2_goal_active = False
        self._nav2_goal_handle = None
        try:
            result = future.result()
            status = result.status
            self.get_logger().info(f"[LeaderNavigator] Nav2 goal finished with status {status}.")
        except Exception as exc:
            self.get_logger().error(f"[LeaderNavigator] Nav2 result callback error: {exc}")

    def _nav2_feedback_cb(self, feedback_msg: Any) -> None:
        pass

    def _generate_waypoint(self, current_x: float, current_y: float) -> tuple[float, float]:
        lo = -(self._workspace_m - self._waypoint_margin_m)
        hi = self._workspace_m - self._waypoint_margin_m
        for _ in range(200):
            wx = random.uniform(lo, hi)
            wy = random.uniform(lo, hi)
            if math.hypot(wx - current_x, wy - current_y) >= self._min_wp_dist_m:
                return wx, wy
        wx = hi if current_x < 0.0 else lo
        wy = hi if current_y < 0.0 else lo
        return wx, wy

    @staticmethod
    def _pose_from_odom(msg: Odometry) -> tuple[float, float, float]:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        yaw = _yaw_from_quaternion(ori.x, ori.y, ori.z, ori.w)
        return float(pos.x), float(pos.y), yaw

    def _publish_velocity(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)

    def _log_throttled(self, text: str, interval_sec: float = 2.0) -> None:
        now = time.monotonic()
        if now - self._last_log_time >= interval_sec:
            self.get_logger().info(text)
            self._last_log_time = now

    def destroy_node(self) -> bool:
        self._publish_velocity(0.0, 0.0)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LeaderNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
