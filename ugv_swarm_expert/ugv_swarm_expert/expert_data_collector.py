from __future__ import annotations

import csv
import math
import os
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

CONTROL_PERIOD_SEC = 0.1
LIDAR_SECTOR_COUNT = 36
LIDAR_MIN_RANGE_M = 0.12
LIDAR_MAX_RANGE_M = 3.5
TB3_MAX_LINEAR_MPS = 0.22
TB3_MAX_ANGULAR_RADPS = 2.84
CSV_COLUMNS = [
    "time_step",
    "pos_x",
    "pos_y",
    "yaw",
    "rel_dist_lead",
    "rel_ang_lead",
    *[f"lidar_s{i}" for i in range(1, LIDAR_SECTOR_COUNT + 1)],
    "target_v",
    "target_w",
]


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class ExpertAction:
    linear: float
    angular: float


class BufferedCsvWriter:
    def __init__(
        self,
        output_dir: Path,
        followers: Sequence[str],
        file_prefix: str,
        fieldnames: Sequence[str],
        flush_every: int = 25,
        flush_interval_sec: float = 1.0,
        max_queue_size: int = 10_000,
    ):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fieldnames = list(fieldnames)
        self._flush_every = max(1, flush_every)
        self._flush_interval_sec = max(0.1, flush_interval_sec)
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._files = {}
        self._writers: dict[str, csv.writer] = {}
        self._rows_since_flush: dict[str, int] = {follower: 0 for follower in followers}
        self._last_flush = time.monotonic()

        for follower in followers:
            safe_name = follower.strip("/").replace("/", "_") or "follower"
            path = self._output_dir / f"{file_prefix}_{safe_name}.csv"
            file_exists = path.exists() and path.stat().st_size > 0
            handle = path.open("a", newline="", buffering=1)
            writer = csv.writer(handle)
            if not file_exists:
                writer.writerow(self._fieldnames)
                handle.flush()
            self._files[follower] = handle
            self._writers[follower] = writer

        self._worker = threading.Thread(target=self._run, name="expert_csv_writer", daemon=True)
        self._worker.start()

    def enqueue(self, follower: str, row: list[float]) -> bool:
        try:
            self._queue.put_nowait((follower, row))
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._worker.join(timeout=5.0)
        self._flush_all()
        for handle in self._files.values():
            handle.close()

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=self._flush_interval_sec)
            except queue.Empty:
                self._flush_all()
                continue

            if item is None:
                self._queue.task_done()
                continue

            follower, row = item
            writer = self._writers.get(follower)
            if writer is not None:
                writer.writerow(row)
                self._rows_since_flush[follower] += 1
                if self._should_flush(follower):
                    self._files[follower].flush()
                    self._rows_since_flush[follower] = 0
                    self._last_flush = time.monotonic()
            self._queue.task_done()

        self._flush_all()

    def _should_flush(self, follower: str) -> bool:
        return (
            self._rows_since_flush[follower] >= self._flush_every
            or time.monotonic() - self._last_flush >= self._flush_interval_sec
        )

    def _flush_all(self) -> None:
        for follower, handle in self._files.items():
            if self._rows_since_flush.get(follower, 0) > 0:
                handle.flush()
                self._rows_since_flush[follower] = 0
        self._last_flush = time.monotonic()


class ExpertDataCollector(Node):
    def __init__(self):
        super().__init__("expert_data_collector")

        self._leader_name = self.declare_parameter("leader_name", "leader").value
        self._follower_names = list(self.declare_parameter("follower_names", ["tb3_1", "tb3_2"]).value)
        if not self._follower_names:
            raise ValueError("Parameter 'follower_names' must contain at least one follower.")

        formation_distance = float(self.declare_parameter("formation_distance", 0.7).value)
        offsets_text = str(self.declare_parameter("formation_offsets", "").value).strip()
        self._formation_offsets = self._parse_or_generate_offsets(
            offsets_text, self._follower_names, formation_distance
        )

        self._odom_topic_template = str(self.declare_parameter("odom_topic_template", "/{agent}/odom").value)
        self._scan_topic_template = str(self.declare_parameter("scan_topic_template", "/{agent}/scan").value)
        self._cmd_vel_topic_template = str(
            self.declare_parameter("cmd_vel_topic_template", "/{agent}/cmd_vel").value
        )
        self._publish_commands = bool(self.declare_parameter("publish_commands", True).value)

        self._k_linear = float(self.declare_parameter("k_linear", 0.8).value)
        self._k_angular = float(self.declare_parameter("k_angular", 2.2).value)
        self._position_tolerance = float(self.declare_parameter("position_tolerance", 0.02).value)
        self._max_data_age_sec = float(self.declare_parameter("max_data_age_sec", 0.5).value)

        output_dir = Path(
            os.path.expanduser(str(self.declare_parameter("output_dir", "~/ugv_swarm_expert_data").value))
        )
        file_prefix = str(self.declare_parameter("file_prefix", "expert_data").value)
        flush_every = int(self.declare_parameter("csv_flush_every", 25).value)
        flush_interval = float(self.declare_parameter("csv_flush_interval_sec", 1.0).value)
        self._csv_writer = BufferedCsvWriter(
            output_dir=output_dir,
            followers=self._follower_names,
            file_prefix=file_prefix,
            fieldnames=CSV_COLUMNS,
            flush_every=flush_every,
            flush_interval_sec=flush_interval,
        )

        self._latest_odom: dict[str, Odometry] = {}
        self._latest_scan: dict[str, LaserScan] = {}
        self._subscriptions = []
        self._cmd_publishers = {}
        self._time_step = 0
        self._last_skip_log_time = 0.0
        self._last_queue_warning_time = 0.0

        self._create_ros_interfaces()
        self._timer = self.create_timer(CONTROL_PERIOD_SEC, self._on_timer)

        self.get_logger().info(
            f"Expert collector running at 10 Hz for leader '{self._leader_name}' "
            f"and followers {self._follower_names}. CSV output: {output_dir}"
        )

    def destroy_node(self) -> bool:
        self._publish_zero_commands()
        self._csv_writer.close()
        return super().destroy_node()

    def _create_ros_interfaces(self) -> None:
        odom_qos = QoSProfile(depth=10)
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        for agent in [self._leader_name, *self._follower_names]:
            odom_topic = self._format_topic(self._odom_topic_template, agent)
            scan_topic = self._format_topic(self._scan_topic_template, agent)
            self._subscriptions.append(
                self.create_subscription(Odometry, odom_topic, partial(self._odom_callback, agent), odom_qos)
            )
            self._subscriptions.append(
                self.create_subscription(LaserScan, scan_topic, partial(self._scan_callback, agent), scan_qos)
            )

        if self._publish_commands:
            for follower in self._follower_names:
                cmd_topic = self._format_topic(self._cmd_vel_topic_template, follower)
                self._cmd_publishers[follower] = self.create_publisher(Twist, cmd_topic, 10)

    @staticmethod
    def _format_topic(template: str, agent: str) -> str:
        return template.format(agent=agent).replace("//", "/")

    def _odom_callback(self, agent: str, msg: Odometry) -> None:
        self._latest_odom[agent] = msg

    def _scan_callback(self, agent: str, msg: LaserScan) -> None:
        self._latest_scan[agent] = msg

    def _on_timer(self) -> None:
        if not self._all_messages_ready():
            self._log_skip_throttled("Waiting for odometry and LiDAR from all agents.")
            return

        if not self._all_messages_fresh():
            self._log_skip_throttled("Skipping tick because one or more sensor messages are stale.")
            return

        leader_pose = self._pose_from_odom(self._latest_odom[self._leader_name])

        for follower in self._follower_names:
            follower_pose = self._pose_from_odom(self._latest_odom[follower])
            target_position = self._target_position(leader_pose, self._formation_offsets[follower])
            action = self._compute_expert_action(follower_pose, target_position)
            lidar_sectors = self.process_lidar_scan(self._latest_scan[follower])
            rel_dist, rel_ang = self._relative_to_leader_frame(leader_pose, follower_pose)

            if self._publish_commands:
                self._publish_action(follower, action)

            row = [
                self._time_step,
                follower_pose.x,
                follower_pose.y,
                follower_pose.yaw,
                rel_dist,
                rel_ang,
                *lidar_sectors,
                action.linear,
                action.angular,
            ]
            if not self._csv_writer.enqueue(follower, row):
                self._log_queue_warning_throttled()

        self._time_step += 1

    def _all_messages_ready(self) -> bool:
        agents = [self._leader_name, *self._follower_names]
        return all(agent in self._latest_odom and agent in self._latest_scan for agent in agents)

    def _all_messages_fresh(self) -> bool:
        if self._max_data_age_sec <= 0.0:
            return True

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec <= 0.0:
            return True

        for msg in [*self._latest_odom.values(), *self._latest_scan.values()]:
            stamp_sec = self._stamp_to_seconds(msg.header.stamp)
            if stamp_sec <= 0.0:
                continue
            if now_sec - stamp_sec > self._max_data_age_sec:
                return False
        return True

    @staticmethod
    def _stamp_to_seconds(stamp: object) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _pose_from_odom(msg: Odometry) -> Pose2D:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = ExpertDataCollector._yaw_from_quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        return Pose2D(x=float(position.x), y=float(position.y), yaw=yaw)

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _parse_or_generate_offsets(
        offsets_text: str, followers: Sequence[str], formation_distance: float
    ) -> dict[str, tuple[float, float]]:
        if not offsets_text:
            return {
                follower: (-(index + 1) * formation_distance, 0.0) for index, follower in enumerate(followers)
            }

        parsed_offsets: list[tuple[float, float]] = []
        for raw_pair in offsets_text.split(";"):
            pair = raw_pair.strip()
            if not pair:
                continue
            values = [value.strip() for value in pair.split(",")]
            if len(values) != 2:
                raise ValueError("Parameter 'formation_offsets' must use 'dx,dy;dx,dy' format.")
            parsed_offsets.append((float(values[0]), float(values[1])))

        if len(parsed_offsets) != len(followers):
            raise ValueError("Parameter 'formation_offsets' must provide exactly one offset per follower.")
        return dict(zip(followers, parsed_offsets, strict=False))

    @staticmethod
    def _target_position(leader_pose: Pose2D, offset: tuple[float, float]) -> tuple[float, float]:
        cos_yaw = math.cos(leader_pose.yaw)
        sin_yaw = math.sin(leader_pose.yaw)
        dx, dy = offset
        target_x = leader_pose.x + cos_yaw * dx - sin_yaw * dy
        target_y = leader_pose.y + sin_yaw * dx + cos_yaw * dy
        return target_x, target_y

    def _compute_expert_action(
        self, follower_pose: Pose2D, target_position: tuple[float, float]
    ) -> ExpertAction:
        dx = target_position[0] - follower_pose.x
        dy = target_position[1] - follower_pose.y
        distance = math.hypot(dx, dy)

        if distance <= self._position_tolerance:
            return ExpertAction(linear=0.0, angular=0.0)

        desired_heading = math.atan2(dy, dx)
        heading_error = self._normalize_angle(desired_heading - follower_pose.yaw)

        linear = self._k_linear * distance * max(0.0, math.cos(heading_error))
        angular = self._k_angular * heading_error

        return ExpertAction(
            linear=self._clip(linear, 0.0, TB3_MAX_LINEAR_MPS),
            angular=self._clip(angular, -TB3_MAX_ANGULAR_RADPS, TB3_MAX_ANGULAR_RADPS),
        )

    @staticmethod
    def process_lidar_scan(scan: LaserScan) -> list[float]:
        ranges = list(scan.ranges)
        if not ranges:
            return [LIDAR_MAX_RANGE_M] * LIDAR_SECTOR_COUNT

        filtered = [ExpertDataCollector._filter_lidar_range(value) for value in ranges]
        sector_size = len(filtered) / LIDAR_SECTOR_COUNT
        sectors: list[float] = []

        for sector_index in range(LIDAR_SECTOR_COUNT):
            start = int(math.floor(sector_index * sector_size))
            end = int(math.floor((sector_index + 1) * sector_size))
            if sector_index == LIDAR_SECTOR_COUNT - 1:
                end = len(filtered)
            if end <= start:
                end = min(start + 1, len(filtered))
            sectors.append(min(filtered[start:end]))

        return sectors

    @staticmethod
    def _filter_lidar_range(value: float) -> float:
        if math.isnan(value) or math.isinf(value) or value > LIDAR_MAX_RANGE_M:
            return LIDAR_MAX_RANGE_M
        if value < LIDAR_MIN_RANGE_M:
            return LIDAR_MIN_RANGE_M
        return float(value)

    @staticmethod
    def _relative_to_leader_frame(leader_pose: Pose2D, follower_pose: Pose2D) -> tuple[float, float]:
        dx = follower_pose.x - leader_pose.x
        dy = follower_pose.y - leader_pose.y
        cos_yaw = math.cos(leader_pose.yaw)
        sin_yaw = math.sin(leader_pose.yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        return math.hypot(local_x, local_y), math.atan2(local_y, local_x)

    def _publish_action(self, follower: str, action: ExpertAction) -> None:
        publisher = self._cmd_publishers.get(follower)
        if publisher is None:
            return
        msg = Twist()
        msg.linear.x = action.linear
        msg.angular.z = action.angular
        publisher.publish(msg)

    def _publish_zero_commands(self) -> None:
        if not self._publish_commands:
            return
        zero = Twist()
        for publisher in self._cmd_publishers.values():
            publisher.publish(zero)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

    def _log_skip_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_skip_log_time >= 2.0:
            self.get_logger().warn(message)
            self._last_skip_log_time = now

    def _log_queue_warning_throttled(self) -> None:
        now = time.monotonic()
        if now - self._last_queue_warning_time >= 2.0:
            self.get_logger().error("CSV writer queue is full; dropping expert demonstration row.")
            self._last_queue_warning_time = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExpertDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
