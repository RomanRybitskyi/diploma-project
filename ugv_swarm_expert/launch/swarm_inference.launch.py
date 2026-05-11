from __future__ import annotations

import math
import os
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _column_offsets(followers: list[str], distance: float) -> dict[str, tuple[float, float]]:
    return {follower: (-(i + 1) * distance, 0.0) for i, follower in enumerate(followers)}


def _wedge_offsets(followers: list[str], distance: float, angle_deg: float) -> dict[str, tuple[float, float]]:
    alpha = math.radians(angle_deg)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    offsets: dict[str, tuple[float, float]] = {}
    for index, follower in enumerate(followers):
        i = index + 1
        level = (i + 1) // 2
        dx = -level * distance * cos_a
        dy = ((-1) ** i) * level * distance * sin_a
        offsets[follower] = (dx, dy)
    return offsets


def _compute_offsets(
    followers: list[str],
    formation_type: str,
    distance: float,
    angle_deg: float,
) -> dict[str, tuple[float, float]]:
    if formation_type == "wedge":
        return _wedge_offsets(followers, distance, angle_deg)
    return _column_offsets(followers, distance)


def _build_inference_nodes(context: LaunchContext, *args: Any, **kwargs: Any) -> list:
    def _get(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    model_path = _get("model_path")
    leader_name = _get("leader_name")
    device = _get("device")
    formation_type = _get("formation_type").strip().lower()
    formation_distance = float(_get("formation_distance"))
    formation_angle_deg = float(_get("formation_angle_deg"))
    follower_names_raw = _get("follower_names")
    use_sim_time_str = _get("use_sim_time")
    odom_tpl = _get("odom_topic_template")
    scan_tpl = _get("scan_topic_template")
    cmd_tpl = _get("cmd_vel_topic_template")

    followers = [f.strip() for f in follower_names_raw.split(",") if f.strip()]
    if not followers:
        raise RuntimeError("follower_names must contain at least one entry.")

    offsets = _compute_offsets(followers, formation_type, formation_distance, formation_angle_deg)

    use_sim_time_bool = use_sim_time_str.lower() in ("true", "1", "yes")

    nodes: list = []
    for follower in followers:
        dx, dy = offsets[follower]
        offset_str = f"{dx:.4f},{dy:.4f}"
        nodes.append(
            Node(
                package="ugv_swarm_expert",
                executable="inference_node",
                name=f"inference_{follower.replace('/', '_')}",
                output="screen",
                parameters=[
                    {
                        "robot_namespace": follower,
                        "leader_name": leader_name,
                        "target_offset": offset_str,
                        "model_path": model_path,
                        "device": device,
                        "use_sim_time": use_sim_time_bool,
                        "odom_topic_template": odom_tpl,
                        "scan_topic_template": scan_tpl,
                        "cmd_vel_topic_template": cmd_tpl,
                    }
                ],
            )
        )

    for follower, (dx, dy) in offsets.items():
        nodes.append(LogInfo(msg=f"  inference:{follower}  offset=({dx:.3f}, {dy:.3f})"))

    return nodes


def _build_leader_navigator(context: LaunchContext, *args: Any, **kwargs: Any) -> list:
    def _get(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    use_sim_time_bool = _get("use_sim_time").lower() in ("true", "1", "yes")

    return [
        Node(
            package="ugv_swarm_expert",
            executable="leader_navigator",
            name="leader_navigator",
            output="screen",
            parameters=[
                {
                    "mode": _get("leader_mode"),
                    "leader_name": _get("leader_name"),
                    "teleop_topic": _get("teleop_topic"),
                    "workspace_boundary_m": float(_get("workspace_boundary_m")),
                    "waypoint_tolerance_m": float(_get("waypoint_tolerance_m")),
                    "nav2_action_name": _get("nav2_action_name"),
                    "nav2_frame_id": _get("nav2_frame_id"),
                    "nav2_goal_timeout_sec": float(_get("nav2_goal_timeout_sec")),
                    "use_sim_time": use_sim_time_bool,
                }
            ],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("ugv_swarm_expert")
    default_world = os.path.join(pkg_share, "worlds", "empty_world.sdf")

    args = [
        DeclareLaunchArgument(
            "model_path",
            default_value="checkpoints/offline/actor_epfinal.pth",
            description="Path to the trained Actor .pth checkpoint.",
        ),
        DeclareLaunchArgument(
            "leader_name",
            default_value="leader",
            description="Leader robot namespace.",
        ),
        DeclareLaunchArgument(
            "leader_mode",
            default_value="waypoint",
            description="Leader navigation mode: manual | waypoint | nav2",
        ),
        DeclareLaunchArgument(
            "teleop_topic",
            default_value="/teleop_cmd_vel",
            description="[manual] Teleop Twist topic relayed to leader.",
        ),
        DeclareLaunchArgument(
            "follower_names",
            default_value="tb3_1,tb3_2",
            description="Comma-separated list of follower namespaces.",
        ),
        DeclareLaunchArgument(
            "formation_type",
            default_value="column",
            description="Formation geometry: column | wedge",
        ),
        DeclareLaunchArgument(
            "formation_distance",
            default_value="0.7",
            description="Inter-robot spacing (metres).",
        ),
        DeclareLaunchArgument(
            "formation_angle_deg",
            default_value="45.0",
            description="[wedge] Opening half-angle in degrees.",
        ),
        DeclareLaunchArgument(
            "workspace_boundary_m",
            default_value="8.0",
            description="[waypoint/nav2] Workspace half-side (metres).",
        ),
        DeclareLaunchArgument(
            "waypoint_tolerance_m",
            default_value="0.35",
            description="[waypoint/nav2] Waypoint reached threshold (metres).",
        ),
        DeclareLaunchArgument(
            "nav2_action_name",
            default_value="navigate_to_pose",
            description="[nav2] Nav2 action server name.",
        ),
        DeclareLaunchArgument(
            "nav2_frame_id",
            default_value="map",
            description="[nav2] Goal pose frame id.",
        ),
        DeclareLaunchArgument(
            "nav2_goal_timeout_sec",
            default_value="60.0",
            description="[nav2] Goal cancellation timeout (seconds).",
        ),
        DeclareLaunchArgument(
            "start_simulation",
            default_value="true",
            description="Launch Gazebo (set false if sim is already running).",
        ),
        DeclareLaunchArgument(
            "world_file",
            default_value=default_world,
            description="Absolute path to the Gazebo world SDF file.",
        ),
        DeclareLaunchArgument(
            "use_gui",
            default_value="true",
            description="Start Gazebo GUI (gzclient).",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use /clock from Gazebo.",
        ),
        DeclareLaunchArgument(
            "inference_delay_sec",
            default_value="8.0",
            description="Seconds to wait for simulation/sensors before starting inference.",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="auto",
            description="PyTorch device: cpu | cuda:0 | auto",
        ),
        DeclareLaunchArgument(
            "odom_topic_template",
            default_value="/{agent}/odom",
        ),
        DeclareLaunchArgument(
            "scan_topic_template",
            default_value="/{agent}/scan",
        ),
        DeclareLaunchArgument(
            "cmd_vel_topic_template",
            default_value="/{agent}/cmd_vel",
        ),
    ]

    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, "launch", "swarm_simulation.launch.py")),
        launch_arguments={
            "world_file": LaunchConfiguration("world_file"),
            "use_gui": LaunchConfiguration("use_gui"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
        condition=_IfLaunchConfigTrue("start_simulation"),
    )

    leader_node_action = OpaqueFunction(function=_build_leader_navigator)
    inference_nodes_action = OpaqueFunction(function=_build_inference_nodes)

    delayed_control = TimerAction(
        period=LaunchConfiguration("inference_delay_sec"),
        actions=[
            LogInfo(msg="[swarm_inference] Starting LeaderNavigator and inference nodes…"),
            leader_node_action,
            inference_nodes_action,
        ],
    )

    return LaunchDescription(
        [
            *args,
            sim_launch,
            delayed_control,
        ]
    )


from launch.condition import Condition  # noqa: E402


class _IfLaunchConfigTrue(Condition):
    def __init__(self, name: str) -> None:
        self._name = name
        super().__init__(
            predicate=lambda context: LaunchConfiguration(name).perform(context).strip().lower()
            in ("true", "1", "yes")
        )

    def describe(self) -> str:  # type: ignore[override]
        return f"IfLaunchConfigTrue({self._name!r})"
