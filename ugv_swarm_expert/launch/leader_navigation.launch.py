from __future__ import annotations

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="manual",
        description="Navigator mode: manual | waypoint | nav2",
    )
    leader_name_arg = DeclareLaunchArgument(
        "leader_name",
        default_value="leader",
        description="Leader robot namespace / name.",
    )
    teleop_topic_arg = DeclareLaunchArgument(
        "teleop_topic",
        default_value="/teleop_cmd_vel",
        description="[manual] Incoming Twist topic to relay to the leader.",
    )
    workspace_arg = DeclareLaunchArgument(
        "workspace_boundary_m",
        default_value="8.0",
        description="[waypoint/nav2] Half-side of the square workspace (metres).",
    )
    tolerance_arg = DeclareLaunchArgument(
        "waypoint_tolerance_m",
        default_value="0.35",
        description="[waypoint/nav2] Goal-reached distance threshold (metres).",
    )
    seed_arg = DeclareLaunchArgument(
        "seed",
        default_value="-1",
        description="[waypoint/nav2] RNG seed. -1 = non-deterministic.",
    )
    nav2_action_arg = DeclareLaunchArgument(
        "nav2_action_name",
        default_value="navigate_to_pose",
        description="[nav2] Name of the Nav2 NavigateToPose action server.",
    )
    nav2_frame_arg = DeclareLaunchArgument(
        "nav2_frame_id",
        default_value="map",
        description="[nav2] TF frame for goal poses sent to Nav2.",
    )
    nav2_timeout_arg = DeclareLaunchArgument(
        "nav2_goal_timeout_sec",
        default_value="60.0",
        description="[nav2] Seconds before an unfinished Nav2 goal is cancelled.",
    )

    navigator_node = Node(
        package="ugv_swarm_expert",
        executable="leader_navigator",
        name="leader_navigator",
        output="screen",
        parameters=[
            {
                "mode": LaunchConfiguration("mode"),
                "leader_name": LaunchConfiguration("leader_name"),
                "teleop_topic": LaunchConfiguration("teleop_topic"),
                "workspace_boundary_m": LaunchConfiguration("workspace_boundary_m"),
                "waypoint_tolerance_m": LaunchConfiguration("waypoint_tolerance_m"),
                "seed": LaunchConfiguration("seed"),
                "nav2_action_name": LaunchConfiguration("nav2_action_name"),
                "nav2_frame_id": LaunchConfiguration("nav2_frame_id"),
                "nav2_goal_timeout_sec": LaunchConfiguration("nav2_goal_timeout_sec"),
            }
        ],
    )

    return LaunchDescription(
        [
            mode_arg,
            leader_name_arg,
            teleop_topic_arg,
            workspace_arg,
            tolerance_arg,
            seed_arg,
            nav2_action_arg,
            nav2_frame_arg,
            nav2_timeout_arg,
            navigator_node,
        ]
    )
