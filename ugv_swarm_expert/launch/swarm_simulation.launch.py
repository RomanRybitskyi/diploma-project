from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

ROBOTS: list[dict] = [
    {"name": "leader", "x": 0.0, "y": 0.0, "yaw": 0.0},
    {"name": "tb3_1", "x": -0.7, "y": 0.0, "yaw": 0.0},
    {"name": "tb3_2", "x": -1.4, "y": 0.0, "yaw": 0.0},
]

SPAWN_DELAY_SEC: float = 5.0


def _tb3_description_dir() -> str:
    try:
        return get_package_share_directory("turtlebot3_description")
    except Exception:
        raise RuntimeError(
            "\n\n[swarm_simulation.launch.py] turtlebot3_description not found.\n"
            "Install it with:\n"
            "    sudo apt install ros-humble-turtlebot3-description\n"
            "or build it from source in your workspace.\n"
        ) from None


def _urdf_path(tb3_desc_dir: str) -> str:
    plain = os.path.join(tb3_desc_dir, "urdf", "turtlebot3_waffle_pi.urdf")
    if os.path.isfile(plain):
        return plain

    xacro = os.path.join(tb3_desc_dir, "urdf", "turtlebot3_waffle_pi.urdf.xacro")
    if os.path.isfile(xacro):
        return xacro

    raise FileNotFoundError(
        f"Cannot find turtlebot3_waffle_pi.urdf[.xacro] under {tb3_desc_dir}/urdf/.\n"
        "Check your turtlebot3_description installation."
    )


def _robot_description_xml(urdf_path: str) -> str:
    if urdf_path.endswith(".xacro"):
        import xacro  # noqa: PLC0415

        return xacro.process_file(urdf_path).toxml()
    return Path(urdf_path).read_text()


def _make_robot_actions(robot: dict, robot_description: str, use_sim_time: LaunchConfiguration) -> list:
    name: str = robot["name"]

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        namespace=name,
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": robot_description,
                "publish_frequency": 50.0,
            }
        ],
        remappings=[
            ("robot_description", f"/{name}/robot_description"),
        ],
    )

    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name=f"spawn_{name}",
        arguments=[
            "-entity",
            name,
            "-topic",
            f"/{name}/robot_description",
            "-robot_namespace",
            name,
            "-x",
            str(robot["x"]),
            "-y",
            str(robot["y"]),
            "-z",
            "0.01",
            "-Y",
            str(robot["yaw"]),
        ],
        output="screen",
    )

    return [rsp, spawn]


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("ugv_swarm_expert")
    tb3_desc_dir = _tb3_description_dir()
    urdf_file = _urdf_path(tb3_desc_dir)
    robot_description_xml = _robot_description_xml(urdf_file)

    world_file_arg = DeclareLaunchArgument(
        "world_file",
        default_value=os.path.join(pkg_share, "worlds", "empty_world.sdf"),
        description="Absolute path to the Gazebo world SDF file.",
    )
    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="true",
        description="Start Gazebo GUI (gzclient). Set false for headless training.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use /clock from Gazebo. Must be true during training.",
    )

    world_file = LaunchConfiguration("world_file")
    use_gui = LaunchConfiguration("use_gui")
    use_sim_time = LaunchConfiguration("use_sim_time")

    set_tb3_model = SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi")

    gzserver = ExecuteProcess(
        cmd=[
            "gzserver",
            "--verbose",
            "-s",
            "libgazebo_ros_init.so",
            "-s",
            "libgazebo_ros_factory.so",
            "-s",
            "libgazebo_ros_state.so",
            world_file,
        ],
        output="screen",
    )

    gzclient = ExecuteProcess(
        cmd=["gzclient", "--verbose"],
        output="screen",
        condition=IfCondition(use_gui),
    )

    all_robot_actions: list = []
    for robot in ROBOTS:
        all_robot_actions.extend(_make_robot_actions(robot, robot_description_xml, use_sim_time))

    delayed_spawn = TimerAction(
        period=SPAWN_DELAY_SEC,
        actions=[
            LogInfo(msg=f"[swarm_simulation] Spawning {len(ROBOTS)} robots…"),
            *all_robot_actions,
        ],
    )

    return LaunchDescription(
        [
            set_tb3_model,
            world_file_arg,
            use_gui_arg,
            use_sim_time_arg,
            gzserver,
            gzclient,
            delayed_spawn,
        ]
    )
