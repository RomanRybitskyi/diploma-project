from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("ugv_swarm_expert")

    use_gui_arg = DeclareLaunchArgument(
        "use_gui",
        default_value="true",
        description="Start Gazebo GUI (gzclient). Set false for headless mode.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use /clock published by Gazebo.",
    )

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_share, "launch", "swarm_simulation.launch.py")),
        launch_arguments={
            "world_file": os.path.join(pkg_share, "worlds", "cones_world.sdf"),
            "use_gui": LaunchConfiguration("use_gui"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }.items(),
    )

    return LaunchDescription(
        [
            use_gui_arg,
            use_sim_time_arg,
            simulation,
        ]
    )
