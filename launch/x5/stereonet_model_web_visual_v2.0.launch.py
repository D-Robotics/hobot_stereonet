# Copyright (c) 2024，D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    stereonet_model_file_path =  os.path.join(
        get_package_share_directory('hobot_stereonet'),
        'config',
        'DStereoV2.0.bin'
    )

    # 双目深度估计模型
    stereonet_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_stereonet"),
                "launch/stereonet_model_web_visual.launch.py",
            )
        ),
        launch_arguments = {
            "stereonet_model_file_path": stereonet_model_file_path,
            "postprocess": "v2",
        }.items(),
    )

    return LaunchDescription(
        [
            stereonet_node
        ]
    )
