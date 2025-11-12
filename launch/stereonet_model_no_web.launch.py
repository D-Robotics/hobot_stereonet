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
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition

def generate_launch_description():

    os.environ['ROS_LOG_DIR'] = '/userdata/.roslog'

    node_list = []

    node_list.append(DeclareLaunchArgument(
        'use_mipi_cam',
        default_value='True',
        description='use_mipi_cam'
    ))

    # stereonet node
    stereonet_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('hobot_stereonet'),
                                                   'launch/stereonet_model.launch.py')),
        launch_arguments={
            'log_level': 'info',
        }.items(),
    )
    node_list.append(stereonet_node)

    # mipi node
    dual_mipi_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("mipi_cam"),
                "launch/mipi_cam_dual_channel.launch.py",
            )
        ),
        launch_arguments={
            # "mipi_frame_ts_type": "sensor", # default is sensor in mipi_cam launch
            "frame_id": "pcl_link",
            "log_level": "warn",
        }.items(),
        condition=IfCondition(
            PythonExpression([
                LaunchConfiguration('use_mipi_cam'),
                ' and ',
                'not ', LaunchConfiguration('use_local_image_flag')
            ])
        )
    )
    node_list.append(dual_mipi_cam)

    return LaunchDescription(node_list)
