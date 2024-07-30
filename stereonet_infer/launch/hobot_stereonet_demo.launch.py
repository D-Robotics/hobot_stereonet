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
import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory
from ament_index_python.packages import get_package_prefix

def generate_launch_description():
    # 启动双目数据采集和发布
    stereo_usb_cam_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_stereo_usb_cam'),
                'launch/hobot_stereo_usb_cam.launch.py')),
        launch_arguments={
            'enable_fb': 'False',
            'video_device': '0'
        }.items()
    )

    # 启动推理
    # hobot_stereonet_node = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory('hobot_stereonet'),
    #             'launch/hobot_stereonet_node.launch.py')),
    # )
    pkg_path = os.path.join(
        get_package_prefix('hobot_stereonet'),
        "lib/hobot_stereonet")
    print("hobot_stereonet path is ", pkg_path)
    config_file_launch_arg = DeclareLaunchArgument(
        "config_file", default_value=TextSubstitution(text=pkg_path+"/config/hobot_stereonet_config.json")
    )
    model_file_launch_arg = DeclareLaunchArgument(
        "model_file", default_value=TextSubstitution(text=pkg_path+"/config/hobot_stereonet.hbm")
    )
    hobot_stereonet_node = Node(
        package='hobot_stereonet',
        executable='hobot_stereonet',
        output='screen',
        parameters=[
            {"pkg_path": pkg_path},
            {"config_file": LaunchConfiguration('config_file')},
            {"model_file": LaunchConfiguration('model_file')}
        ],
        arguments=['--ros-args', '--log-level', 'warn']
    )

    # 启动渲染
    # hobot_stereonet_render_node = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory('hobot_stereonet_render'),
    #             'launch/hobot_stereonet_render.launch.py'))
    # )
    hobot_stereonet_render_node = Node(
        package='hobot_stereonet_render',
        executable='talker',
        output='screen',
        arguments=['--ros-args', '--log-level', 'warn']
    )

    # 启动WEB展示
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_topic': '/image_jpeg',
            'websocket_only_show_image': 'True'
        }.items()
    )

    return LaunchDescription([
        stereo_usb_cam_node,
        hobot_stereonet_render_node,
        web_node,
        config_file_launch_arg,
        model_file_launch_arg,
        hobot_stereonet_node
    ])
