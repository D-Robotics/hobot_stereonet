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
        'stereonet_pub_web',
        default_value='True',
        description='stereonet_pub_web, if not, we will disable websocket and codec of stereonet depth'
    ))
    node_list.append(DeclareLaunchArgument(
        'use_mipi_cam',
        default_value='True',
        description='use_mipi_cam'
    ))
    node_list.append(DeclareLaunchArgument(
        'codec_sub_topic',
        default_value='/StereoNetNode/stereonet_visual',
        description='codec_sub_topic'
    ))
    node_list.append(DeclareLaunchArgument(
        'codec_in_format',
        default_value='bgr8',
        description='codec_in_format'
    ))
    node_list.append(DeclareLaunchArgument(
        'codec_pub_topic',
        default_value='/image_jpeg',
        description='codec_pub_topic'
    ))
    node_list.append(DeclareLaunchArgument(
        'websocket_image_topic',
        default_value='/image_jpeg',
        description='websocket_image_topic'
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
            # "frame_id": "pcl_link",
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

    # codec node
    codec_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_encode.launch.py')),
        launch_arguments={
            'codec_in_mode': 'ros',
            'codec_out_mode': 'ros',
            'codec_sub_topic': LaunchConfiguration('codec_sub_topic'),
            'codec_in_format': LaunchConfiguration('codec_in_format'),
            'codec_pub_topic': LaunchConfiguration('codec_pub_topic'),
            'codec_out_format': 'jpeg',
            'log_level': 'warn'
        }.items(),
        condition=IfCondition(LaunchConfiguration('stereonet_pub_web'))
    )
    node_list.append(codec_node)

    # web node
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_topic': LaunchConfiguration('websocket_image_topic'),
            'websocket_only_show_image': 'true',
        }.items(),
        condition=IfCondition(LaunchConfiguration('stereonet_pub_web'))
    )
    node_list.append(web_node)

    return LaunchDescription(node_list)
