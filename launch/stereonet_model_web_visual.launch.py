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
        'use_local_image',
        default_value='False',
        description='use_local_image'
    ))

    dual_mipi_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('mipi_cam'),
                                                   'launch/mipi_cam_dual_channel.launch.py')),
        launch_arguments={'mipi_image_width': '1280',
                          'mipi_image_height': '640',
                          'mipi_frame_ts_type': 'realtime',
                          'frame_id': 'default_cam',
                          'log_level': 'warn'
                          }.items(),
        condition=UnlessCondition(LaunchConfiguration('use_local_image'))
    )
    node_list.append(dual_mipi_cam)

    # 双目深度估计模型
    stereonet_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('stereonet_model'),
                                                   'launch/stereonet_model.launch.py')),
        launch_arguments={'stereo_image_topic': '/image_combine_raw',
                          'stereo_combine_mode': '1',
                          'log_level': 'info'
                          }.items()
    )
    node_list.append(stereonet_node)

    # 编码节点
    codec_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('hobot_codec'),
                'launch/hobot_codec_encode.launch.py')),
        launch_arguments={
            'codec_in_mode': 'ros',
            'codec_out_mode': 'ros',
            # 左图和深度拼接后的图
            'codec_sub_topic': '/StereoNetNode/stereonet_visual',
            'codec_in_format': 'bgr8',
            'codec_pub_topic': '/image_jpeg',
            'codec_out_format': 'jpeg',
            'log_level': 'warn'
        }.items()
    )
    node_list.append(codec_node)

    # web展示节点
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('websocket'),
                'launch/websocket.launch.py')),
        launch_arguments={
            'websocket_image_topic': '/image_jpeg',
            'websocket_only_show_image': 'true',
            # 'websocket_smart_topic': '/detect_depth_result'
        }.items()
    )
    node_list.append(web_node)

    return LaunchDescription(node_list)