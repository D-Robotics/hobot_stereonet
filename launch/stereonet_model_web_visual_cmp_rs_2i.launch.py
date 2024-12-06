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

    rs_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('realsense2_camera'),
                                                   'launch/rs_launch.py')),
        launch_arguments={'rgb_camera.color_profile': '640x480x15',
                          'depth_module.depth_profile': '640x480x15',
                          'enable_sync': 'True',
                          'align_depth.enable': 'True',
                          'serial_no': '"215122252596"',
                          }.items()
    )
    #node_list.append(rs_cam)


    zed_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('hobot_zed_cam'),
                                                   'launch/pub_stereo_imgs_nv12_noweb.launch.py')),
        launch_arguments={'need_rectify': 'True',
                          'user_rectify': 'False'
                          }.items()
    )
    node_list.append(zed_cam)


    # 双目深度估计模型
    stereonet_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('hobot_stereonet'),
                                                   'launch/stereonet_model.launch.py')),
        launch_arguments={'stereo_image_topic': '/image_combine_raw',
                          'stereo_combine_mode': '1',
                          'log_level': 'info',
                          "camera_fx": "514.5878861678406",
                          "base_line": "0.119893",
                          'need_rectify': 'False'
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
