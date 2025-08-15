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

    node_list = []

    zed_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('hobot_zed_cam'),
                                                   'launch/pub_stereo_imgs_noweb.launch.py')),
        launch_arguments={'need_rectify': 'True',
                          'user_rectify': 'False',
                          'resolution': '720p',
                          'dst_width': '640',
                          'dst_height': '352',
                          'camera_info_topic': '/image_combine_raw/camera_info',
                          }.items()
    )
    node_list.append(zed_cam)

    stereonet_model_file_path =  os.path.join(
        get_package_share_directory('hobot_stereonet'),
        'config',
        'DStereoV2.2.bin'
    )

    # 双目深度估计模型
    stereonet_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('hobot_stereonet'),
                                                   'launch/stereonet_model_web_visual.launch.py')),
        launch_arguments = {"stereonet_model_file_path": stereonet_model_file_path,
                            "postprocess": "v2.2",
                            'need_rectify': 'False'
                          }.items()
    )
    node_list.append(stereonet_node)

    return LaunchDescription(node_list)
