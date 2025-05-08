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
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    stereonet_model_file_path = os.path.join(
        get_package_share_directory("hobot_stereonet"),
        "config",
        "x5baseplus_alldata_woIsaac.bin",
    )

    # 零拷贝环境配置
    shared_mem_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_shm"), "launch/hobot_shm.launch.py"
            )
        )
    )

    # 创建组件容器（关键步骤）
    container = ComposableNodeContainer(
        name="components_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        output="screen",
        arguments=["--ros-args", "--log-level", "warn"],
    )

    # mipi相机节点
    mipi_cam_component = ComposableNode(
        package="mipi_cam",
        plugin="mipi_cam::MipiCamNode",
        name="mipi_cam_component",
        parameters=[
            {"mipi_frame_ts_type": "sensor"},
            {"frame_id": "pcl_link"},
            {"device_mode": "dual"},
            {"dual_combine": 1},
            {"image_width": 640},
            {"image_height": 352},
            {"lpwm_enable": True},
            {"channel": 2},
            {"channel2": 0},
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    # 双目节点
    stereonet_component = ComposableNode(
        package="hobot_stereonet",
        plugin="stereonet::StereoNetNode",
        name="stereonet_component",
        parameters=[
            {"stereo_image_topic": "/image_combine_raw"},
            {"camera_info_topic": "/image_right_raw/camera_info"},
            {"need_rectify": False},
            {"stereonet_model_file_path": stereonet_model_file_path},
            {"postprocess": "v2"},
            {"render_type": 1},
            {"render_need_filter": False},
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    # 编码节点
    codec_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_codec"),
                "launch/hobot_codec_encode.launch.py",
            )
        ),
        launch_arguments={
            "codec_in_mode": "ros",
            "codec_out_mode": "ros",
            # 左图和深度拼接后的图
            "codec_sub_topic": "/stereonet_component/stereonet_visual",
            "codec_in_format": "bgr8",
            "codec_pub_topic": "/image_jpeg",
            "codec_out_format": "jpeg",
            "log_level": "warn",
        }.items(),
    )

    # web展示节点
    web_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("websocket"), "launch/websocket.launch.py"
            )
        ),
        launch_arguments={
            "websocket_image_topic": "/image_jpeg",
            "websocket_only_show_image": "true",
        }.items(),
    )

    return LaunchDescription(
        [
            shared_mem_node,
            container,
            LoadComposableNodes(
                target_container="components_container",
                composable_node_descriptions=[
                    mipi_cam_component,
                    stereonet_component,
                ],
            ),
            codec_node,
            web_node,
        ]
    )
