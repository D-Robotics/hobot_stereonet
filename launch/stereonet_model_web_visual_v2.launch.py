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
    stereo_calib_file_path = os.path.join(
        get_package_share_directory("hobot_stereonet"), "config", "stereo.yaml"
    )

    stereonet_model_file_path =  os.path.join(
        get_package_share_directory('hobot_stereonet'),
        'config',
        'x5baseplus_alldata_woIsaac_yuv444.bin'
    )

    stereo_calib_path_arg = DeclareLaunchArgument(
        "stereo_calib_path",
        default_value=stereo_calib_file_path,
        description="Description of my_param",
    )

    visual_alpha_arg = DeclareLaunchArgument(
        "visual_alpha", default_value="4", description="visual alpha"
    )

    visual_beta_arg = DeclareLaunchArgument(
        "visual_beta", default_value="0", description="visual beta"
    )

    # 零拷贝环境配置
    shared_mem_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_shm"), "launch/hobot_shm.launch.py"
            )
        )
    )

    # mipi双目相机
    dual_mipi_cam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("mipi_cam"),
                "launch/mipi_cam_dual_channel.launch.py",
            )
        ),
        launch_arguments={
            "mipi_frame_ts_type": "sensor",
            "frame_id": "pcl_link",
            "log_level": "warn",
        }.items(),
    )

    # 双目深度估计模型
    stereonet_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_stereonet"),
                "launch/stereonet_model.launch.py",
            )
        ),
        launch_arguments={
            "stereo_image_topic": "/image_combine_raw",
            "stereo_calib_file_path": LaunchConfiguration("stereo_calib_path"),
            "alpha": LaunchConfiguration("visual_alpha"),
            "beta": LaunchConfiguration("visual_beta"),
            "stereonet_model_file_path": stereonet_model_file_path,
            "postprocess": "v2",
            "stereo_combine_mode": "1",
        }.items(),
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
            "codec_sub_topic": "/StereoNetNode/stereonet_visual",
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
            # 'websocket_smart_topic': '/detect_depth_result'
        }.items(),
    )

    return LaunchDescription(
        [
            stereo_calib_path_arg,
            visual_alpha_arg,
            visual_beta_arg,
            shared_mem_node,
            dual_mipi_cam,
            stereonet_node,
            codec_node,
            web_node,
        ]
    )
