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
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_prefix
from launch.substitutions import TextSubstitution
import os

def generate_launch_description():
    stereonet_pub_web_arg = DeclareLaunchArgument(
        "stereonet_pub_web",
        default_value="True",
        description="stereonet_pub_web, if not, we will disable websocket and codec of stereonet depth",
    )

    target_container_name_arg = DeclareLaunchArgument(
        "target_container_name",
        default_value="stereonet_components_container",
        description="The name of the target container to load the component into.",
    )

    # 零拷贝环境配置
    shared_mem_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_shm"), "launch/hobot_shm.launch.py"
            )
        )
    )

    config_file_path = os.path.join(
        get_package_prefix('mipi_cam'),
        "lib/mipi_cam/config/")
    print("config_file_path is ", config_file_path)

    # 创建组件容器（关键步骤）
    container = ComposableNodeContainer(
        name="stereonet_components_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        output="screen",
        arguments=["--ros-args", "--log-level", "warn"],
        condition=IfCondition(
            PythonExpression(
                [
                    '"',
                    LaunchConfiguration("target_container_name"),
                    '" == "stereonet_components_container"',
                ]
            )
        ),
    )

    # mipi相机节点
    mipi_cam_component = ComposableNode(
        package="mipi_cam",
        plugin="mipi_cam::MipiCamNode",
        name="mipi_cam_component",
        parameters=[
            {"config_path": LaunchConfiguration('mipi_config_path')},
            {"camera_calibration_file_path": LaunchConfiguration(
                'mipi_camera_calibration_file_path')},
            {"out_format": LaunchConfiguration('mipi_out_format')},
            {"image_width": LaunchConfiguration('mipi_image_width')},
            {"image_height": LaunchConfiguration('mipi_image_height')},
            {"framerate": LaunchConfiguration('mipi_image_framerate')},
            {"io_method": LaunchConfiguration('mipi_io_method')},
            {"video_device": LaunchConfiguration('mipi_video_device')},
            {"device_mode": LaunchConfiguration('device_mode')},
            {"gdc_bin_file": LaunchConfiguration('mipi_gdc_bin_file')},
            {"dual_combine": LaunchConfiguration('dual_combine')},
            {"lpwm_enable": LaunchConfiguration('mipi_lpwm_enable')},
            {"channel": LaunchConfiguration('mipi_channel')},
            {"channel2": LaunchConfiguration('mipi_channel2')},
            {"frame_ts_type": LaunchConfiguration('mipi_frame_ts_type')},
            {"rotation": LaunchConfiguration('mipi_rotation')},
            {"gdc_enable": LaunchConfiguration('mipi_gdc_enable')},
            {"frame_id": LaunchConfiguration('frame_id')},
        ],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    # 双目节点
    stereonet_model_component = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("hobot_stereonet"),
                "launch/stereonet_model_component.launch.py",
            )
        ),
        launch_arguments={
            "target_container": LaunchConfiguration("target_container_name"),
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
        condition=IfCondition(LaunchConfiguration("stereonet_pub_web")),
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
        condition=IfCondition(LaunchConfiguration("stereonet_pub_web")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mipi_config_path", 
                default_value=TextSubstitution(text=str(config_file_path)),
                description='mipi camera calibration file path'),
            DeclareLaunchArgument(
                'mipi_camera_calibration_file_path',
                default_value=TextSubstitution(text=str(config_file_path)+"calib_params.yaml"),
                description='mipi camera calibration file path'),
            DeclareLaunchArgument(
                'mipi_out_format',
                default_value='nv12',
                description='mipi camera out format'),
            DeclareLaunchArgument(
                'mipi_image_width',
                default_value='1280',
                description='mipi camera out image width'),
            DeclareLaunchArgument(
                'mipi_image_height',
                default_value='720',
                description='mipi camera out image height'),
            DeclareLaunchArgument(
                'mipi_image_framerate',
                default_value='10.0',
                description='mipi camera out image framerate'),
            DeclareLaunchArgument(
                'mipi_io_method',
                default_value='ros',
                description='mipi camera out io_method'),
            DeclareLaunchArgument(
                'mipi_video_device',
                default_value='default',
                description='mipi camera device'),
            DeclareLaunchArgument(
                'device_mode',
                default_value='dual',
                description='mipi camera device mode single or dual'),
            DeclareLaunchArgument(
                'dual_combine',
                default_value='1',
                description='dual mode output channel'),
            DeclareLaunchArgument(
                'mipi_channel',
                default_value='2',
                description='mipi camera host channel'),
            DeclareLaunchArgument(
                'mipi_channel2',
                default_value='0',
                description='mipi dual camera right channel'),
            DeclareLaunchArgument(
                'mipi_frame_ts_type',
                default_value='sensor',
                description='type(sensor/realtime) of timestamp for publishing messages'),
            DeclareLaunchArgument(
                'frame_id',
                default_value='default_cam',
                description=''),
            DeclareLaunchArgument(
                'mipi_gdc_bin_file',
                default_value='',
                description='mipi camera gdc bin_file'),
            DeclareLaunchArgument(
                'mipi_lpwm_enable',
                default_value='False',
                description='mipi dual camera lpwm enable'),
            DeclareLaunchArgument(
                'mipi_rotation',
                default_value='0.0',
                description='mipi camera out image rotation'),
            DeclareLaunchArgument(
                'mipi_gdc_enable',
                default_value='True',
                description='mipi camera gdc enable'),
            DeclareLaunchArgument(
                'log_level',
                default_value='warn',
                description='log level'),

            DeclareLaunchArgument(
                "mipi_image_width",
                default_value="640",
                description="mipi camera out image width",
            ),
            DeclareLaunchArgument(
                "mipi_image_height",
                default_value="352",
                description="mipi camera out image height",
            ),
            DeclareLaunchArgument(
                "mipi_image_framerate",
                default_value="10.0",
                description="mipi camera out image framerate",
            ),
            DeclareLaunchArgument(
                "mipi_lpwm_enable",
                default_value="False",
                description="mipi dual camera lpwm enable",
            ),
            DeclareLaunchArgument(
                "mipi_rotation",
                default_value="0.0",
                description="mipi camera out image rotation",
            ),
            DeclareLaunchArgument(
                "mipi_gdc_enable",
                default_value="True",
                description="mipi camera gdc enable",
            ),
            stereonet_pub_web_arg,
            target_container_name_arg,
            shared_mem_node,
            container,
            LoadComposableNodes(
                target_container=LaunchConfiguration("target_container_name"),
                composable_node_descriptions=[
                    mipi_cam_component,
                ],
            ),
            stereonet_model_component,
            codec_node,
            web_node,
        ]
    )
