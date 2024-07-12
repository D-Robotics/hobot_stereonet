# Copyright (c) 2022，Horizon Robotics.
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


from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
import os
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    model_file_path =  os.path.join(
        get_package_share_directory('stereonet_model'),
        'config',
        'model.hbm'
    )
    local_image_path =  os.path.join(
        get_package_share_directory('stereonet_model'),
        'config'
    )
    stereo_image_topic = DeclareLaunchArgument(
        'stereo_image_topic',
        default_value='/image_combine_raw',
        description='stereo_image_topic')

    camera_cx = DeclareLaunchArgument(
        'camera_cx',
        default_value='640',
        description='camera_cx')

    camera_cy = DeclareLaunchArgument(
        'camera_cy',
        default_value='320.0',
        description='camera_cy')

    camera_fx = DeclareLaunchArgument(
        'camera_fx',
        default_value='300.0',
        description='camera_fx')

    camera_fy = DeclareLaunchArgument(
        'camera_fy',
        default_value='300.0',
        description='camera_fy')

    base_line = DeclareLaunchArgument(
        'base_line',
        default_value='0.06',
        description='base_line')

    depth_min = DeclareLaunchArgument(
        'depth_min',
        default_value='0.03',
        description='depth_min')

    depth_max = DeclareLaunchArgument(
        'depth_max',
        default_value='5.0',
        description='depth_max')

    use_local_image = DeclareLaunchArgument(
        'use_local_image',
        default_value='False',
        description='use_local_image')

    use_usb_camera = DeclareLaunchArgument(
        'use_usb_camera',
        default_value='False',
        description='use_usb_camera')

    stereo_combine_mode = DeclareLaunchArgument(
        'stereo_combine_mode',
        default_value='1',
        description='stereo_combine_mode')

    return LaunchDescription([
        camera_cx,
        camera_fx,
        camera_cy,
        camera_fy,
        base_line,
        stereo_image_topic,
        depth_max,
        depth_min,
        use_local_image,
        use_usb_camera,
        stereo_combine_mode,
        # 启动图片发布pkg
        Node(
            package='stereonet_model',
            executable='stereonet_model_node',
            output='screen',
            parameters=[
                {"use_usb_camera": LaunchConfiguration('use_usb_camera') },
                {"use_local_image": LaunchConfiguration('use_local_image') },
                {"camera_cx": LaunchConfiguration('camera_cx') },
                {"camera_fx": LaunchConfiguration('camera_fx') },
                {"camera_cy": LaunchConfiguration('camera_cy')},
                {"camera_fy": LaunchConfiguration('camera_fy') },
                {"base_line": LaunchConfiguration('base_line') },
                {"depth_max": LaunchConfiguration('depth_max') },
                {"depth_min": LaunchConfiguration('depth_min') },
                {"stereo_combine_mode": LaunchConfiguration('stereo_combine_mode') },
                {"stereonet_model_file_path": model_file_path},
                {"local_image_path": local_image_path},
                {"stereo_image_topic": LaunchConfiguration('stereo_image_topic')}
            ],
            arguments=['--ros-args', '--log-level', 'info']
        )
    ])
