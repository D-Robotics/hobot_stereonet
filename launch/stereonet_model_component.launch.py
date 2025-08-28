# Copyright (c) 2025，D-Robotics.
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
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import LoadComposableNodes

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default_value'], description=param['description']) for param in parameters]

def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])

def generate_launch_description():

    stereo_calib_file_path =  os.path.join(
        get_package_share_directory('hobot_stereonet'),
        'config',
        'stereo.yaml'
    )

    stereonet_model_file_path =  os.path.join(
        get_package_share_directory('hobot_stereonet'),
        'config',
        'DStereoV2.0.bin'
    )

    local_image_path =  os.path.join(
        get_package_share_directory('hobot_stereonet'),
        'config'
    )

    node_params = [
        {'name':'log_level', 'default_value':'info', 'description': 'log_level'},

        {'name':'stereo_node_name', 'default_value':'StereoNetNode', 'description': 'stereo_node_name'},

        {'name':'stereonet_model_file_path', 'default_value': stereonet_model_file_path, 'description': 'stereonet_model_file_path'},

        {'name':'stereo_image_topic', 'default_value':'/image_combine_raw', 'description': 'stereo_image_topic'},
        {'name':'camera_info_topic', 'default_value':'/image_right_raw/camera_info', 'description': 'camera_info_topic'},

        {'name':'depth_image_topic', 'default_value':'/StereoNetNode/stereonet_depth', 'description': 'depth_topic'},
        {'name':'depth_camera_info_topic', 'default_value':'/StereoNetNode/stereonet_depth/camera_info', 'description': 'depth_camera_info_topic'},
        {'name':'pointcloud2_topic', 'default_value':'/StereoNetNode/stereonet_pointcloud2', 'description': 'pointcloud2_topic'},
        {'name':'rectify_left_image_topic', 'default_value':'/StereoNetNode/rectify_left_image', 'description': 'rectify_left_image_topic'},
        {'name':'rectify_right_image_topic', 'default_value':'/StereoNetNode/rectify_right_image', 'description': 'rectify_right_image_topic'},
        {'name':'rectify_bgr', 'default_value':'False', 'description': 'rectify_bgr'},
        {'name':'visual_image_topic', 'default_value':'/StereoNetNode/stereonet_visual', 'description': 'visual_topic'},

        {'name':'visual_alpha', 'default_value':'3', 'description': 'visual_alpha'},
        {'name':'visual_beta', 'default_value':'0', 'description': 'visual_beta'},
        {'name':'render_type', 'default_value':'0', 'description': 'render_type: 0-render disp, 1-render disp auto'},
        {'name':'render_perf', 'default_value':'True', 'description': 'render_perf'},

        {'name':'pointcloud_height_min', 'default_value':'-5.0', 'description': 'pointcloud_height_min'},
        {'name':'pointcloud_height_max', 'default_value':'5.0', 'description': 'pointcloud_height_max'},
        {'name':'pointcloud_depth_max', 'default_value':'5.0', 'description': 'pointcloud_depth_max'},

        {'name':'calib_method', 'default_value':'gdc', 'description': '[gdc none custom]'},
        {'name':'stereo_calib_file_path', 'default_value': stereo_calib_file_path, 'description': 'stereo_calib_file_path'},

        {'name':'camera_cx', 'default_value':'659.710', 'description': 'rectified_camera_cx'},
        {'name':'camera_cy', 'default_value':'360.584', 'description': 'rectified_camera_cy'},
        {'name':'camera_fx', 'default_value':'527.1931', 'description': 'rectified_camera_fx'},
        {'name':'camera_fy', 'default_value':'527.1931', 'description': 'rectified_camera_fy'},
        {'name':'baseline', 'default_value':'0.119893', 'description': 'baseline of stereo'},

        {'name':'uncertainty_th', 'default_value':'-0.09', 'description': 'uncertainty_th'},

        {'name':'save_image_flag', 'default_value':'False', 'description': 'save_image_flag'},
        {'name':'save_dir', 'default_value':'./stereonet_images', 'description': 'save_dir'},
        {'name':'save_freq', 'default_value':'1', 'description': 'save_freq'},
        {'name':'save_total', 'default_value':'-1', 'description': 'save_total'},

        {'name':'postprocess', 'default_value':'convex_upsampling', 'description': '[convex_upsampling convex_upsampling_with_uncert convex_upsampling_with_interp convex_upsampling_with_interp_uncert]'},

        {'name':'use_local_image', 'default_value':'False', 'description': 'use_local_image'},

        {'name':'local_image_path', 'default_value': local_image_path, 'description': 'local_image_path'},

        {'name':'speckle_filter_enable', 'default_value':'False', 'description': 'speckle_filter_enable'},
        {'name':'pcl_filter_enable', 'default_value':'False', 'description': 'pcl_filter_enable'},
        {'name':'leaf_size', 'default_value':'0.05', 'description': 'leaf_size'},
        {'name':'stdv', 'default_value':'0.01', 'description': 'stdv'},
        {'name':'KMean', 'default_value':'10', 'description': 'KMean'},

        {'name':'infer_thread_num', 'default_value':'2', 'description': 'infer_thread_num'},
        {'name':'resize_before_rectify', 'default_value':'False', 'description': 'resize_before_rectify'},
        {'name':'load_rectify_param', 'default_value':'False', 'description': 'load rectify param whether need_rectify or not'},
    ]

    launch = declare_configurable_parameters(node_params)
    launch.append(
        LoadComposableNodes(
            target_container=LaunchConfiguration("target_container_name"),
            composable_node_descriptions=[
                ComposableNode(
                    package="hobot_stereonet",
                    namespace='',
                    plugin="stereonet::StereoNetNode",
                    name="StereoNetNode",
                    parameters=[set_configurable_parameters(node_params)],
                    extra_arguments=[{"use_intra_process_comms": True}],
                )
            ]
        )
    )
    return LaunchDescription(launch)
