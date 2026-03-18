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


from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
import os
from launch.substitutions import LaunchConfiguration

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default_value'], description=param['description']) for param in parameters]

def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])

def generate_launch_description():

    stereonet_model_file_path =  os.path.join(
        get_package_share_directory('hobot_stereonet'),
        'config',
        'DStereoV2.0.bin'
    )

    local_image_dir =  os.path.join(
        get_package_share_directory('hobot_stereonet'),
        'config'
    )

    node_params = [
        {'name':'log_level', 'default_value':'info', 'description': 'log_level'},

        {'name':'stereo_node_name', 'default_value':'StereoNetNode', 'description': 'stereo_node_name'},

        {'name':'stereonet_model_file_path', 'default_value': stereonet_model_file_path, 'description': 'stereonet_model_file_path'},

        {'name':'stereo_image_topic', 'default_value':'/image_combine_raw', 'description': 'stereo_image_topic'},
        {'name':'camera_info_topic', 'default_value':'/image_combine_raw/right/camera_info', 'description': 'camera_info_topic'},
        {'name':'left_camera_info_topic', 'default_value':'/image_combine_raw/left/camera_info', 'description': 'left_camera_info_topic'},

        {'name':'depth_image_topic', 'default_value':'/StereoNetNode/stereonet_depth', 'description': 'depth_topic'},
        {'name':'depth_camera_info_topic', 'default_value':'/StereoNetNode/stereonet_depth/camera_info', 'description': 'depth_camera_info_topic'},
        {'name':'rectify_left_camera_info_topic', 'default_value':'/StereoNetNode/rectify_left_image/camera_info', 'description': 'rectify_left_camera_info_topic'},
        {'name':'rectify_right_camera_info_topic', 'default_value':'/StereoNetNode/rectify_right_image/camera_info', 'description': 'rectify_right_camera_info_topic'},
        {'name':'pointcloud2_topic', 'default_value':'/StereoNetNode/stereonet_pointcloud2', 'description': 'pointcloud2_topic'},
        {'name':'publish_pcd_enabled', 'default_value':'True', 'description': 'publish_pcd_enabled'},
        {'name':'rectify_left_image_topic', 'default_value':'/StereoNetNode/rectify_left_image', 'description': 'rectify_left_image_topic'},
        {'name':'rectify_right_image_topic', 'default_value':'/StereoNetNode/rectify_right_image', 'description': 'rectify_right_image_topic'},
        {'name':'publish_rectify_bgr', 'default_value':'False', 'description': 'publish_rectify_bgr'},
        {'name':'origin_left_image_topic', 'default_value':'/StereoNetNode/origin_left_image', 'description': 'origin_left_image_topic'},
        {'name':'origin_right_image_topic', 'default_value':'/StereoNetNode/origin_right_image', 'description': 'origin_right_image_topic'},
        {'name':'publish_origin_enable', 'default_value':'True', 'description': 'publish_origin_enable'},
        {'name':'visual_image_topic', 'default_value':'/StereoNetNode/stereonet_visual', 'description': 'visual_topic'},
        {'name':'publish_visual_enabled', 'default_value':'True', 'description': 'publish_visual_enabled'},
        {'name':'stereonet_frame_id', 'default_value':'camera_link', 'description': 'stereonet_frame_id'},
        {'name':'stereonet_frame_id_right', 'default_value':'camera_link_right', 'description': 'stereonet_frame_id_right'},

        {'name':'render_type', 'default_value':'indoor', 'description': 'render_type: [indoor, outdoor, indoor-reverse, outdoor-reverse, distance, distance-reverse]'},
        {'name':'render_perf', 'default_value':'True', 'description': 'render_perf'},
        {'name':'depth_decimal_num', 'default_value':'2', 'description': 'depth_decimal_num'},
        {'name':'render_max_disp', 'default_value':'80', 'description': 'render_max_disp'},
        {'name':'render_z_near', 'default_value':'-1.0', 'description': 'render_z_near'},
        {'name':'render_z_range', 'default_value':'3.0', 'description': 'render_z_range'},

        {'name':'pointcloud_downsample_step', 'default_value':'2', 'description': 'pointcloud_downsample_step'},
        {'name':'pointcloud_height_min', 'default_value':'-5.0', 'description': 'pointcloud_height_min'},
        {'name':'pointcloud_height_max', 'default_value':'5.0', 'description': 'pointcloud_height_max'},
        {'name':'pointcloud_depth_max', 'default_value':'5.0', 'description': 'pointcloud_depth_max'},

        {'name':'calib_method', 'default_value':'none', 'description': '[none custom]'},
        {'name':'stereo_calib_file_path', 'default_value': '', 'description': 'stereo_calib_file_path'},

        {'name':'camera_cx', 'default_value':'0.0', 'description': 'rectified_camera_cx'},
        {'name':'camera_cy', 'default_value':'0.0', 'description': 'rectified_camera_cy'},
        {'name':'camera_fx', 'default_value':'0.0', 'description': 'rectified_camera_fx'},
        {'name':'camera_fy', 'default_value':'0.0', 'description': 'rectified_camera_fy'},
        {'name':'baseline', 'default_value':'0.0', 'description': 'baseline of stereo'},
        {'name':'doffs', 'default_value':'0.0', 'description': 'doffs of stereo'},

        {'name':'uncertainty_th', 'default_value':'-0.10', 'description': 'uncertainty_th'},

        {'name':'save_result_flag', 'default_value':'False', 'description': 'save_result_flag'},
        {'name':'save_dir', 'default_value':'./stereonet_result', 'description': 'save_dir'},
        {'name':'save_freq', 'default_value':'1', 'description': 'save_freq'},
        {'name':'save_total', 'default_value':'-1', 'description': 'save_total'},
        {'name':'save_stereo_flag', 'default_value':'True', 'description': 'save_stereo_flag'},
        {'name':'save_origin_flag', 'default_value':'False', 'description': 'save_origin_flag'},
        {'name':'save_disp_flag', 'default_value':'True', 'description': 'save_disp_flag'},
        {'name':'save_uncert_flag', 'default_value':'False', 'description': 'save_uncert_flag'},
        {'name':'save_depth_flag', 'default_value':'True', 'description': 'save_depth_flag'},
        {'name':'save_visual_flag', 'default_value':'True', 'description': 'save_visual_flag'},
        {'name':'save_pcd_flag', 'default_value':'False', 'description': 'save_pcd_flag'},

        {'name':'use_local_image_flag', 'default_value':'False', 'description': 'use_local_image_flag'},
        {'name':'local_image_dir', 'default_value': local_image_dir, 'description': 'local_image_dir'},
        {'name':'image_sleep', 'default_value': "0", 'description': 'image_sleep ms'},

        {'name':'speckle_filter_enable', 'default_value':'False', 'description': 'speckle_filter_enable'},
        {'name':'max_speckle_size', 'default_value':'100', 'description': 'max_speckle_size'},
        {'name':'max_disp_diff', 'default_value':'1.0', 'description': 'max_speckle_size'},
        {'name':'pcl_filter_enable', 'default_value':'False', 'description': 'pcl_filter_enable'},
        # {'name':'voxel_leaf_size', 'default_value':'0.05', 'description': 'voxel_leaf_size'},
        # {'name':'mean_k', 'default_value':'10', 'description': 'mean_k'},
        # {'name':'std_thresh', 'default_value':'1.0', 'description': 'std_dev_mul_thresh'},
        {'name':'grid_size', 'default_value':'0.10', 'description': 'grid_size'},
        {'name':'grid_min_point_count', 'default_value':'5', 'description': 'grid_min_point_count'},

        {'name':'left_img_mask_enable', 'default_value':'False', 'description': 'left_img_mask_enable'},
        {'name':'measure_mode', 'default_value':'False', 'description': 'measure_mode'},
        {'name':'roi_size', 'default_value':'10', 'description': 'roi_size'},
        {'name':'gt_depth', 'default_value':'0.0', 'description': 'gt_depth'},

        {'name':'epipolar_mode', 'default_value':'False', 'description': 'epipolar_mode'},
        {'name':'epipolar_img', 'default_value':'rect', 'description': 'epipolar_img'},
        {'name':'chessboard_per_rows', 'default_value':'20', 'description': 'chessboard_per_rows'},
        {'name':'chessboard_per_cols', 'default_value':'11', 'description': 'chessboard_per_cols'},
        {'name':'chessboard_square_size', 'default_value':'0.06', 'description': 'chessboard_square_size'},
        {'name':'feature_epipolar_mode', 'default_value':'False', 'description': 'feature_epipolar_mode'},

        {'name':'infer_thread_num', 'default_value':'2', 'description': 'infer_thread_num'},
        {'name':'save_thread_num', 'default_value':'4', 'description': 'save_thread_num'},
        {'name':'max_save_task', 'default_value':'50', 'description': 'max_save_task'},

        {'name':'post_version', 'default_value':'auto', 'description': 'post_version'},
    ]

    launch = declare_configurable_parameters(node_params)
    launch.append(Node(
        package='hobot_stereonet',
        executable='stereonet_model_node',
        name=LaunchConfiguration('stereo_node_name'),
        output='screen',
        parameters=[set_configurable_parameters(node_params)],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    ))

    return LaunchDescription(launch)
