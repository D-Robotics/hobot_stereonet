// Copyright (c) 2025，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "stereonet_component.h"
namespace stereonet {
StereoNetNode::StereoNetNode(const rclcpp::NodeOptions &node_options, const std::string &node_name)
    : Node(node_name, node_options) {
  set_node_params();
  set_subscription_publisher();
  set_dnn_model();
  publish_static_tf();
  set_worker_threads();
}

StereoNetNode::~StereoNetNode() {
  for (auto &t : infer_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  if (publish_thread_.joinable()) {
    publish_thread_.join();
  }
  if (save_thread_pool_ptr_) {
    save_thread_pool_ptr_->wait();
    save_thread_pool_ptr_.reset();
  }
  RCLCPP_WARN_STREAM(this->get_logger(), "=> release " << this->get_name());
}

void StereoNetNode::set_node_params() {
  RCLCPP_WARN_STREAM(this->get_logger(),
                     "=> ===================== init " << this->get_name() << " =====================" << std::endl);
  this->declare_parameter<std::string>("stereonet_model_file_path", "");
  stereonet_model_file_path_ = this->get_parameter("stereonet_model_file_path").as_string();
  if (stereonet_model_file_path_.empty() || !fs::exists(stereonet_model_file_path_)) {
    RCLCPP_ERROR(this->get_logger(), "=> stereonet_model_file_path: [%s] not exist, please set it",
                 stereonet_model_file_path_.c_str());
    rclcpp::shutdown();
  }

  this->declare_parameter<std::string>("stereo_image_topic", "/image_combine_raw");
  stereo_image_topic_ = this->get_parameter("stereo_image_topic").as_string();
  this->declare_parameter<std::string>("camera_info_topic", "/image_combine_raw/right/camera_info");
  camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
  this->declare_parameter<std::string>("left_camera_info_topic", "/image_combine_raw/left/camera_info");
  left_camera_info_topic_ = this->get_parameter("left_camera_info_topic").as_string();

  this->declare_parameter<std::string>("depth_image_topic", "~/stereonet_depth");
  depth_image_topic_ = this->get_parameter("depth_image_topic").as_string();
  this->declare_parameter<std::string>("depth_camera_info_topic", "~/stereonet_depth/camera_info");
  depth_camera_info_topic_ = this->get_parameter("depth_camera_info_topic").as_string();
  this->declare_parameter<std::string>("rectify_left_camera_info_topic", "~/rectify_left_image/camera_info");
  rectify_left_camera_info_topic_ = this->get_parameter("rectify_left_camera_info_topic").as_string();
  this->declare_parameter<std::string>("rectify_right_camera_info_topic", "~/rectify_right_image/camera_info");
  rectify_right_camera_info_topic_ = this->get_parameter("rectify_right_camera_info_topic").as_string();
  this->declare_parameter<std::string>("rectify_left_image_topic", "~/rectify_left_image");
  rectify_left_image_topic_ = this->get_parameter("rectify_left_image_topic").as_string();
  this->declare_parameter<std::string>("rectify_right_image_topic", "~/rectify_right_image");
  rectify_right_image_topic_ = this->get_parameter("rectify_right_image_topic").as_string();
  this->declare_parameter<bool>("publish_rectify_bgr", false);
  publish_rectify_bgr_ = this->get_parameter("publish_rectify_bgr").as_bool();
  this->declare_parameter<std::string>("origin_left_image_topic", "~/origin_left_image");
  origin_left_image_topic_ = this->get_parameter("origin_left_image_topic").as_string();
  this->declare_parameter<std::string>("origin_right_image_topic", "~/origin_right_image");
  origin_right_image_topic_ = this->get_parameter("origin_right_image_topic").as_string();
  this->declare_parameter<bool>("publish_origin_enable", true);
  publish_origin_enable_ = this->get_parameter("publish_origin_enable").as_bool();
  this->declare_parameter<std::string>("pointcloud2_topic", "~/stereonet_pointcloud2");
  pointcloud2_topic_ = this->get_parameter("pointcloud2_topic").as_string();
  this->declare_parameter<bool>("publish_pcd_enabled", true);
  publish_pcd_enabled_ = this->get_parameter("publish_pcd_enabled").as_bool();
  this->declare_parameter<std::string>("visual_image_topic", "~/stereonet_visual");
  visual_image_topic_ = this->get_parameter("visual_image_topic").as_string();
  this->declare_parameter<bool>("publish_visual_enabled", true);
  publish_visual_enabled_ = this->get_parameter("publish_visual_enabled").as_bool();
  this->declare_parameter<std::string>("stereonet_frame_id", "camera_link");
  stereonet_frame_id_ = this->get_parameter("stereonet_frame_id").as_string();
  this->declare_parameter<std::string>("stereonet_frame_id_right", "camera_link_right");
  stereonet_frame_id_right_ = this->get_parameter("stereonet_frame_id_right").as_string();
  this->declare_parameter<bool>("render_perf", true);
  render_perf_ = this->get_parameter("render_perf").as_bool();
  if (render_perf_) {
    performance_writer::Get();
  }

  this->declare_parameter<double>("uncertainty_th", 0.0);
  uncertainty_th_ = this->get_parameter("uncertainty_th").as_double();

  camera_intrinsic_ = std::make_shared<CameraIntrinsic>();
  this->declare_parameter<double>("camera_fx", 0.0);
  this->declare_parameter<double>("camera_fy", 0.0);
  this->declare_parameter<double>("camera_cx", 0.0);
  this->declare_parameter<double>("camera_cy", 0.0);
  this->declare_parameter<double>("baseline", 0.0);
  this->declare_parameter<double>("doffs", 0.0);
  camera_intrinsic_->fx = this->get_parameter("camera_fx").as_double();
  camera_intrinsic_->fy = this->get_parameter("camera_fy").as_double();
  camera_intrinsic_->cx = this->get_parameter("camera_cx").as_double();
  camera_intrinsic_->cy = this->get_parameter("camera_cy").as_double();
  camera_intrinsic_->baseline = this->get_parameter("baseline").as_double();
  camera_intrinsic_->doffs = this->get_parameter("doffs").as_double();
  orignal_camera_intrinsic_ = std::make_shared<CameraIntrinsic>(*camera_intrinsic_);

  this->declare_parameter<int>("pointcloud_downsample_step", 2);
  this->declare_parameter<double>("pointcloud_height_min", -5.0);
  this->declare_parameter<double>("pointcloud_height_max", 5.0);
  this->declare_parameter<double>("pointcloud_depth_max", 5.0);
  pointcloud_downsample_step_ = this->get_parameter("pointcloud_downsample_step").as_int();
  pointcloud_height_min_ = this->get_parameter("pointcloud_height_min").as_double();
  pointcloud_height_max_ = this->get_parameter("pointcloud_height_max").as_double();
  pointcloud_depth_max_ = this->get_parameter("pointcloud_depth_max").as_double();
  if (pointcloud_downsample_step_ <= 0) pointcloud_downsample_step_ = 2;

  this->declare_parameter<bool>("save_result_flag", "false");
  save_result_flag_ = this->get_parameter("save_result_flag").as_bool();
  this->declare_parameter<std::string>("save_dir", "./stereonet_result");
  save_dir_ = this->get_parameter("save_dir").as_string();
  this->declare_parameter<int>("save_freq", 1);
  save_freq_ = this->get_parameter("save_freq").as_int();
  this->declare_parameter<int>("save_total", -1);
  save_total_ = this->get_parameter("save_total").as_int();

  this->declare_parameter<int>("infer_thread_num", 2);
  infer_thread_num_ = this->get_parameter("infer_thread_num").as_int();
  this->declare_parameter<int>("save_thread_num", 4);
  save_thread_num_ = this->get_parameter("save_thread_num").as_int();
  this->declare_parameter<int>("max_save_task", 50);
  max_save_task_ = this->get_parameter("max_save_task").as_int();
  if (infer_thread_num_ <= 0) infer_thread_num_ = 1;
  if (infer_thread_num_ > 4) infer_thread_num_ = 4;
  if (save_thread_num_ <= 0) save_thread_num_ = 1;
  if (save_thread_num_ > 8) save_thread_num_ = 8;
  if (max_save_task_ <= 20) max_save_task_ = 20;
  this->declare_parameter<bool>("save_stereo_flag", true);
  save_stereo_flag_ = this->get_parameter("save_stereo_flag").as_bool();
  this->declare_parameter<bool>("save_origin_flag", false);
  save_origin_flag_ = this->get_parameter("save_origin_flag").as_bool();
  this->declare_parameter<bool>("save_disp_flag", true);
  save_disp_flag_ = this->get_parameter("save_disp_flag").as_bool();
  this->declare_parameter<bool>("save_depth_flag", true);
  save_depth_flag_ = this->get_parameter("save_depth_flag").as_bool();
  this->declare_parameter<bool>("save_uncert_flag", false);
  save_uncert_flag_ = this->get_parameter("save_uncert_flag").as_bool();
  this->declare_parameter<bool>("save_visual_flag", true);
  save_visual_flag_ = this->get_parameter("save_visual_flag").as_bool();
  this->declare_parameter<bool>("save_pcd_flag", false);
  save_pcd_flag_ = this->get_parameter("save_pcd_flag").as_bool();

  this->declare_parameter<std::string>("calib_method", "none");
  calib_method_ = this->get_parameter("calib_method").as_string();
  auto is_valid_calib_method = [](const std::string &calib_method) {
    return calib_method == "none" || calib_method == "custom";
  };
  if (!is_valid_calib_method(calib_method_)) {
    RCLCPP_ERROR(this->get_logger(),
                 "\033[32m=> calib_method parameter invalid, should be one of [none, custom]\033[0m");
    rclcpp::shutdown();
  }
  this->declare_parameter<std::string>("stereo_calib_file_path", "");
  stereo_calib_file_path_ = this->get_parameter("stereo_calib_file_path").as_string();
  if (calib_method_ == "custom") {
    if (stereo_calib_file_path_.empty() || !fs::exists(stereo_calib_file_path_)) {
      RCLCPP_ERROR(this->get_logger(),
                   "=> stereo_calib_file_path: [%s] not exist, please set it when calib_method is custom",
                   stereo_calib_file_path_.c_str());
      rclcpp::shutdown();
    } else {
      stereo_rectifier_ = std::make_shared<StereoRectify>(stereo_calib_file_path_, this->get_logger());
    }
  }

  this->declare_parameter<bool>("use_local_image_flag", false);
  use_local_image_flag_ = this->get_parameter("use_local_image_flag").as_bool();
  this->declare_parameter<std::string>("local_image_dir", "./offline_image");
  local_image_dir_ = this->get_parameter("local_image_dir").as_string();
  this->declare_parameter<int>("image_sleep", 0);
  image_sleep_ = this->get_parameter("image_sleep").as_int();
  if (use_local_image_flag_) {
    if (!fs::exists(local_image_dir_)) {
      RCLCPP_ERROR(this->get_logger(), "\033[31m=> local_image_dir: %s not exist\033[0m", local_image_dir_.c_str());
      rclcpp::shutdown();
    }
    if (save_result_flag_) {
      if (local_image_dir_ == save_dir_) {
        RCLCPP_ERROR(this->get_logger(),
                     "\033[31m=> local_image_dir: %s and save_dir: %s conflict, please set them differen\033[0m",
                     local_image_dir_.c_str(), save_dir_.c_str());
        rclcpp::shutdown();
      }
      // when use local image, always save all results
      save_freq_ = 1;
      save_total_ = -1;
      save_thread_num_ = 1;

      save_stereo_flag_ = true;
      save_origin_flag_ = false;
      save_disp_flag_ = true;
      save_depth_flag_ = true;
      save_visual_flag_ = true;
      save_pcd_flag_ = true;
    }
    if (image_sleep_ < 0) image_sleep_ = 0;
    infer_thread_num_ = 1;
    render_perf_ = false;
  }

  this->declare_parameter<bool>("speckle_filter_enable", false);
  speckle_filter_enable_ = this->get_parameter("speckle_filter_enable").as_bool();
  this->declare_parameter<int>("max_speckle_size", 100);
  max_speckle_size_ = this->get_parameter("max_speckle_size").as_int();
  this->declare_parameter<double>("max_disp_diff", 1.0);
  max_disp_diff_ = this->get_parameter("max_disp_diff").as_double();

  this->declare_parameter<bool>("pcl_filter_enable", false);
  pcl_filter_enable_ = this->get_parameter("pcl_filter_enable").as_bool();
  // this->declare_parameter<float>("voxel_leaf_size", 0.05f);
  // voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
  // this->declare_parameter<int>("mean_k", 10);
  // mean_k_ = this->get_parameter("mean_k").as_int();
  // this->declare_parameter<double>("std_thresh", 1.0);
  // std_thresh_ = this->get_parameter("std_thresh").as_double();
  this->declare_parameter<float>("grid_size", 0.05f);
  grid_size_ = this->get_parameter("grid_size").as_double();
  this->declare_parameter<int>("grid_min_point_count", 5);
  grid_min_point_count_ = this->get_parameter("grid_min_point_count").as_int();
  if (grid_size_ <= 0.0f) grid_size_ = 0.05f;
  if (grid_min_point_count_ <= 0) grid_min_point_count_ = 5;

  this->declare_parameter<std::string>("render_type", "indoor");
  render_type_ = this->get_parameter("render_type").as_string();
  auto is_valid_render_type = [](const std::string &render_type) {
    return render_type == "indoor" || render_type == "outdoor" || render_type == "indoor-reverse" ||
           render_type == "outdoor-reverse" || render_type == "distance" || render_type == "distance-reverse";
  };
  if (!is_valid_render_type(render_type_)) {
    RCLCPP_ERROR(this->get_logger(), "\033[31m=> render_type parameter invalid, should be one of [indoor, outdoor, "
                                     "indoor-reverse, outdoor-reverse, distance, distance-reverse]\033[0m");
    rclcpp::shutdown();
  }

  this->declare_parameter<bool>("left_img_mask_enable", false);
  left_img_mask_enable_ = this->get_parameter("left_img_mask_enable").as_bool();

  this->declare_parameter<bool>("measure_mode", false);
  measure_mode_ = this->get_parameter("measure_mode").as_bool();
  this->declare_parameter<int>("roi_size", 10);
  roi_size_ = this->get_parameter("roi_size").as_int();
  if (roi_size_ <= 0) roi_size_ = 10;
  this->declare_parameter<double>("gt_depth", 0.0);
  gt_depth_ = this->get_parameter("gt_depth").as_double();

  this->declare_parameter<int>("depth_decimal_num", 2);
  depth_decimal_num_ = this->get_parameter("depth_decimal_num").as_int();
  if (depth_decimal_num_ < 2) depth_decimal_num_ = 2; // cm
  if (depth_decimal_num_ > 3) depth_decimal_num_ = 3; // mm

  this->declare_parameter<int>("render_max_disp", 192);
  render_max_disp_ = this->get_parameter("render_max_disp").as_int();
  if (render_max_disp_ < 0) render_max_disp_ = 192;

  this->declare_parameter<double>("render_z_near", -1.0);
  render_z_near_ = this->get_parameter("render_z_near").as_double();
  this->declare_parameter<double>("render_z_range", 3.0);
  render_z_range_ = this->get_parameter("render_z_range").as_double();
  if (render_z_range_ < 0.0) render_z_range_ = 3.0;

  this->declare_parameter<bool>("epipolar_mode", false);
  epipolar_mode_ = this->get_parameter("epipolar_mode").as_bool();
  this->declare_parameter<int>("chessboard_per_rows", 20);
  chessboard_per_rows_ = this->get_parameter("chessboard_per_rows").as_int();
  this->declare_parameter<int>("chessboard_per_cols", 11);
  chessboard_per_cols_ = this->get_parameter("chessboard_per_cols").as_int();
  this->declare_parameter<double>("chessboard_square_size", 0.06);
  chessboard_square_size_ = this->get_parameter("chessboard_square_size").as_double();
  this->declare_parameter<std::string>("epipolar_img", "rect");
  epipolar_img_ = this->get_parameter("epipolar_img").as_string();
  if (epipolar_img_ != "rect" && epipolar_img_ != "origin") epipolar_img_ = "rect";
  this->declare_parameter<bool>("feature_epipolar_mode", false);
  feature_epipolar_mode_ = this->get_parameter("feature_epipolar_mode").as_bool();

  this->declare_parameter<std::string>("post_version", "auto");
  post_version_ = this->get_parameter("post_version").as_string();

  this->declare_parameter<bool>("ground_angle_enable", false);
  this->declare_parameter<int>("ground_roi_center_x", -1);
  this->declare_parameter<int>("ground_roi_center_y", -1);
  this->declare_parameter<int>("ground_roi_width", 80);
  this->declare_parameter<int>("ground_roi_height", 40);
  this->declare_parameter<int>("ground_roi_min_valid_points", 100);
  ground_angle_enable_ = this->get_parameter("ground_angle_enable").as_bool();
  ground_roi_center_x_ = this->get_parameter("ground_roi_center_x").as_int();
  ground_roi_center_y_ = this->get_parameter("ground_roi_center_y").as_int();
  ground_roi_width_ = this->get_parameter("ground_roi_width").as_int();
  ground_roi_height_ = this->get_parameter("ground_roi_height").as_int();
  ground_roi_min_valid_points_ = this->get_parameter("ground_roi_min_valid_points").as_int();
  if (ground_roi_width_ <= 0) ground_roi_width_ = 160;
  if (ground_roi_height_ <= 0) ground_roi_height_ = 80;
  if (ground_roi_min_valid_points_ <= 0) ground_roi_min_valid_points_ = 100;

  RCLCPP_WARN_STREAM(
      this->get_logger(),
      std::endl
          << "stereonet_model_file_path: " << stereonet_model_file_path_ << std::endl
          << "stereo_image_topic: " << stereo_image_topic_ << std::endl
          << "camera_info_topic: " << camera_info_topic_ << std::endl
          << "depth_image_topic: " << depth_image_topic_ << std::endl
          << "depth_camera_info_topic: " << depth_camera_info_topic_ << std::endl
          << "rectify_left_image_topic: " << rectify_left_image_topic_ << std::endl
          << "rectify_right_image_topic: " << rectify_right_image_topic_ << std::endl
          << "publish_rectify_bgr: " << publish_rectify_bgr_ << std::endl
          << "[origin_left_image_topic, origin_right_image_topic, publish_origin_enable]: [" << origin_left_image_topic_
          << ", " << origin_right_image_topic_ << ", " << publish_origin_enable_ << "]" << std::endl
          << "[pointcloud2_topic, publish_pcd_enabled]: [" << pointcloud2_topic_ << ", " << publish_pcd_enabled_ << "]"
          << std::endl
          << "[visual_image_topic, publish_visual_enabled]: [" << visual_image_topic_ << ", " << publish_visual_enabled_
          << "]" << std::endl
          << "[stereonet_frame_id, stereonet_frame_id_right]: [" << stereonet_frame_id_ << ", "
          << stereonet_frame_id_right_ << "]" << std::endl
          << "uncertainty_th: " << uncertainty_th_ << std::endl
          << "[camera_fx, camera_fy, camera_cx, camera_cy, baseline, doffs]: [" << camera_intrinsic_->fx << ", "
          << camera_intrinsic_->fy << ", " << camera_intrinsic_->cx << ", " << camera_intrinsic_->cy << ", "
          << camera_intrinsic_->baseline << "(m)" << camera_intrinsic_->doffs << "]" << std::endl
          << "[pointcloud_downsample_step, pointcloud_height_min, pointcloud_height_max, pointcloud_depth_max]: ["
          << pointcloud_downsample_step_ << ", " << pointcloud_height_min_ << "(m), " << pointcloud_height_max_
          << "(m), " << pointcloud_depth_max_ << "(m)]" << std::endl
          << "[use_local_image_flag, local_image_dir, image_sleep]: [" << use_local_image_flag_ << ", "
          << local_image_dir_ << ", " << image_sleep_ << "]" << std::endl
          << "[save_result_flag, save_dir, save_freq, save_total, save_stereo_flag, save_origin_flag, save_disp_flag, "
             "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
          << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_ << ", " << save_total_ << ", "
          << save_stereo_flag_ << ", " << save_origin_flag_ << ", " << save_disp_flag_ << ", " << save_uncert_flag_
          << ", " << save_depth_flag_ << ", " << save_visual_flag_ << ", " << save_pcd_flag_ << "]" << std::endl
          << "[calib_method, stereo_calib_file_path]: [" << calib_method_ << ", " << stereo_calib_file_path_ << "]"
          << std::endl
          << "[speckle_filter_enable, max_speckle_size, max_disp_diff]: [" << speckle_filter_enable_ << ", "
          << max_speckle_size_ << ", " << max_disp_diff_ << "]" << std::endl
          << "[pcl_filter_enable, grid_size, grid_min_point_count]: [" << pcl_filter_enable_ << ", " << grid_size_
          << ", " << grid_min_point_count_ << "]" << std::endl
          << "[render_type, render_perf, depth_decimal_num, render_max_disp, render_z_near, render_z_range]: ["
          << render_type_ << ", " << render_perf_ << ", " << depth_decimal_num_ << ", " << render_max_disp_ << ", "
          << render_z_near_ << "(m), " << render_z_range_ << "(m)]" << std::endl
          << "left_img_mask_enable: " << left_img_mask_enable_ << std::endl
          << "[measure_mode, roi_size, gt_depth]: [" << measure_mode_ << ", " << roi_size_ << ", " << gt_depth_
          << "(mm)]" << std::endl
          << "[epipolar_mode, feature_epipolar_mode, epipolar_img, chessboard_per_rows, chessboard_per_cols, "
             "chessboard_square_size]: ["
          << epipolar_mode_ << ", " << feature_epipolar_mode_ << ", " << epipolar_img_ << ", " << chessboard_per_rows_
          << ", " << chessboard_per_cols_ << ", " << chessboard_square_size_ << "(m)]" << std::endl
          << "feature_epipolar_mode: " << feature_epipolar_mode_ << std::endl
          << "[ground_angle_enable, ground_roi_center_x, ground_roi_center_y, ground_roi_width, ground_roi_height, "
             "ground_roi_min_valid_points]: ["
          << ground_angle_enable_ << ", " << ground_roi_center_x_ << ", " << ground_roi_center_y_ << ", "
          << ground_roi_width_ << ", " << ground_roi_height_ << ", " << ground_roi_min_valid_points_ << "]" << std::endl
          << "post_version: " << post_version_ << std::endl
          << "[infer_thread_num, save_thread_num, max_save_task]: [" << infer_thread_num_ << ", " << save_thread_num_
          << ", " << max_save_task_ << "]" << std::endl
          << std::endl
          << "=> ==================================================================" << std::endl);

  if (save_result_flag_) {
    if (!fs::exists(save_dir_)) {
      if (fs::create_directories(save_dir_)) {
        RCLCPP_WARN(this->get_logger(), "\033[32m=> create save_dir: %s\033[0m", save_dir_.c_str());
      } else {
        RCLCPP_ERROR(this->get_logger(), "\033[31m=> create save_dir: %s failed\033[0m", save_dir_.c_str());
        rclcpp::shutdown();
      }
    }
    if (save_freq_ <= 0) save_freq_ = 1;
    render_perf_ = false;
  }

  this->declare_parameter<bool>("save_result_once", false);
  param_cb_handle_ = this->add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter> &params)
                                                              -> rcl_interfaces::msg::SetParametersResult {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto &param : params) {
      if (param.get_name() == "save_result_once") {
        save_result_once_ = param.as_bool();
        if (save_result_once_) {
          if (!fs::exists(save_dir_)) {
            if (fs::create_directories(save_dir_)) {
              RCLCPP_WARN(this->get_logger(), "\033[32m=> create save_dir: %s\033[0m", save_dir_.c_str());
            } else {
              RCLCPP_ERROR(this->get_logger(), "\033[31m=> create save_dir: %s failed\033[0m", save_dir_.c_str());
            }
          }
          if (save_total_ < 0)
            save_total_ = 1;
          else
            save_total_ += 1;
        }
      }
      if (param.get_name() == "save_result_flag") {
        save_result_flag_ = param.as_bool();
        if (save_result_flag_) {
          if (!fs::exists(save_dir_)) {
            if (fs::create_directories(save_dir_)) {
              RCLCPP_WARN(this->get_logger(), "\033[32m=> create save_dir: %s\033[0m", save_dir_.c_str());
            } else {
              RCLCPP_ERROR(this->get_logger(), "\033[31m=> create save_dir: %s failed\033[0m", save_dir_.c_str());
            }
          }
        }
      }
      if (param.get_name() == "save_dir") {
        save_dir_ = param.as_string();
        RCLCPP_WARN(this->get_logger(), "\033[32m=> save_dir: %s\033[0m", save_dir_.c_str());
      }
      if (param.get_name() == "save_freq") {
        save_freq_ = param.as_int();
        if (save_freq_ <= 0) save_freq_ = 1;
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> [save_result_flag, save_dir, save_freq, save_total, "
                                               "save_stereo_flag, save_origin_flag, save_disp_flag, "
                                               "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
                                                   << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_
                                                   << ", " << save_total_ << ", " << save_stereo_flag_ << ", "
                                                   << save_origin_flag_ << ", " << save_disp_flag_ << ", "
                                                   << save_uncert_flag_ << ", " << save_depth_flag_ << ", "
                                                   << save_visual_flag_ << ", " << save_pcd_flag_ << "]\033[0m");
      }
      if (param.get_name() == "save_total") {
        auto value = param.as_int();
        if (value > 0) {
          if (save_total_ < 0)
            save_total_ = value;
          else
            save_total_ += value;
        } else {
          save_total_ = -1;
        }
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> [save_result_flag, save_dir, save_freq, save_total, "
                                               "save_stereo_flag, save_origin_flag, save_disp_flag, "
                                               "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
                                                   << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_
                                                   << ", " << save_total_ << ", " << save_stereo_flag_ << ", "
                                                   << save_origin_flag_ << ", " << save_disp_flag_ << ", "
                                                   << save_uncert_flag_ << ", " << save_depth_flag_ << ", "
                                                   << save_visual_flag_ << ", " << save_pcd_flag_ << "]\033[0m");
      }
      if (param.get_name() == "save_stereo_flag") {
        save_stereo_flag_ = param.as_bool();
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> [save_result_flag, save_dir, save_freq, save_total, "
                                               "save_stereo_flag, save_origin_flag, save_disp_flag, "
                                               "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
                                                   << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_
                                                   << ", " << save_total_ << ", " << save_stereo_flag_ << ", "
                                                   << save_origin_flag_ << ", " << save_disp_flag_ << ", "
                                                   << save_uncert_flag_ << ", " << save_depth_flag_ << ", "
                                                   << save_visual_flag_ << ", " << save_pcd_flag_ << "]\033[0m");
      }
      if (param.get_name() == "save_origin_flag") {
        save_origin_flag_ = param.as_bool();
        if (!publish_origin_enable_) save_origin_flag_ = false;
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> [save_result_flag, save_dir, save_freq, save_total, "
                                               "save_stereo_flag, save_origin_flag, save_disp_flag, "
                                               "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
                                                   << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_
                                                   << ", " << save_total_ << ", " << save_stereo_flag_ << ", "
                                                   << save_origin_flag_ << ", " << save_disp_flag_ << ", "
                                                   << save_uncert_flag_ << ", " << save_depth_flag_ << ", "
                                                   << save_visual_flag_ << ", " << save_pcd_flag_ << "]\033[0m");
      }
      if (param.get_name() == "save_disp_flag") {
        save_disp_flag_ = param.as_bool();
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> [save_result_flag, save_dir, save_freq, save_total, "
                                               "save_stereo_flag, save_origin_flag, save_disp_flag, "
                                               "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
                                                   << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_
                                                   << ", " << save_total_ << ", " << save_stereo_flag_ << ", "
                                                   << save_origin_flag_ << ", " << save_disp_flag_ << ", "
                                                   << save_uncert_flag_ << ", " << save_depth_flag_ << ", "
                                                   << save_visual_flag_ << ", " << save_pcd_flag_ << "]\033[0m");
      }
      if (param.get_name() == "save_uncert_flag") {
        save_uncert_flag_ = param.as_bool();
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> [save_result_flag, save_dir, save_freq, save_total, "
                                               "save_stereo_flag, save_origin_flag, save_disp_flag, "
                                               "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
                                                   << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_
                                                   << ", " << save_total_ << ", " << save_stereo_flag_ << ", "
                                                   << save_origin_flag_ << ", " << save_disp_flag_ << ", "
                                                   << save_uncert_flag_ << ", " << save_depth_flag_ << ", "
                                                   << save_visual_flag_ << ", " << save_pcd_flag_ << "]\033[0m");
      }
      if (param.get_name() == "save_depth_flag") {
        save_depth_flag_ = param.as_bool();
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> [save_result_flag, save_dir, save_freq, save_total, "
                                               "save_stereo_flag, save_origin_flag, save_disp_flag, "
                                               "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
                                                   << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_
                                                   << ", " << save_total_ << ", " << save_stereo_flag_ << ", "
                                                   << save_origin_flag_ << ", " << save_disp_flag_ << ", "
                                                   << save_uncert_flag_ << ", " << save_depth_flag_ << ", "
                                                   << save_visual_flag_ << ", " << save_pcd_flag_ << "]\033[0m");
      }
      if (param.get_name() == "save_visual_flag") {
        save_visual_flag_ = param.as_bool();
        if (!publish_visual_enabled_) save_visual_flag_ = false;
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> [save_result_flag, save_dir, save_freq, save_total, "
                                               "save_stereo_flag, save_origin_flag, save_disp_flag, "
                                               "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
                                                   << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_
                                                   << ", " << save_total_ << ", " << save_stereo_flag_ << ", "
                                                   << save_origin_flag_ << ", " << save_disp_flag_ << ", "
                                                   << save_uncert_flag_ << ", " << save_depth_flag_ << ", "
                                                   << save_visual_flag_ << ", " << save_pcd_flag_ << "]\033[0m");
      }
      if (param.get_name() == "save_pcd_flag") {
        save_pcd_flag_ = param.as_bool();
        if (!publish_pcd_enabled_) save_pcd_flag_ = false;
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> [save_result_flag, save_dir, save_freq, save_total, "
                                               "save_stereo_flag, save_origin_flag, save_disp_flag, "
                                               "save_uncert_flag, save_depth_flag, save_visual_flag, save_pcd_flag]: ["
                                                   << save_result_flag_ << ", " << save_dir_ << ", " << save_freq_
                                                   << ", " << save_total_ << ", " << save_stereo_flag_ << ", "
                                                   << save_origin_flag_ << ", " << save_disp_flag_ << ", "
                                                   << save_uncert_flag_ << ", " << save_depth_flag_ << ", "
                                                   << save_visual_flag_ << ", " << save_pcd_flag_ << "]\033[0m");
      }
      if (param.get_name() == "max_save_task") {
        max_save_task_ = param.as_int();
        if (max_save_task_ <= 20) max_save_task_ = 20;
        RCLCPP_WARN_STREAM(this->get_logger(), "\033[32m=> max_save_task: " << max_save_task_ << "\033[0m");
      }
    }
    return result;
  });
}

void StereoNetNode::set_subscription_publisher() {
  // Set up subscriptions
  if (use_local_image_flag_) {
    infer_offline_timer_ = this->create_wall_timer(std::chrono::milliseconds(0), [this]() {
      this->infer_offline();
      infer_offline_timer_->cancel();
    });
  } else {
    stereo_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        stereo_image_topic_, 1, std::bind(&StereoNetNode::stereo_image_callback, this, std::placeholders::_1));
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, 10, std::bind(&StereoNetNode::camera_info_callback, this, std::placeholders::_1));
    left_camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        left_camera_info_topic_, 10, std::bind(&StereoNetNode::left_camera_info_callback, this, std::placeholders::_1));
  }

  // Set up publishers
  if (publish_visual_enabled_) {
    visual_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(visual_image_topic_, 10);
  }
  depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(depth_image_topic_, 10);
  depth_camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(depth_camera_info_topic_, 10);
  rectify_left_camera_info_pub_ =
      this->create_publisher<sensor_msgs::msg::CameraInfo>(rectify_left_camera_info_topic_, 10);
  rectify_right_camera_info_pub_ =
      this->create_publisher<sensor_msgs::msg::CameraInfo>(rectify_right_camera_info_topic_, 10);
  if (publish_pcd_enabled_) {
    pointcloud2_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(pointcloud2_topic_, 10);
  }
  rectify_left_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(rectify_left_image_topic_, 10);
  rectify_right_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(rectify_right_image_topic_, 10);
  if (publish_origin_enable_) {
    origin_left_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(origin_left_image_topic_, 10);
    origin_right_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(origin_right_image_topic_, 10);
  }

  // Set monitor timer
  monitor_timer_ = this->create_wall_timer(std::chrono::milliseconds(5000), [this]() {
    if (!camera_intrinsic_->is_valid()) {
      RCLCPP_ERROR(this->get_logger(), "\033[31m=> Haven't received any camera info from topic %s\033[0m",
                   camera_info_topic_.c_str());
    } else {
      monitor_timer_->cancel();
    }
  });
}

void StereoNetNode::set_dnn_model() {
  stereonet_process_ = std::make_shared<StereonetProcess>(this->get_logger());
  int ret_code = stereonet_process_->init(stereonet_model_file_path_, post_version_);
  if (ret_code != 0) {
    RCLCPP_ERROR(this->get_logger(), "=> StereonetProcess init failed");
    rclcpp::shutdown();
  }
}

void StereoNetNode::set_worker_threads() {
  for (int i = 0; i < infer_thread_num_; ++i) {
    infer_threads_.emplace_back(&StereoNetNode::infer_function, this, i);
  }
  publish_thread_ = std::thread(&StereoNetNode::publish_function, this);
  save_thread_pool_ptr_ = std::make_unique<BS::thread_pool<>>(save_thread_num_);
}

void StereoNetNode::publish_static_tf() {
  static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
  geometry_msgs::msg::TransformStamped t;
  t.header.stamp = now();
  t.header.frame_id = stereonet_frame_id_;
  t.child_frame_id = "camera_optical_frame";

  t.transform.translation.x = 0.0;
  t.transform.translation.y = 0.0;
  t.transform.translation.z = 0.0;

  tf2::Quaternion q;
  q.setRPY(-M_PI / 2, 0, -M_PI / 2);
  q.normalize();

  t.transform.rotation.x = q.x();
  t.transform.rotation.y = q.y();
  t.transform.rotation.z = q.z();
  t.transform.rotation.w = q.w();

  static_broadcaster_->sendTransform(t);
}

void StereoNetNode::stereo_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  auto now = this->get_clock()->now();
  auto latency = (now - msg->header.stamp).seconds() * 1000;
  RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                       "=> receive stereo image, format: %s, stamp: %u.%u, latency: %.2f ms", msg->encoding.c_str(),
                       msg->header.stamp.sec, msg->header.stamp.nanosec, latency);
  if (global_frame_cnt_ < 5) global_frame_cnt_++;

  if (calib_method_ == "custom" && camera_info_updated_ == false) {
    int model_input_w = 0, model_input_h = 0;
    stereonet_process_->get_model_input_size(model_input_w, model_input_h);
    int single_img_w = msg->width;
    int single_img_h = msg->height / 2;
    stereo_rectifier_->build_undistmap(single_img_w, single_img_h, model_input_w, model_input_h);
    stereo_rectifier_->get_intrinsic(camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx,
                                     camera_intrinsic_->cy, camera_intrinsic_->baseline);
    camera_info_updated_ = true;
  }

  // ================================== Check camera_info ===========================
  if (!camera_intrinsic_->is_valid()) return;
  if (infer_thread_num_ > 1) {
    // ================================== Enqueue =====================================
    while (input_image_queue_.size_approx() >= 1) {
      sensor_msgs::msg::Image::SharedPtr drop;
      input_image_queue_.try_dequeue(drop);
    }
    input_image_queue_.enqueue(msg);
  } else {
    // ================================== Preprocess ==================================
    int model_input_w = 0, model_input_h = 0;
    stereonet_process_->get_model_input_size(model_input_w, model_input_h);
    std::vector<uint8_t> rectify_left_img_data, rectify_right_img_data;
    {
      ScopeProcessTime t(this->get_logger(), "preprocess");
      preprocess(msg, model_input_w, model_input_h, rectify_left_img_data, rectify_right_img_data);
      // ================================== Enqueue =====================================
      if (pre_process_queue_.size_approx() >= 1) {
        std::shared_ptr<PreProcessData> drop;
        pre_process_queue_.try_dequeue(drop);
      }
      pre_process_queue_.enqueue(std::make_shared<PreProcessData>(msg, rectify_left_img_data, rectify_right_img_data));
    }

    // ================================== Calc FOV ====================================
    if (!calc_fov_flag_) {
      float HFOV = 2 * atan(model_input_w / (2 * camera_intrinsic_->fx)) * 180 / M_PI;
      float VFOV = 2 * atan(model_input_h / (2 * camera_intrinsic_->fy)) * 180 / M_PI;
      RCLCPP_WARN_STREAM(this->get_logger(), "\033[31m=> HFOV: " << HFOV << "°, VFOV: " << VFOV << "°\033[0m");
      calc_fov_flag_ = true;
    }
  }
}

void StereoNetNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  origin_camera_info_ = msg;

  int model_input_w = 0, model_input_h = 0;
  double scale_w = 1.0, scale_h = 1.0;
  stereonet_process_->get_model_input_size(model_input_w, model_input_h);
  if (model_input_w != msg->width || model_input_h != msg->height) {
    scale_w = model_input_w / static_cast<double>(msg->width);
    scale_h = model_input_h / static_cast<double>(msg->height);
  }

  // only subscribe once
  camera_intrinsic_->fx = msg->p[0] * scale_w;
  camera_intrinsic_->fy = msg->p[5] * scale_h;
  camera_intrinsic_->cx = msg->p[2] * scale_w;
  camera_intrinsic_->cy = msg->p[6] * scale_h;
  camera_intrinsic_->baseline = std::abs(msg->p[3] / msg->p[0]);
  camera_intrinsic_->doffs = msg->p[11];
  camera_info_updated_ = true;

  if (camera_intrinsic_->baseline > 1) camera_intrinsic_->baseline *= 0.001f; // convert mm to m
  orignal_camera_intrinsic_ = std::make_shared<CameraIntrinsic>(*camera_intrinsic_);

  RCLCPP_WARN(this->get_logger(),
              "\033[31m=> sub rectified [fx, fy, cx, cy, baseline(m), doffs] : [%f, %f, %f, %f, %f, %f]\033[0m",
              camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy,
              camera_intrinsic_->baseline, camera_intrinsic_->doffs);
  // only subscribe once
  camera_info_sub_.reset();
}

void StereoNetNode::left_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  origin_left_camera_info_ = msg;
  RCLCPP_WARN(this->get_logger(), "=> receive left camera info");
  // only subscribe once
  left_camera_info_sub_.reset();
}

void StereoNetNode::infer_function(const int &thread_id) {
  while (rclcpp::ok()) {
    if (infer_thread_num_ > 1 || use_local_image_flag_) {
      sensor_msgs::msg::Image::SharedPtr stereo_msg;
      if (input_image_queue_.wait_dequeue_timed(stereo_msg, std::chrono::milliseconds(100))) {
        // Process the image message
        // RCLCPP_INFO(this->get_logger(), "Thread %d: Processing image with timestamp %u.%u", thread_id,
        //             stereo_msg->header.stamp.sec, stereo_msg->header.stamp.nanosec);

        // ================================== Preprocess ==================================
        int model_input_w = 0, model_input_h = 0;
        stereonet_process_->get_model_input_size(model_input_w, model_input_h);
        std::vector<uint8_t> rectify_left_img_data, rectify_right_img_data;
        {
          ScopeProcessTime t(this->get_logger(), "preprocess");
          preprocess(stereo_msg, model_input_w, model_input_h, rectify_left_img_data, rectify_right_img_data);
        }

        // ================================== Calc FOV ====================================
        if (!calc_fov_flag_) {
          float HFOV = 2 * atan(model_input_w / (2 * camera_intrinsic_->fx)) * 180 / M_PI;
          float VFOV = 2 * atan(model_input_h / (2 * camera_intrinsic_->fy)) * 180 / M_PI;
          RCLCPP_WARN_STREAM(this->get_logger(), "\033[31m=> HFOV: " << HFOV << "°, VFOV: " << VFOV << "°\033[0m");
          calc_fov_flag_ = true;
        }

        // ================================== Inference ==================================
        cv::Mat disp, uncert;
        stereonet_process_->forward_sync(rectify_left_img_data, rectify_right_img_data, uncertainty_th_, disp, uncert);
        cv::Mat depth;
        {
          ScopeProcessTime t(this->get_logger(), "disp_to_depth");
          if (camera_intrinsic_->is_valid()) {
            StereonetProcess::disp_to_depth(disp, depth, *camera_intrinsic_);
          } else {
            if (calib_method_ == "none") {
              RCLCPP_ERROR_ONCE(
                  this->get_logger(),
                  "\033[31m=> unable to receive topic %s to obtain camera intrinsic parameters, and the camera "
                  "intrinsic parameters [camera_fx, camera_fy, camera_cx, camera_cy, baseline] are not manually "
                  "set, please confirm whether the topic is correct or manually set the camera intrinsic "
                  "parameters. when calib_method is none\033[0m",
                  camera_info_topic_.c_str());
              rclcpp::shutdown();
            } else if (calib_method_ == "custom") {
              RCLCPP_ERROR_ONCE(
                  this->get_logger(),
                  "\033[31m=> calib_method is custom, camera intrinsic should be set from stereo_calib_file_path: "
                  "%s\033[0m",
                  stereo_calib_file_path_.c_str());
              rclcpp::shutdown();
            }
            continue;
          }
        }
        // ================================== Postprocess ================================
        if (speckle_filter_enable_) {
          ScopeProcessTime t(this->get_logger(), "speckle_filter");
          SpeckleFilter::filter(disp, 0.0f, 100, 1.0f);
          cv::Mat mask = (disp > 0);
          depth.setTo(0, ~mask);
        }

        if (left_img_mask_enable_) {
          ScopeProcessTime t(this->get_logger(), "left_img_mask");
          cv::Mat left_bgr;
          ImgConvertUtils::nv12_to_bgr_mat(rectify_left_img_data.data(), left_bgr, disp.cols, disp.rows);
          cv::Mat mask;
          cv::inRange(left_bgr, cv::Scalar(0, 0, 0), cv::Scalar(1, 1, 1), mask);
          cv::Mat labels, stats, centroids;
          int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
          cv::Mat filtered_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
          for (int i = 1; i < num_labels; i++) {
            int left = stats.at<int>(i, cv::CC_STAT_LEFT);
            int top = stats.at<int>(i, cv::CC_STAT_TOP);
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            int right = left + w;
            int bottom = top + h;
            bool touch_border = (left == 0) || (top == 0) || (right >= mask.cols) || (bottom >= mask.rows);
            if (touch_border) {
              filtered_mask.setTo(1, labels == i);
            }
          }
          disp.setTo(0, filtered_mask);
          depth.setTo(0, filtered_mask);
        }

        // ================================== Publish ====================================
        auto pub_data = std::make_shared<PubData>();
        pub_data->timestamp = static_cast<uint64_t>(stereo_msg->header.stamp.sec) * 1'000'000'000 +
                              static_cast<uint64_t>(stereo_msg->header.stamp.nanosec);
        pub_data->header = stereo_msg->header;
        pub_data->origin_stereo_msg = stereo_msg;
        pub_data->disp = disp;
        pub_data->uncert = uncert;
        pub_data->depth = depth;

        pub_data->rectify_left_img_data = rectify_left_img_data;
        pub_data->rectify_right_img_data = rectify_right_img_data;
        cv::Mat left_bgr, right_bgr;
        ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_left_img_data.data(), left_bgr, pub_data->disp.cols,
                                         pub_data->disp.rows);
        ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_right_img_data.data(), right_bgr, pub_data->disp.cols,
                                         pub_data->disp.rows);
        pub_data->left_bgr = left_bgr;
        pub_data->right_bgr = right_bgr;

        if (!use_local_image_flag_ && pub_data_queue_.size() >= infer_thread_num_) {
          // if not use local image, drop one message to avoid publish too many messages
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                               "\033[31m=> drop one message to avoid publish too many messages\033[0m");
          pub_data_queue_.pop_front();
        }
        while (use_local_image_flag_ && pub_data_queue_.size() >= 10) {
          // if use local image, wait for 100ms to avoid publish too many messages
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        pub_data_queue_.put(pub_data->timestamp, pub_data);
      }
    } else {
      // ================================== Inference ==================================
      std::shared_ptr<PreProcessData> pre_process_data;
      if (pre_process_queue_.wait_dequeue_timed(pre_process_data, std::chrono::milliseconds(100))) {
        stereonet_process_->forward_async(pre_process_data->rectify_left_img_data,
                                          pre_process_data->rectify_right_img_data, uncertainty_th_, camera_intrinsic_,
                                          pre_process_data->stereo_msg, pub_data_queue_);
      }
    }
  }
}

void StereoNetNode::preprocess(const sensor_msgs::msg::Image::SharedPtr &stereo_msg, const int &model_input_w,
                               const int &model_input_h, std::vector<uint8_t> &left_img_data,
                               std::vector<uint8_t> &right_img_data) {
  size_t model_input_nv12_size = model_input_w * model_input_h * 3 / 2;
  left_img_data.resize(model_input_nv12_size);
  right_img_data.resize(model_input_nv12_size);
  if (stereo_msg->encoding == "nv12") {
    int single_img_w = stereo_msg->width;
    int single_img_h = stereo_msg->height / 2;

    if (calib_method_ == "none") {
      if (single_img_w != model_input_w || single_img_h != model_input_h) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 5000,
            "\033[31m=> input image size not match model input size, need resize, [%d x %d] -> [%d x %d]\033[0m",
            single_img_w, single_img_h, model_input_w, model_input_h);
        cv::Mat stereo_bgr;
        ImgConvertUtils::nv12_to_bgr_mat(const_cast<uint8_t *>(stereo_msg->data.data()), stereo_bgr, stereo_msg->width,
                                         stereo_msg->height);
        cv::Mat left_bgr = stereo_bgr.rowRange(0, single_img_h).clone();
        cv::Mat right_bgr = stereo_bgr.rowRange(single_img_h, stereo_msg->height).clone();
        cv::resize(left_bgr, left_bgr, cv::Size(model_input_w, model_input_h));
        cv::resize(right_bgr, right_bgr, cv::Size(model_input_w, model_input_h));
        ImgConvertUtils::bgr_mat_to_nv12(left_bgr, left_img_data.data());
        ImgConvertUtils::bgr_mat_to_nv12(right_bgr, right_img_data.data());
        if (camera_info_updated_ == false) {
          camera_info_updated_ = true;
          camera_intrinsic_->cx = camera_intrinsic_->cx * model_input_w / single_img_w;
          camera_intrinsic_->cy = camera_intrinsic_->cy * model_input_h / single_img_h;
          camera_intrinsic_->fx = camera_intrinsic_->fx * model_input_w / single_img_w;
          camera_intrinsic_->fy = camera_intrinsic_->fy * model_input_h / single_img_h;
          RCLCPP_WARN(this->get_logger(),
                      "\033[31m=> after resize, update camera intrinsic [fx, fy, cx, cy, baseline(m), doffs] : [%f, "
                      "%f, %f, %f, %f, %f]\033[0m",
                      camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy,
                      camera_intrinsic_->baseline, camera_intrinsic_->doffs);
        }
      } else {
        std::memcpy(left_img_data.data(), stereo_msg->data.data(), single_img_w * single_img_h);
        std::memcpy(left_img_data.data() + single_img_w * single_img_h,
                    stereo_msg->data.data() + stereo_msg->width * stereo_msg->height, single_img_w * single_img_h / 2);
        std::memcpy(right_img_data.data(), stereo_msg->data.data() + single_img_w * single_img_h,
                    single_img_w * single_img_h);
        std::memcpy(right_img_data.data() + single_img_w * single_img_h,
                    stereo_msg->data.data() + stereo_msg->width * stereo_msg->height + single_img_w * single_img_h / 2,
                    single_img_w * single_img_h / 2);
      }
    } else if (calib_method_ == "custom") {
      ScopeProcessTime t(this->get_logger(), "stereo rectify");
      cv::Mat stereo_bgr;
      ImgConvertUtils::nv12_to_bgr_mat(const_cast<uint8_t *>(stereo_msg->data.data()), stereo_bgr, stereo_msg->width,
                                       stereo_msg->height);

      cv::Mat left_bgr = stereo_bgr.rowRange(0, single_img_h).clone();
      cv::Mat right_bgr = stereo_bgr.rowRange(single_img_h, stereo_msg->height).clone();
      // rectify
      cv::Mat left_bgr_rectify, right_bgr_rectify;
      stereo_rectifier_->rectify(left_bgr, right_bgr, left_bgr_rectify, right_bgr_rectify);
      ImgConvertUtils::bgr_mat_to_nv12(left_bgr_rectify, left_img_data.data());
      ImgConvertUtils::bgr_mat_to_nv12(right_bgr_rectify, right_img_data.data());
    }

  } else if (stereo_msg->encoding == "rgb8" || stereo_msg->encoding == "bgr8") {
    RCLCPP_INFO_ONCE(this->get_logger(), "=> stereo image size: %d x %d, format: %s", stereo_msg->width,
                     stereo_msg->height, stereo_msg->encoding.c_str());
    int single_img_w = stereo_msg->width;
    int single_img_h = stereo_msg->height / 2;
    size_t single_bgr_size = single_img_w * single_img_h * 3;

    cv::Mat stereo_bgr;
    if (stereo_msg->encoding == "rgb8") {
      cv::Mat stereo_rgb(stereo_msg->height, stereo_msg->width, CV_8UC3, const_cast<uint8_t *>(stereo_msg->data.data()),
                         stereo_msg->step);
      cv::cvtColor(stereo_rgb, stereo_bgr, cv::COLOR_RGB2BGR);
    } else {
      stereo_bgr = cv::Mat(stereo_msg->height, stereo_msg->width, CV_8UC3,
                           const_cast<uint8_t *>(stereo_msg->data.data()), stereo_msg->step);
    }

    cv::Mat left_bgr = stereo_bgr.rowRange(0, single_img_h).clone();
    cv::Mat right_bgr = stereo_bgr.rowRange(single_img_h, stereo_msg->height).clone();

    if (calib_method_ == "none") {
      if (single_img_w != model_input_w || single_img_h != model_input_h) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 5000,
            "\033[31m=> input image size not match model input size, need resize, [%d x %d] -> [%d x %d]\033[0m",
            single_img_w, single_img_h, model_input_w, model_input_h);
        cv::resize(left_bgr, left_bgr, cv::Size(model_input_w, model_input_h));
        cv::resize(right_bgr, right_bgr, cv::Size(model_input_w, model_input_h));
        if (camera_info_updated_ == false) {
          camera_info_updated_ = true;
          camera_intrinsic_->cx = camera_intrinsic_->cx * model_input_w / single_img_w;
          camera_intrinsic_->cy = camera_intrinsic_->cy * model_input_h / single_img_h;
          camera_intrinsic_->fx = camera_intrinsic_->fx * model_input_w / single_img_w;
          camera_intrinsic_->fy = camera_intrinsic_->fy * model_input_h / single_img_h;
          RCLCPP_WARN(this->get_logger(),
                      "\033[31m=> after resize, update camera intrinsic [fx, fy, cx, cy, baseline(m), doffs] : [%f, "
                      "%f, %f, %f, %f, %f]\033[0m",
                      camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy,
                      camera_intrinsic_->baseline, camera_intrinsic_->doffs);
        }
      }

      ImgConvertUtils::bgr_mat_to_nv12(left_bgr, left_img_data.data());
      ImgConvertUtils::bgr_mat_to_nv12(right_bgr, right_img_data.data());
    } else if (calib_method_ == "custom") {
      ScopeProcessTime t(this->get_logger(), "stereo rectify");
      // rectify
      cv::Mat left_bgr_rectify, right_bgr_rectify;
      stereo_rectifier_->rectify(left_bgr, right_bgr, left_bgr_rectify, right_bgr_rectify);
      ImgConvertUtils::bgr_mat_to_nv12(left_bgr_rectify, left_img_data.data());
      ImgConvertUtils::bgr_mat_to_nv12(right_bgr_rectify, right_img_data.data());
    }

  } else {
    RCLCPP_ERROR(this->get_logger(), "=> unsupported image encoding: %s", stereo_msg->encoding.c_str());
  }
  if (global_frame_cnt_ == 5) {
    cv::Mat top_bgr, bottom_bgr;
    ImgConvertUtils::nv12_to_bgr_mat(left_img_data.data(), top_bgr, model_input_w, model_input_h);
    ImgConvertUtils::nv12_to_bgr_mat(right_img_data.data(), bottom_bgr, model_input_w, model_input_h);
    top_is_left_ = judge_top_is_left_by_ORB(top_bgr, bottom_bgr);
    global_frame_cnt_++;
  }
}

void StereoNetNode::resize_nv12_image(const uint8_t *src_nv12, int src_w, int src_h, uint8_t *dst_nv12, int dst_w,
                                      int dst_h) {
  const uint8_t *src_y = src_nv12;
  const uint8_t *src_uv = src_nv12 + src_w * src_h;

  uint8_t *dst_y = dst_nv12;
  uint8_t *dst_uv = dst_nv12 + dst_w * dst_h;

  cv::Mat src_y_mat(src_h, src_w, CV_8UC1, const_cast<uint8_t *>(src_y));
  cv::Mat src_uv_mat(src_h / 2, src_w, CV_8UC1, const_cast<uint8_t *>(src_uv));

  cv::Mat dst_y_mat(dst_h, dst_w, CV_8UC1, dst_y);
  cv::Mat dst_uv_mat(dst_h / 2, dst_w, CV_8UC1, dst_uv);

  cv::resize(src_y_mat, dst_y_mat, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);
  cv::resize(src_uv_mat, dst_uv_mat, cv::Size(dst_w, dst_h / 2), 0, 0, cv::INTER_LINEAR);
}

void StereoNetNode::resize_stereo_nv12_image(const sensor_msgs::msg::Image::SharedPtr &stereo_msg, int src_w, int src_h,
                                             int dst_w, int dst_h, std::vector<uint8_t> &left_img_data,
                                             std::vector<uint8_t> &right_img_data) {
  const uint8_t *stereo_data = stereo_msg->data.data();

  const uint8_t *left_y = stereo_data;
  const uint8_t *right_y = stereo_data + src_w * src_h;

  const uint8_t *uv_base = stereo_data + stereo_msg->width * stereo_msg->height;
  const uint8_t *left_uv = uv_base;
  const uint8_t *right_uv = uv_base + src_w * src_h / 2;

  std::vector<uint8_t> left_src_nv12(src_w * src_h * 3 / 2);
  std::vector<uint8_t> right_src_nv12(src_w * src_h * 3 / 2);

  std::memcpy(left_src_nv12.data(), left_y, src_w * src_h);
  std::memcpy(left_src_nv12.data() + src_w * src_h, left_uv, src_w * src_h / 2);

  std::memcpy(right_src_nv12.data(), right_y, src_w * src_h);
  std::memcpy(right_src_nv12.data() + src_w * src_h, right_uv, src_w * src_h / 2);

  resize_nv12_image(left_src_nv12.data(), src_w, src_h, left_img_data.data(), dst_w, dst_h);
  resize_nv12_image(right_src_nv12.data(), src_w, src_h, right_img_data.data(), dst_w, dst_h);
}

void StereoNetNode::split_stereo_nv12_image(const sensor_msgs::msg::Image::SharedPtr &stereo_msg, int single_img_w,
                                            int single_img_h, std::vector<uint8_t> &left_nv12,
                                            std::vector<uint8_t> &right_nv12) {

  const uint8_t *stereo_data = stereo_msg->data.data();

  const int y_size = single_img_w * single_img_h;
  const int uv_size = y_size / 2;
  const int nv12_size = y_size * 3 / 2;

  left_nv12.resize(nv12_size);
  right_nv12.resize(nv12_size);

  // ================= Y =================
  const uint8_t *left_y = stereo_data;
  const uint8_t *right_y = stereo_data + y_size;

  std::memcpy(left_nv12.data(), left_y, y_size);
  std::memcpy(right_nv12.data(), right_y, y_size);

  // ================= UV =================
  const uint8_t *uv_base = stereo_data + stereo_msg->width * stereo_msg->height;

  const uint8_t *left_uv = uv_base;
  const uint8_t *right_uv = uv_base + uv_size;

  std::memcpy(left_nv12.data() + y_size, left_uv, uv_size);
  std::memcpy(right_nv12.data() + y_size, right_uv, uv_size);
}

void StereoNetNode::publish_function() {
  int count = 0;
  while (rclcpp::ok()) {
    do_save_result_once_ = save_result_once_;

    std::shared_ptr<PubData> pub_data;
    if (pub_data_queue_.get(pub_data, 100)) {
      // check timestamp disorder
      if (last_frame_timestamp_ == 0) {
        last_frame_timestamp_ = pub_data->timestamp;
      } else if (pub_data->timestamp <= last_frame_timestamp_) {
        RCLCPP_WARN(this->get_logger(), "=> drop one message to avoid timestamp disorder");
        continue;
      }
      last_frame_timestamp_ = pub_data->timestamp;

      auto now = this->get_clock()->now();
      auto latency = (now - pub_data->header.stamp).seconds() * 1000;
      // if latency > 1000ms, maybe the time stamp is not correct, set latency to 0
      pub_data->latency = latency > 1000 ? 0 : latency;
      performance_writer::Get()->record_performance(pub_data->latency);
      pub_data->fps = performance_writer::Get()->get_fps();
      pub_data->cpu_usage = performance_writer::Get()->get_cpu_usage();
      pub_data->bpu_usage = performance_writer::Get()->get_bpu_usage();
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 5000,
          "=> publish result, stamp: %u.%u, fps: %.2f, latency: %.2f ms, cpu_usage: %d%%, bpu_usage: %d%%",
          pub_data->header.stamp.sec, pub_data->header.stamp.nanosec, pub_data->fps, latency, pub_data->cpu_usage,
          pub_data->bpu_usage);

      if (!top_is_left_) {
        RCLCPP_ERROR_THROTTLE(
            this->get_logger(), *this->get_clock(), 5000,
            "\033[31m=> top is not left image, please swap the mipi cables, or set the mipi_channel and "
            "mipi_channel2 parameters.\033[0m");
      }

      // publish depth image
      {
        ScopeProcessTime t(this->get_logger(), "publish_depth_image");
        publish_depth_image(pub_data);
        publish_depth_camera_info(pub_data);
        publish_rectify_left_camera_info(pub_data);
        publish_rectify_right_camera_info(pub_data);
      }
      // pub rectified images
      {
        ScopeProcessTime t(this->get_logger(), "publish_rectified_image");
        publish_rectified_left_image(pub_data);
        publish_rectified_right_image(pub_data);
      }
      // publish pointcloud2
      {
        if (publish_pcd_enabled_) {
          ScopeProcessTime t(this->get_logger(), "publish_pointcloud2");
          publish_pointcloud2(pub_data);
        }
      }
      // publish origin images
      {
        if (publish_origin_enable_) {
          // publish origin images
          ScopeProcessTime t(this->get_logger(), "publish_origin_image");
          publish_origin_left_image(pub_data);
          publish_origin_right_image(pub_data);
        }
      }
      // publish visual image
      {
        if (!epipolar_mode_ && !feature_epipolar_mode_ && publish_visual_enabled_) {
          ScopeProcessTime t(this->get_logger(), "publish_visual_image");
          publish_visual_image(pub_data);
        }
      }
      // publish epipolar image
      {
        if (epipolar_mode_) {
          ScopeProcessTime t(this->get_logger(), "publish_epipolar_image", "warn");
          publish_epipolar_image(pub_data);
        }
      }
      // publish feature epipolar image
      {
        if (feature_epipolar_mode_) {
          ScopeProcessTime t(this->get_logger(), "publish_feature_epipolar_image", "warn");
          publish_feature_epipolar_image(pub_data);
        }
      }
      {
        // save result
        pub_data->count = count;
        if (save_result_flag_ && save_thread_num_ == 1) {
          save_result(pub_data);
          count++;
        } else if (save_result_flag_ && save_thread_num_ > 1) {
          if (save_thread_pool_ptr_->get_tasks_total() < max_save_task_) {
            if (save_origin_flag_ && (pub_data->origin_left.empty() || pub_data->origin_right.empty())) continue;
            if (save_pcd_flag_ && (!pub_data->pointcloud || pub_data->pointcloud->points.empty())) continue;
            if (save_visual_flag_ && pub_data->visual_img.empty()) continue;
            save_thread_pool_ptr_->detach_task([this, pub_data]() { save_result(pub_data); });
            count++;
          }
        } else if (do_save_result_once_) {
          save_result_once_ = false;
          do_save_result_once_ = false;
          if (save_thread_pool_ptr_->get_tasks_total() < max_save_task_) {
            save_thread_pool_ptr_->detach_task([this, pub_data]() { save_result_once(pub_data); });
            count++;
          }
        }
      }
    }
  }
}

void StereoNetNode::publish_depth_image(const std::shared_ptr<PubData> &pub_data) {
  if (depth_image_pub_->get_subscription_count() == 0) return;

  auto depth_msg = std::make_shared<sensor_msgs::msg::Image>();
  depth_msg->header = pub_data->header;
  depth_msg->header.frame_id = "camera_optical_frame";
  depth_msg->height = pub_data->depth.rows;
  depth_msg->width = pub_data->depth.cols;
  depth_msg->encoding = "mono16"; // Use 16-bit unsigned integer for depth in millimeters
  depth_msg->is_bigendian = false;
  depth_msg->step = pub_data->depth.cols * pub_data->depth.elemSize();
  size_t size = depth_msg->step * depth_msg->height;
  depth_msg->data.resize(size);
  std::memcpy(depth_msg->data.data(), pub_data->depth.data, size);
  depth_image_pub_->publish(*depth_msg);
}

void StereoNetNode::publish_depth_camera_info(const std::shared_ptr<PubData> &pub_data) {
  if (depth_camera_info_pub_->get_subscription_count() == 0) return;

  auto depth_camera_info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
  depth_camera_info_msg->header = pub_data->header;
  depth_camera_info_msg->header.frame_id = "camera_optical_frame";
  depth_camera_info_msg->height = pub_data->depth.rows;
  depth_camera_info_msg->width = pub_data->depth.cols;
  depth_camera_info_msg->distortion_model = "plumb_bob";
  depth_camera_info_msg->d = {0.0, 0.0, 0.0, 0.0, 0.0};

  // Set intrinsic parameters
  depth_camera_info_msg->k[0] = camera_intrinsic_->fx; // fx
  depth_camera_info_msg->k[2] = camera_intrinsic_->cx; // cx
  depth_camera_info_msg->k[4] = camera_intrinsic_->fy; // fy
  depth_camera_info_msg->k[5] = camera_intrinsic_->cy; // cy
  depth_camera_info_msg->k[8] = 1.0;

  // Set R
  depth_camera_info_msg->r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  // Set projection matrix
  depth_camera_info_msg->p[0] = camera_intrinsic_->fx; // fx
  depth_camera_info_msg->p[2] = camera_intrinsic_->cx; // cx
  depth_camera_info_msg->p[5] = camera_intrinsic_->fy; // fy
  depth_camera_info_msg->p[6] = camera_intrinsic_->cy; // cy
  depth_camera_info_msg->p[10] = 1.0;

  depth_camera_info_pub_->publish(*depth_camera_info_msg);
}

void StereoNetNode::publish_rectify_left_camera_info(const std::shared_ptr<PubData> &pub_data) {
  if (rectify_left_camera_info_pub_->get_subscription_count() == 0) return;

  auto left_camera_info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
  left_camera_info_msg->header = pub_data->header;
  left_camera_info_msg->header.frame_id = stereonet_frame_id_;
  left_camera_info_msg->height = pub_data->depth.rows;
  left_camera_info_msg->width = pub_data->depth.cols;
  left_camera_info_msg->distortion_model = "plumb_bob";
  left_camera_info_msg->d = {0.0, 0.0, 0.0, 0.0, 0.0};

  // Set intrinsic parameters
  left_camera_info_msg->k[0] = camera_intrinsic_->fx; // fx
  left_camera_info_msg->k[2] = camera_intrinsic_->cx; // cx
  left_camera_info_msg->k[4] = camera_intrinsic_->fy; // fy
  left_camera_info_msg->k[5] = camera_intrinsic_->cy; // cy
  left_camera_info_msg->k[8] = 1.0;

  // Set R
  if (origin_left_camera_info_ != nullptr)
    left_camera_info_msg->r = origin_left_camera_info_->r;
  else
    left_camera_info_msg->r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  // Set projection matrix
  left_camera_info_msg->p[0] = camera_intrinsic_->fx; // fx
  left_camera_info_msg->p[2] = camera_intrinsic_->cx; // cx
  left_camera_info_msg->p[5] = camera_intrinsic_->fy; // fy
  left_camera_info_msg->p[6] = camera_intrinsic_->cy; // cy
  left_camera_info_msg->p[10] = 1.0;

  rectify_left_camera_info_pub_->publish(*left_camera_info_msg);
}

void StereoNetNode::publish_rectify_right_camera_info(const std::shared_ptr<PubData> &pub_data) {
  if (rectify_right_camera_info_pub_->get_subscription_count() == 0) return;

  auto right_camera_info_msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
  right_camera_info_msg->header = pub_data->header;
  right_camera_info_msg->header.frame_id = stereonet_frame_id_right_;
  right_camera_info_msg->height = pub_data->depth.rows;
  right_camera_info_msg->width = pub_data->depth.cols;
  right_camera_info_msg->distortion_model = "plumb_bob";
  right_camera_info_msg->d = {0.0, 0.0, 0.0, 0.0, 0.0};

  // Set intrinsic parameters
  right_camera_info_msg->k[0] = camera_intrinsic_->fx; // fx
  right_camera_info_msg->k[2] = camera_intrinsic_->cx; // cx
  right_camera_info_msg->k[4] = camera_intrinsic_->fy; // fy
  right_camera_info_msg->k[5] = camera_intrinsic_->cy; // cy
  right_camera_info_msg->k[8] = 1.0;

  // Set R
  if (origin_camera_info_ != nullptr)
    right_camera_info_msg->r = origin_camera_info_->r;
  else
    right_camera_info_msg->r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  // Set projection matrix
  right_camera_info_msg->p[0] = camera_intrinsic_->fx;                                // fx
  right_camera_info_msg->p[2] = camera_intrinsic_->cx;                                // cx
  right_camera_info_msg->p[3] = -camera_intrinsic_->baseline * camera_intrinsic_->fx; // -b*fx
  right_camera_info_msg->p[5] = camera_intrinsic_->fy;                                // fy
  right_camera_info_msg->p[6] = camera_intrinsic_->cy;                                // cy
  right_camera_info_msg->p[10] = 1.0;

  rectify_right_camera_info_pub_->publish(*right_camera_info_msg);
}

void StereoNetNode::publish_rectified_left_image(const std::shared_ptr<PubData> &pub_data) {
  if (rectify_left_image_pub_->get_subscription_count() == 0) return;
  auto left_msg = std::make_shared<sensor_msgs::msg::Image>();
  left_msg->header = pub_data->header;
  left_msg->header.frame_id = "camera_optical_frame";
  int width = pub_data->disp.cols;
  int height = pub_data->disp.rows;
  left_msg->height = height;
  left_msg->width = width;
  left_msg->is_bigendian = false;

  if (publish_rectify_bgr_) {
    left_msg->encoding = "bgr8";
    left_msg->step = width * 3; // BGR step
    size_t size = width * height * 3;
    left_msg->data.resize(size);
    cv::Mat bgr;
    ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_left_img_data.data(), bgr, width, height);
    std::memcpy(left_msg->data.data(), bgr.data, size);
    rectify_left_image_pub_->publish(*left_msg);
  } else {
    left_msg->encoding = "nv12";
    left_msg->step = width; // Y plane step
    size_t size = width * height * 3 / 2;
    left_msg->data.resize(size);
    std::memcpy(left_msg->data.data(), pub_data->rectify_left_img_data.data(), size);
    rectify_left_image_pub_->publish(*left_msg);
  }
}

void StereoNetNode::publish_rectified_right_image(const std::shared_ptr<PubData> &pub_data) {
  if (rectify_right_image_pub_->get_subscription_count() == 0) return;
  auto right_msg = std::make_shared<sensor_msgs::msg::Image>();
  right_msg->header = pub_data->header;
  right_msg->header.frame_id = stereonet_frame_id_right_;
  int width = pub_data->disp.cols;
  int height = pub_data->disp.rows;
  right_msg->height = height;
  right_msg->width = width;
  right_msg->is_bigendian = false;
  if (publish_rectify_bgr_) {
    right_msg->encoding = "bgr8";
    right_msg->step = width * 3; // BGR step
    size_t size = width * height * 3;
    right_msg->data.resize(size);
    cv::Mat bgr;
    ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_right_img_data.data(), bgr, width, height);
    std::memcpy(right_msg->data.data(), bgr.data, size);
    rectify_right_image_pub_->publish(*right_msg);
  } else {
    right_msg->encoding = "nv12";
    right_msg->step = width; // Y plane step
    size_t size = width * height * 3 / 2;
    right_msg->data.resize(size);
    std::memcpy(right_msg->data.data(), pub_data->rectify_right_img_data.data(), size);
    rectify_right_image_pub_->publish(*right_msg);
  }
}

void StereoNetNode::publish_pointcloud2(const std::shared_ptr<PubData> &pub_data) {
  if (pointcloud2_pub_->get_subscription_count() == 0 && !save_result_flag_ && !do_save_result_once_) return;
  cv::Mat bgr;
  ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_left_img_data.data(), bgr, pub_data->disp.cols,
                                   pub_data->disp.rows);
  const int step = pointcloud_downsample_step_; // downsample
  const int rows = pub_data->depth.rows;
  const int cols = pub_data->depth.cols;
  const float fx = camera_intrinsic_->fx;
  const float fy = camera_intrinsic_->fy;
  const float cx = camera_intrinsic_->cx;
  const float cy = camera_intrinsic_->cy;

  int num_threads = omp_get_max_threads();
  std::vector<std::vector<pcl::PointXYZRGB>> thread_points(num_threads);

  // Estimate the capacity needed for each thread to avoid frequent resizing during push_back
  size_t est_points_per_thread = (rows / step) * (cols / step) / num_threads;
  for (auto &v : thread_points) v.reserve(est_points_per_thread);

#pragma omp parallel for schedule(static)
  for (int v = 0; v < rows; v += step) {
    int tid = omp_get_thread_num();
    std::vector<pcl::PointXYZRGB> &local_points = thread_points[tid];
    const uint16_t *depth_row = pub_data->depth.ptr<uint16_t>(v);
    const cv::Vec3b *bgr_row = bgr.ptr<cv::Vec3b>(v);
    for (int u = 0; u < cols; u += step) {
      float z = depth_row[u] * 0.001f; // mm to m
      if (z <= 0 || z > pointcloud_depth_max_) continue;
      float x = (u - cx) * z / fx;
      float y = (v - cy) * z / fy;
      if (-y > pointcloud_height_max_ || -y < pointcloud_height_min_) continue;
      pcl::PointXYZRGB pt;
      pt.x = z;
      pt.y = -x;
      pt.z = -y;
      const cv::Vec3b &color = bgr_row[u];
      pt.r = color[2];
      pt.g = color[1];
      pt.b = color[0];
      local_points.push_back(pt);
    }
  }

  // Count total points and reserve space in pcl_cloud
  size_t total_points = 0;
  for (auto &v : thread_points) total_points += v.size();
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl_cloud->points.reserve(total_points);
  for (auto &v : thread_points) pcl_cloud->points.insert(pcl_cloud->points.end(), v.begin(), v.end());
  pcl_cloud->width = pcl_cloud->points.size();
  pcl_cloud->height = 1;
  pcl_cloud->is_dense = false;

  sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  if (pcl_filter_enable_) {
    ScopeProcessTime t(this->get_logger(), "pcl filter");
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud =
    // PCLFilterUtils::statisticalOutlierRemoval(pcl_cloud, voxel_leaf_size_, mean_k_, std_thresh_);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud =
        PCLFilterUtils::gridBasedOutlierRemoval(pcl_cloud, grid_size_, grid_min_point_count_);
    pcl::toROSMsg(*filtered_cloud, *cloud_msg);
    if (save_pcd_flag_ || do_save_result_once_) pub_data->pointcloud = filtered_cloud;
  } else {
    pcl::toROSMsg(*pcl_cloud, *cloud_msg);
    if (save_pcd_flag_ || do_save_result_once_) pub_data->pointcloud = pcl_cloud;
  }
  cloud_msg->header = pub_data->header;
  cloud_msg->header.frame_id = stereonet_frame_id_;
  cloud_msg->is_dense = false;
  cloud_msg->is_bigendian = false;
  pointcloud2_pub_->publish(*cloud_msg);
}

void StereoNetNode::publish_origin_left_image(const std::shared_ptr<PubData> &pub_data) {
  if (origin_left_image_pub_->get_subscription_count() == 0 && !save_result_flag_ && !do_save_result_once_ &&
      !epipolar_mode_ && !feature_epipolar_mode_)
    return;
  if (pub_data->origin_stereo_msg->encoding == "nv12") {
    auto left_msg = std::make_shared<sensor_msgs::msg::Image>();
    left_msg->header = pub_data->header;
    left_msg->header.frame_id = stereonet_frame_id_;
    int single_img_w = pub_data->origin_stereo_msg->width;
    int single_img_h = pub_data->origin_stereo_msg->height / 2;
    left_msg->height = single_img_h;
    left_msg->width = single_img_w;
    left_msg->encoding = "nv12";
    left_msg->is_bigendian = false;
    left_msg->step = single_img_w; // Y plane step
    size_t size = single_img_w * single_img_h * 3 / 2;
    left_msg->data.resize(size);
    std::memcpy(left_msg->data.data(), pub_data->origin_stereo_msg->data.data(), single_img_w * single_img_h);
    std::memcpy(left_msg->data.data() + single_img_w * single_img_h,
                pub_data->origin_stereo_msg->data.data() +
                    pub_data->origin_stereo_msg->width * pub_data->origin_stereo_msg->height,
                single_img_w * single_img_h / 2);
    origin_left_image_pub_->publish(*left_msg);
    if (save_origin_flag_ || do_save_result_once_ || epipolar_mode_ || feature_epipolar_mode_) {
      ImgConvertUtils::nv12_to_bgr_mat(left_msg->data.data(), pub_data->origin_left, left_msg->width, left_msg->height);
    }
  } else if (pub_data->origin_stereo_msg->encoding == "rgb8" || pub_data->origin_stereo_msg->encoding == "bgr8") {
    auto left_msg = std::make_shared<sensor_msgs::msg::Image>();
    left_msg->header = pub_data->header;
    left_msg->header.frame_id = stereonet_frame_id_;
    int single_img_w = pub_data->origin_stereo_msg->width;
    int single_img_h = pub_data->origin_stereo_msg->height / 2;
    left_msg->height = single_img_h;
    left_msg->width = single_img_w;
    left_msg->is_bigendian = false;

    if (pub_data->origin_stereo_msg->encoding == "rgb8") {
      left_msg->encoding = "rgb8";
      left_msg->step = single_img_w * 3; // RGB step
      size_t size = single_img_w * single_img_h * 3;
      left_msg->data.resize(size);
      cv::Mat stereo_rgb(pub_data->origin_stereo_msg->height, pub_data->origin_stereo_msg->width, CV_8UC3,
                         const_cast<uint8_t *>(pub_data->origin_stereo_msg->data.data()),
                         pub_data->origin_stereo_msg->step);
      cv::Mat left_rgb = stereo_rgb.rowRange(0, single_img_h).clone();
      std::memcpy(left_msg->data.data(), left_rgb.data, size);
      origin_left_image_pub_->publish(*left_msg);

      if (save_origin_flag_ || do_save_result_once_ || epipolar_mode_ || feature_epipolar_mode_) {
        cv::Mat rgb(left_msg->height, left_msg->width, CV_8UC3, const_cast<uint8_t *>(left_msg->data.data()),
                    left_msg->step);
        cv::cvtColor(rgb, pub_data->origin_left, cv::COLOR_RGB2BGR);
      }
    } else {
      left_msg->encoding = "bgr8";
      left_msg->step = single_img_w * 3; // BGR step
      size_t size = single_img_w * single_img_h * 3;
      left_msg->data.resize(size);
      cv::Mat stereo_bgr(pub_data->origin_stereo_msg->height, pub_data->origin_stereo_msg->width, CV_8UC3,
                         const_cast<uint8_t *>(pub_data->origin_stereo_msg->data.data()),
                         pub_data->origin_stereo_msg->step);
      cv::Mat left_bgr = stereo_bgr.rowRange(0, single_img_h).clone();
      std::memcpy(left_msg->data.data(), left_bgr.data, size);
      origin_left_image_pub_->publish(*left_msg);

      if (save_origin_flag_ || do_save_result_once_ || epipolar_mode_ || feature_epipolar_mode_) {
        pub_data->origin_left = cv::Mat(left_msg->height, left_msg->width, CV_8UC3,
                                        const_cast<uint8_t *>(left_msg->data.data()), left_msg->step)
                                    .clone();
      }
    }
  }
}

void StereoNetNode::publish_origin_right_image(const std::shared_ptr<PubData> &pub_data) {
  if (origin_right_image_pub_->get_subscription_count() == 0 && !save_result_flag_ && !do_save_result_once_ &&
      !epipolar_mode_ && !feature_epipolar_mode_)
    return;
  if (pub_data->origin_stereo_msg->encoding == "nv12") {
    auto right_msg = std::make_shared<sensor_msgs::msg::Image>();
    right_msg->header = pub_data->header;
    right_msg->header.frame_id = stereonet_frame_id_right_;
    int single_img_w = pub_data->origin_stereo_msg->width;
    int single_img_h = pub_data->origin_stereo_msg->height / 2;
    right_msg->height = single_img_h;
    right_msg->width = single_img_w;
    right_msg->encoding = "nv12";
    right_msg->is_bigendian = false;
    right_msg->step = single_img_w; // Y plane step
    size_t size = single_img_w * single_img_h * 3 / 2;
    right_msg->data.resize(size);
    std::memcpy(right_msg->data.data(), pub_data->origin_stereo_msg->data.data() + single_img_w * single_img_h,
                single_img_w * single_img_h);
    std::memcpy(right_msg->data.data() + single_img_w * single_img_h,
                pub_data->origin_stereo_msg->data.data() +
                    pub_data->origin_stereo_msg->width * pub_data->origin_stereo_msg->height +
                    single_img_w * single_img_h / 2,
                single_img_w * single_img_h / 2);
    origin_right_image_pub_->publish(*right_msg);

    if (save_origin_flag_ || do_save_result_once_ || epipolar_mode_ || feature_epipolar_mode_) {
      ImgConvertUtils::nv12_to_bgr_mat(right_msg->data.data(), pub_data->origin_right, right_msg->width,
                                       right_msg->height);
    }
  } else if (pub_data->origin_stereo_msg->encoding == "rgb8" || pub_data->origin_stereo_msg->encoding == "bgr8") {
    auto right_msg = std::make_shared<sensor_msgs::msg::Image>();
    right_msg->header = pub_data->header;
    right_msg->header.frame_id = stereonet_frame_id_right_;
    int single_img_w = pub_data->origin_stereo_msg->width;
    int single_img_h = pub_data->origin_stereo_msg->height / 2;
    right_msg->height = single_img_h;
    right_msg->width = single_img_w;
    right_msg->is_bigendian = false;

    if (pub_data->origin_stereo_msg->encoding == "rgb8") {
      right_msg->encoding = "rgb8";
      right_msg->step = single_img_w * 3; // RGB step
      size_t size = single_img_w * single_img_h * 3;
      right_msg->data.resize(size);
      cv::Mat stereo_rgb(pub_data->origin_stereo_msg->height, pub_data->origin_stereo_msg->width, CV_8UC3,
                         const_cast<uint8_t *>(pub_data->origin_stereo_msg->data.data()),
                         pub_data->origin_stereo_msg->step);
      cv::Mat right_rgb = stereo_rgb.rowRange(single_img_h, pub_data->origin_stereo_msg->height).clone();
      std::memcpy(right_msg->data.data(), right_rgb.data, size);
      origin_right_image_pub_->publish(*right_msg);

      if (save_origin_flag_ || do_save_result_once_ || epipolar_mode_ || feature_epipolar_mode_) {
        cv::Mat rgb(right_msg->height, right_msg->width, CV_8UC3, const_cast<uint8_t *>(right_msg->data.data()),
                    right_msg->step);
        cv::cvtColor(rgb, pub_data->origin_right, cv::COLOR_RGB2BGR);
      }
    } else {
      right_msg->encoding = "bgr8";
      right_msg->step = single_img_w * 3; // BGR step
      size_t size = single_img_w * single_img_h * 3;
      right_msg->data.resize(size);
      cv::Mat stereo_bgr(pub_data->origin_stereo_msg->height, pub_data->origin_stereo_msg->width, CV_8UC3,
                         const_cast<uint8_t *>(pub_data->origin_stereo_msg->data.data()),
                         pub_data->origin_stereo_msg->step);
      cv::Mat right_bgr = stereo_bgr.rowRange(single_img_h, pub_data->origin_stereo_msg->height).clone();
      std::memcpy(right_msg->data.data(), right_bgr.data, size);
      origin_right_image_pub_->publish(*right_msg);

      if (save_origin_flag_ || do_save_result_once_ || epipolar_mode_ || feature_epipolar_mode_) {
        pub_data->origin_right = cv::Mat(right_msg->height, right_msg->width, CV_8UC3,
                                         const_cast<uint8_t *>(right_msg->data.data()), right_msg->step)
                                     .clone();
      }
    }
  }
}

/**
 * @brief Compute the near depth percentile
 * @param depth The depth image
 * @param percentile The percentile
 * @return The near depth
 */
static double compute_near_depth_percentile(const cv::Mat &depth, double percentile = 0.02) {
  // std::vector<double> valid;
  // valid.reserve(depth.total());

  // for (int y = 0; y < depth.rows; ++y) {
  //   const uint16_t *ptr = depth.ptr<uint16_t>(y);
  //   for (int x = 0; x < depth.cols; ++x)
  //     if (ptr[x] > 0) valid.push_back(ptr[x]);
  // }
  // if (valid.empty()) return -1;
  // size_t k = static_cast<size_t>(percentile * valid.size());
  // std::nth_element(valid.begin(), valid.begin() + k, valid.end());
  // return valid[k];
  const int rows = depth.rows;
  const int cols = depth.cols;

  int num_threads = omp_get_max_threads();
  std::vector<std::vector<uint16_t>> locals(num_threads);
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    auto &buf = locals[tid];
    buf.reserve(depth.total() / num_threads);
#pragma omp for schedule(static)
    for (int y = 0; y < rows; ++y) {
      const uint16_t *ptr = depth.ptr<uint16_t>(y);
      for (int x = 0; x < cols; ++x) {
        if (ptr[x] > 0) buf.push_back(ptr[x]);
      }
    }
  }

  size_t total = 0;
  for (auto &v : locals) total += v.size();
  if (total == 0) return -1;

  std::vector<uint16_t> valid;
  valid.reserve(total);
  for (auto &v : locals) valid.insert(valid.end(), v.begin(), v.end());

  size_t k = static_cast<size_t>(percentile * valid.size());
  std::nth_element(valid.begin(), valid.begin() + k, valid.end());
  return valid[k];
}

/**
 * @brief Extract center ROI from depth image
 * @param depth The depth image
 * @param cx The center x coordinate
 * @param cy The center y coordinate
 * @return The extracted ROI
 */
static RoiVec extract_center_roi(const cv::Mat &depth, int cx, int cy, int roi_size) {
  RoiVec out;
  if (depth.empty() || depth.type() != CV_16UC1) return out;

  int x0 = cx - roi_size / 2;
  int y0 = cy - roi_size / 2;

  // clamp
  x0 = std::max(0, std::min(x0, depth.cols - roi_size));
  y0 = std::max(0, std::min(y0, depth.rows - roi_size));

  out.reserve(roi_size * roi_size);
  for (int r = 0; r < roi_size; ++r) {
    const uint16_t *rowPtr = depth.ptr<uint16_t>(y0 + r);
    for (int c = 0; c < roi_size; ++c) {
      out.push_back(rowPtr[x0 + c]);
    }
  }
  return out;
}

/**
 * @brief Merge buffer frames into a single vector and filter out invalid (zero) depths
 * @param buf The buffer frames
 * @return The merged and filtered vector
 */
static std::vector<uint16_t> merge_and_filter_valid(const std::deque<RoiVec> &buf) {
  size_t total = 0;
  for (const auto &rv : buf) total += rv.size();
  std::vector<uint16_t> merged;
  merged.reserve(total);

  for (const auto &rv : buf) {
    for (uint16_t v : rv) {
      if (v != 0) merged.push_back(v); // drop 0 as invalid
    }
  }
  return merged;
}

/**
 * @brief Compute trimmed mean and ranges after removing lowest k and highest k elements
 * Returns tuple(mean_mm, neg_range_mm, pos_range_mm, trimmed_count)
 * @param vals The vector of depth values
 * @param trim_ratio The ratio of trimming
 * @return The trimmed mean, negative range, positive range, and trimmed count
 */
static std::tuple<double, double, double, size_t> compute_trimmed_stats(std::vector<uint16_t> &vals,
                                                                        double trim_ratio) {
  // Remove invalid early
  if (vals.empty()) return {0.0, 0.0, 0.0, 0};

  std::sort(vals.begin(), vals.end());
  size_t N = vals.size();
  size_t k = static_cast<size_t>(floor(N * trim_ratio));

  // ensure we keep at least 1 element
  if (N <= 2 * k) {
    // reduce k so at least 1 element remains
    if (N > 1)
      k = (N - 1) / 2;
    else
      k = 0;
  }

  size_t start = k;
  size_t end = N - k; // exclusive
  if (start >= end) {
    // fallback: use entire range
    start = 0;
    end = N;
  }

  // accumulate mean
  double sum = 0.0;
  for (size_t i = start; i < end; ++i) sum += static_cast<double>(vals[i]);
  size_t count = end - start;
  double mean = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;

  double min_trim = (count > 0) ? static_cast<double>(vals[start]) : 0.0;
  double max_trim = (count > 0) ? static_cast<double>(vals[end - 1]) : 0.0;

  double neg_range = mean - min_trim; // >=0
  double pos_range = max_trim - mean; // >=0

  return {mean, neg_range, pos_range, count};
}

/**
 * @brief Compute the ground angle based on the depth image
 * @param depth The depth image
 * @param intr The camera intrinsic
 * @param roi The ROI
 * @param min_valid_points The minimum valid points
 * @param angle_deg The output ground angle
 * @return true if the ground angle is valid
 */
static bool compute_ground_xz_angle(const cv::Mat &depth, const CameraIntrinsic &intr, const cv::Rect &roi,
                                    int min_valid_points, double &angle_deg) {
  if (depth.empty() || depth.type() != CV_16UC1) return false;
  if (intr.fx <= 0 || intr.fy <= 0) return false;

  std::vector<cv::Point3d> pts;
  pts.reserve(roi.width * roi.height);

  for (int v = roi.y; v < roi.y + roi.height; ++v) {
    const uint16_t *row = depth.ptr<uint16_t>(v);
    for (int u = roi.x; u < roi.x + roi.width; ++u) {
      uint16_t d_mm = row[u];
      if (d_mm == 0) continue;

      double z = d_mm * 0.001; // mm -> m
      double x = (static_cast<double>(u) - intr.cx) * z / intr.fx;
      double y = (static_cast<double>(v) - intr.cy) * z / intr.fy;

      pts.emplace_back(x, y, z);
    }
  }

  if (static_cast<int>(pts.size()) < min_valid_points) return false;

  // Fit plane as: y = a * x + b * z + c
  cv::Mat A(static_cast<int>(pts.size()), 3, CV_64F);
  cv::Mat B(static_cast<int>(pts.size()), 1, CV_64F);

  for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
    A.at<double>(i, 0) = pts[i].x;
    A.at<double>(i, 1) = pts[i].z;
    A.at<double>(i, 2) = 1.0;
    B.at<double>(i, 0) = pts[i].y;
  }

  cv::Mat X;
  if (!cv::solve(A, B, X, cv::DECOMP_SVD)) return false;

  double a = X.at<double>(0, 0);
  double b = X.at<double>(1, 0);

  // Plane normal is [-a, 1, -b].
  // XZ plane normal is [0, 1, 0].
  // Angle between ground plane and XZ plane.
  angle_deg = std::atan(std::sqrt(a * a + b * b)) * 180.0 / M_PI;
  return true;
}

void StereoNetNode::publish_visual_image(const std::shared_ptr<PubData> &pub_data) {
  if (visual_image_pub_->get_subscription_count() == 0 && !save_result_flag_ && !do_save_result_once_) return;
  // ===================================== render visual image ==============================================
  int width = pub_data->disp.cols;
  int height = pub_data->disp.rows;
  cv::Mat left_bgr = pub_data->left_bgr;

  cv::Mat visual_img;

  if (render_type_.rfind("outdoor", 0) == 0) {
    /*
    std::vector<float> disp_vals;
    disp_vals.reserve(pub_data->disp.rows * pub_data->disp.cols);

    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (int r = 0; r < pub_data->disp.rows; ++r) {
      const float *row_ptr = pub_data->disp.ptr<float>(r);
      for (int c = 0; c < pub_data->disp.cols; ++c) {
        float v = row_ptr[c];
        if (v > 0) {
          disp_vals.push_back(v);
          if (v < min_val) min_val = v;
          if (v > max_val) max_val = v;
        }
      }
    }

    auto getQuantile = [&](double q) {
      size_t idx = static_cast<size_t>(q * (disp_vals.size() - 1));
      std::nth_element(disp_vals.begin(), disp_vals.begin() + idx, disp_vals.end());
      return disp_vals[idx];
    };
    float q10 = getQuantile(0.10);
    float q50 = getQuantile(0.50);
    float q90 = getQuantile(0.90);

    double scale10 = 0.25 * 255.0 / (q10 - min_val + 1e-6);
    double scale50 = 0.25 * 255.0 / (q50 - q10 + 1e-6);
    double scale90 = 0.25 * 255.0 / (q90 - q50 + 1e-6);
    double scaleMax = 0.25 * 255.0 / (max_val - q90 + 1e-6);

    visual_img.create(pub_data->disp.size(), CV_8UC3);
    for (int r = 0; r < pub_data->disp.rows; ++r) {
      const float *row_ptr = pub_data->disp.ptr<float>(r);
      cv::Vec3b *out_ptr = visual_img.ptr<cv::Vec3b>(r);
      for (int c = 0; c < pub_data->disp.cols; ++c) {
        float pixel = row_ptr[c];
        uint8_t val = 0;
        if (pixel > 0) {
          if (pixel <= q10) {
            val = static_cast<uint8_t>((pixel - min_val) * scale10);
          } else if (pixel <= q50) {
            val = static_cast<uint8_t>(0.25 * 255 + (pixel - q10) * scale50);
          } else if (pixel <= q90) {
            val = static_cast<uint8_t>(0.5 * 255 + (pixel - q50) * scale90);
          } else {
            val = static_cast<uint8_t>(0.75 * 255 + (pixel - q90) * scaleMax);
          }
        }
        out_ptr[c] = cv::Vec3b(val, val, val);
      }
    }
    */

    // Using OpenMP to parallelize the extraction of valid disparity values and computation of min/max
    std::vector<float> disp_vals;
    disp_vals.reserve(pub_data->disp.rows * pub_data->disp.cols);
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
#pragma omp parallel
    {
      std::vector<float> local_vals;
      local_vals.reserve(pub_data->disp.rows * pub_data->disp.cols / 4);
      float local_min = std::numeric_limits<float>::max();
      float local_max = std::numeric_limits<float>::lowest();
#pragma omp for nowait
      for (int r = 0; r < pub_data->disp.rows; ++r) {
        const float *row_ptr = pub_data->disp.ptr<float>(r);
        for (int c = 0; c < pub_data->disp.cols; ++c) {
          float v = row_ptr[c];
          if (v > 0) {
            local_vals.push_back(v);
            if (v < local_min) local_min = v;
            if (v > local_max) local_max = v;
          }
        }
      }
#pragma omp critical
      {
        disp_vals.insert(disp_vals.end(), local_vals.begin(), local_vals.end());
        if (local_min < min_val) min_val = local_min;
        if (local_max > max_val) max_val = local_max;
      }
    }

    // calculate quantiles
    auto getQuantile = [&](double q) {
      size_t idx = static_cast<size_t>(q * (disp_vals.size() - 1));
      std::nth_element(disp_vals.begin(), disp_vals.begin() + idx, disp_vals.end());
      return disp_vals[idx];
    };

    float q10 = getQuantile(0.10);
    float q50 = getQuantile(0.50);
    float q90 = getQuantile(0.90);

    // OpenMP + NEON to parallelize the mapping of disparity values to visual representation
    double scale10 = 0.25 * 255.0 / (q10 - min_val + 1e-6);
    double scale50 = 0.25 * 255.0 / (q50 - q10 + 1e-6);
    double scale90 = 0.25 * 255.0 / (q90 - q50 + 1e-6);
    double scaleMax = 0.25 * 255.0 / (max_val - q90 + 1e-6);

    visual_img.create(pub_data->disp.size(), CV_8UC3);

#pragma omp parallel for
    for (int r = 0; r < pub_data->disp.rows; ++r) {
      const float *row_ptr = pub_data->disp.ptr<float>(r);
      cv::Vec3b *out_ptr = visual_img.ptr<cv::Vec3b>(r);

      int c = 0;
      for (; c + 4 <= pub_data->disp.cols; c += 4) {
        float32x4_t pixels = vld1q_f32(row_ptr + c);
        float px[4];
        vst1q_f32(px, pixels);
        uint8_t vals[4];
        for (int i = 0; i < 4; ++i) {
          float p = px[i];
          if (p <= 0)
            vals[i] = 0;
          else if (p <= q10)
            vals[i] = static_cast<uint8_t>((p - min_val) * scale10);
          else if (p <= q50)
            vals[i] = static_cast<uint8_t>(0.25 * 255 + (p - q10) * scale50);
          else if (p <= q90)
            vals[i] = static_cast<uint8_t>(0.5 * 255 + (p - q50) * scale90);
          else
            vals[i] = static_cast<uint8_t>(0.75 * 255 + (p - q90) * scaleMax);
        }
        for (int i = 0; i < 4; ++i) out_ptr[c + i] = cv::Vec3b(vals[i], vals[i], vals[i]);
      }

      for (; c < pub_data->disp.cols; ++c) {
        float pixel = row_ptr[c];
        uint8_t val = 0;
        if (pixel > 0) {
          if (pixel <= q10)
            val = static_cast<uint8_t>((pixel - min_val) * scale10);
          else if (pixel <= q50)
            val = static_cast<uint8_t>(0.25 * 255 + (pixel - q10) * scale50);
          else if (pixel <= q90)
            val = static_cast<uint8_t>(0.5 * 255 + (pixel - q50) * scale90);
          else
            val = static_cast<uint8_t>(0.75 * 255 + (pixel - q90) * scaleMax);
        }
        out_ptr[c] = cv::Vec3b(val, val, val);
      }
    }
  } else if (render_type_.rfind("distance", 0) == 0) {
    static double fb = camera_intrinsic_->baseline * camera_intrinsic_->fx;
    static double z_near_ema_mm = -1;
    static int frame_cnt = 0;
    static int interval = use_local_image_flag_ ? 1 : 5;
    if (render_z_near_ > 0.0) {
      z_near_ema_mm = render_z_near_ * 1000.0;
    } else {
      if (frame_cnt++ % interval == 0) {
        double z_near = compute_near_depth_percentile(pub_data->depth, 0.02);
        if (z_near > 0) {
          if (z_near_ema_mm < 0)
            z_near_ema_mm = z_near;
          else
            z_near_ema_mm = 0.05 * z_near + 0.95 * z_near_ema_mm;
        }
      }
    }
    double z_far = z_near_ema_mm + render_z_range_ * 1000.0;
    int d_max = static_cast<int>(fb / (z_near_ema_mm / 1000.0) - camera_intrinsic_->doffs);
    int d_min = static_cast<int>(fb / (z_far / 1000.0) - camera_intrinsic_->doffs);
    // pub_data->disp.convertTo(visual_img, CV_8UC1, 255.0 / (d_max - d_min), -d_min * 255.0 / (d_max - d_min));
    // visual_img.setTo(0, pub_data->disp < d_min);
    // visual_img.setTo(255, pub_data->disp > d_max);
    // cv::cvtColor(visual_img, visual_img, cv::COLOR_GRAY2BGR);

    float32x4_t v_dmin = vdupq_n_f32(d_min);
    float32x4_t v_zero = vdupq_n_f32(0.f);
    float32x4_t v_255 = vdupq_n_f32(255.f);
    const float scale = 255.0f / (d_max - d_min);
    float32x4_t v_scale = vdupq_n_f32(scale);
    visual_img.create(pub_data->disp.size(), CV_8UC1);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < pub_data->disp.rows; ++y) {
      const float *dptr = pub_data->disp.ptr<float>(y);
      uchar *optr = visual_img.ptr<uchar>(y);
      int x = 0;
      for (; x <= pub_data->disp.cols - 4; x += 4) {
        float32x4_t v = vld1q_f32(dptr + x);
        v = vsubq_f32(v, v_dmin);
        v = vmulq_f32(v, v_scale);
        v = vmaxq_f32(v, v_zero);
        v = vminq_f32(v, v_255);
        uint8x8_t u8 = vqmovn_u16(vcombine_u16(vmovn_u32(vcvtq_u32_f32(v)), vdup_n_u16(0)));
        vst1_u8(optr + x, u8);
      }

      // tail
      for (; x < pub_data->disp.cols; ++x) {
        float d = dptr[x];
        uchar v;
        if (d <= d_min)
          v = 0;
        else if (d >= d_max)
          v = 255;
        else
          v = static_cast<uchar>((d - d_min) * scale);
        optr[x] = v;
      }
    }
    cv::cvtColor(visual_img, visual_img, cv::COLOR_GRAY2BGR);
  } else {
    // Convert to 8-bit scaled image
    pub_data->disp.convertTo(visual_img, CV_8UC1, 255.0 / render_max_disp_);
    cv::cvtColor(visual_img, visual_img, cv::COLOR_GRAY2BGR);
  }

  static cv::Mat lut;
  if (lut.empty()) {
    cv::Mat tmp(1, 256, CV_8UC1);
    if (render_type_.find("reverse") != std::string::npos) {
      for (int i = 0; i < 256; i++) tmp.at<uchar>(i) = 255 - i;
    } else {
      for (int i = 0; i < 256; i++) tmp.at<uchar>(i) = i;
    }
    cv::applyColorMap(tmp, lut, cv::COLORMAP_JET);
  }
  cv::LUT(visual_img, lut, visual_img);
  cv::Mat mask = (pub_data->disp == 0);
  visual_img.setTo(cv::Vec3b(0, 0, 0), mask);

  cv::vconcat(left_bgr, visual_img, visual_img);
  // ===================================== render depth =====================================================
  double font_scale = std::min(left_bgr.cols, left_bgr.rows) / 700.0;
  int set_num = 6;
  int x_step = left_bgr.cols / set_num;
  int y_step = left_bgr.rows / set_num;
  for (int i = 1; i < set_num; ++i) {
    // vertical line
    cv::line(visual_img, cv::Point(i * x_step, 0), cv::Point(i * x_step, visual_img.rows), cv::Scalar(255, 255, 255),
             1);
    // horizontal line
    cv::line(visual_img, cv::Point(0, i * y_step), cv::Point(left_bgr.cols, i * y_step), cv::Scalar(255, 255, 255), 1);
    cv::line(visual_img, cv::Point(0, left_bgr.rows + i * y_step), cv::Point(left_bgr.cols, left_bgr.rows + i * y_step),
             cv::Scalar(255, 255, 255), 1);
  }

  for (int i = 1; i < set_num; ++i) {
    for (int j = 1; j < set_num; ++j) {
      int x = i * x_step;
      int y = j * y_step;
      float depth_value = pub_data->depth.at<uint16_t>(y, x) * 0.001f; // convert mm to m
      std::stringstream depth_text;
      depth_text << std::fixed << std::setprecision(depth_decimal_num_) << depth_value << "m";
      cv::putText(visual_img, depth_text.str(), cv::Point(x + 5, y - 5), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                  CV_RGB(255, 255, 255), 2);
      cv::putText(visual_img, depth_text.str(), cv::Point(x + 5, left_bgr.rows + y - 5), cv::FONT_HERSHEY_SIMPLEX,
                  font_scale, CV_RGB(255, 255, 255), 2);
      if (measure_mode_) {
        if (i == set_num / 2 && j == set_num / 2) {
          RoiVec roi = extract_center_roi(pub_data->depth, x, y, roi_size_);
          roi_buffer.push_back(roi);
          if (roi_buffer.size() > 10) roi_buffer.pop_front();
          std::vector<uint16_t> merged = merge_and_filter_valid(roi_buffer);
          if (!merged.empty()) {
            auto temp = merged;
            double mean = 0.0, neg_range = 0.0, pos_range = 0.0, err_percent = 0.0;
            size_t cnt;
            double trim_ratio = 0.05;
            std::tie(mean, neg_range, pos_range, cnt) = compute_trimmed_stats(temp, trim_ratio);
            mean = std::round(mean);
            neg_range = std::round(neg_range);
            pos_range = std::round(pos_range);
            // double max_dev = std::max(neg_range, pos_range);
            // if (mean > 0.0) err_percent = 100.0 * max_dev / mean;
            if (gt_depth_ > 0.0) err_percent = 100.0 * std::abs(mean - gt_depth_) / gt_depth_;

            char buf[256];
            std::vector<std::string> lines;
            // GT (m)
            snprintf(buf, sizeof(buf), "GT: %.3f m", gt_depth_ / 1000.0);
            lines.push_back(std::string(buf));
            // mean (m)
            snprintf(buf, sizeof(buf), "Mean: %.3f m", mean / 1000.0);
            lines.push_back(std::string(buf));
            // +/- ranges (m)
            snprintf(buf, sizeof(buf), "Range: +%.3f / -%.3f m", pos_range / 1000.0, neg_range / 1000.0);
            lines.push_back(std::string(buf));
            // percent error
            snprintf(buf, sizeof(buf), "Err[ (Mean-GT)/GT ]: %.3f %%", err_percent);
            lines.push_back(std::string(buf));

            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "=> GT: %.3f m, mean: %.3f m, range: "
                                 "+%.3f / -%.3f m, "
                                 "err[ (Mean-GT)/GT ]: "
                                 "%.3f %%",
                                 gt_depth_ / 1000.0, mean / 1000.0, pos_range / 1000.0, neg_range / 1000.0,
                                 err_percent);

            for (size_t i = 0; i < lines.size(); ++i)
              cv::putText(visual_img, lines[i],
                          cv::Point(x - roi_size_ * 2, y - (lines.size() - i) * roi_size_ * 3 + pub_data->depth.rows),
                          cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 255), 2);

            cv::rectangle(visual_img, cv::Point(x - roi_size_ / 2, y - roi_size_ / 2),
                          cv::Point(x + roi_size_ / 2, y + roi_size_ / 2), cv::Scalar(0, 0, 255), 2);
            cv::rectangle(visual_img, cv::Point(x - roi_size_ / 2, y - roi_size_ / 2 + pub_data->depth.rows),
                          cv::Point(x + roi_size_ / 2, y + roi_size_ / 2 + pub_data->depth.rows), cv::Scalar(0, 0, 255),
                          2);
          }
        }
      }
    }
  }
  // ===================================== render performance metrics =======================================
  if (render_perf_ && !save_result_flag_ && !do_save_result_once_) {
    std::stringstream perf_text;
    perf_text << "FPS: " << std::fixed << std::setprecision(2) << pub_data->fps << " Latency: " << pub_data->latency
              << "ms CPU: " << pub_data->cpu_usage << "% BPU: " << pub_data->bpu_usage << "%";
    static int text_height = 0;
    if (text_height == 0) {
      int baseline = 0;
      cv::Size textSize = cv::getTextSize(perf_text.str(), cv::FONT_HERSHEY_SIMPLEX, font_scale, 2, &baseline);
      text_height = textSize.height + baseline;
    }
    cv::putText(visual_img, perf_text.str(), cv::Point(10, text_height), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                CV_RGB(0, 0, 255), 2);
    if (!top_is_left_) {
      cv::putText(visual_img, "top is not left image,", cv::Point(10, text_height * 2), cv::FONT_HERSHEY_SIMPLEX,
                  font_scale, CV_RGB(0, 0, 255), 2);
      cv::putText(visual_img, "please swap the mipi cables,", cv::Point(10, text_height * 3), cv::FONT_HERSHEY_SIMPLEX,
                  font_scale, CV_RGB(0, 0, 255), 2);
      cv::putText(visual_img, "or set the mipi_channel and mipi_channel2 parameters.", cv::Point(10, text_height * 4),
                  cv::FONT_HERSHEY_SIMPLEX, font_scale, CV_RGB(0, 0, 255), 2);
    }
  }
  // ===================================== render ground angle ===============================================
  if (ground_angle_enable_) {
    int cx = ground_roi_center_x_;
    int cy = ground_roi_center_y_;

    // Default ROI: image center x, lower area y.
    if (cx < 0) cx = width / 2;
    if (cy < 0) cy = height - (ground_roi_height_ / 2) - 5;

    int roi_w = std::min(ground_roi_width_, width);
    int roi_h = std::min(ground_roi_height_, height);

    int x0 = cx - roi_w / 2;
    int y0 = cy - roi_h / 2;

    x0 = std::max(0, std::min(x0, width - roi_w));
    y0 = std::max(0, std::min(y0, height - roi_h));

    cv::Rect ground_roi(x0, y0, roi_w, roi_h);

    double ground_angle_deg = 0.0;
    bool ok = compute_ground_xz_angle(pub_data->depth, *camera_intrinsic_, ground_roi, ground_roi_min_valid_points_,
                                      ground_angle_deg);

    // Draw ROI on upper left image and lower depth visualization.
    cv::rectangle(visual_img, ground_roi, CV_RGB(0, 255, 0), 2);

    cv::Rect ground_roi_bottom(ground_roi.x, ground_roi.y + left_bgr.rows, ground_roi.width, ground_roi.height);
    cv::rectangle(visual_img, ground_roi_bottom, CV_RGB(0, 255, 0), 2);

    std::stringstream ss;
    if (ok) {
      ss << "Ground-XZ angle: " << std::fixed << std::setprecision(2) << ground_angle_deg << " deg";
    } else {
      ss << "Ground-XZ angle: invalid";
    }

    int text_y = std::max(30, ground_roi.y - 10);
    cv::putText(visual_img, ss.str(), cv::Point(ground_roi.x, text_y), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                CV_RGB(0, 255, 0), 2);
    cv::putText(visual_img, ss.str(), cv::Point(ground_roi_bottom.x, ground_roi_bottom.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, CV_RGB(0, 255, 0), 2);
  }

  // ===================================== publish visual image ============================================
  pub_data->visual_img = visual_img;
  // Convert cv::Mat to sensor_msgs::msg::Image
  auto visual_msg = std::make_shared<sensor_msgs::msg::Image>();
  visual_msg->header = pub_data->header;
  visual_msg->header.frame_id = stereonet_frame_id_;
  visual_msg->height = visual_img.rows;
  visual_msg->width = visual_img.cols;
  visual_msg->encoding = "bgr8";
  visual_msg->is_bigendian = false;
  visual_msg->step = visual_img.cols * visual_img.elemSize();
  size_t size = visual_msg->step * visual_msg->height;
  visual_msg->data.resize(size);
  std::memcpy(visual_msg->data.data(), visual_img.data, size);

  visual_image_pub_->publish(*visual_msg);
}

void StereoNetNode::publish_epipolar_image(const std::shared_ptr<PubData> &pub_data) {
  cv::Mat visual_img;
  if (epipolar_img_ == "origin") {
    cv::Mat origin_left_img = pub_data->origin_left.clone();
    cv::Mat origin_right_img = pub_data->origin_right.clone();
    if (origin_left_img.empty() || origin_right_img.empty()) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "\033[31m=> epipolar image is empty\033[0m");
      return;
    }
    EpipolarAlign::check_epipolar_alignment(origin_left_img, origin_right_img,
                                            cv::Size(chessboard_per_rows_, chessboard_per_cols_),
                                            chessboard_square_size_, orignal_camera_intrinsic_, visual_img);
  } else {
    cv::Mat left_img = pub_data->left_bgr;
    cv::Mat right_img = pub_data->right_bgr;
    if (left_img.empty() || right_img.empty()) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "\033[31m=> epipolar image is empty\033[0m");
      return;
    }
    EpipolarAlign::check_epipolar_alignment(left_img, right_img, cv::Size(chessboard_per_rows_, chessboard_per_cols_),
                                            chessboard_square_size_, camera_intrinsic_, visual_img);
  }
  // ===================================== publish visual image ============================================
  if (visual_img.empty()) return;
  pub_data->visual_img = visual_img;
  // Convert cv::Mat to sensor_msgs::msg::Image
  auto visual_msg = std::make_shared<sensor_msgs::msg::Image>();
  visual_msg->header = pub_data->header;
  visual_msg->header.frame_id = stereonet_frame_id_;
  visual_msg->height = visual_img.rows;
  visual_msg->width = visual_img.cols;
  visual_msg->encoding = "bgr8";
  visual_msg->is_bigendian = false;
  visual_msg->step = visual_img.cols * visual_img.elemSize();
  size_t size = visual_msg->step * visual_msg->height;
  visual_msg->data.resize(size);
  std::memcpy(visual_msg->data.data(), visual_img.data, size);

  visual_image_pub_->publish(*visual_msg);
}

void StereoNetNode::publish_feature_epipolar_image(const std::shared_ptr<PubData> &pub_data) {
  cv::Mat visual_img;
  if (epipolar_img_ == "origin") {
    cv::Mat origin_left_img = pub_data->origin_left.clone();
    cv::Mat origin_right_img = pub_data->origin_right.clone();
    if (origin_left_img.empty() || origin_right_img.empty()) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "\033[31m=> epipolar image is empty\033[0m");
      return;
    }
    FeatureEpipolarAlign::check_epipolar_alignment(origin_left_img, origin_right_img, orignal_camera_intrinsic_,
                                                   visual_img);
  } else {
    cv::Mat left_img = pub_data->left_bgr;
    cv::Mat right_img = pub_data->right_bgr;
    if (left_img.empty() || right_img.empty()) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "\033[31m=> epipolar image is empty\033[0m");
      return;
    }
    FeatureEpipolarAlign::check_epipolar_alignment(left_img, right_img, camera_intrinsic_, visual_img);
  }
  // ===================================== publish visual image ============================================
  if (visual_img.empty()) return;
  pub_data->visual_img = visual_img;
  // Convert cv::Mat to sensor_msgs::msg::Image
  auto visual_msg = std::make_shared<sensor_msgs::msg::Image>();
  visual_msg->header = pub_data->header;
  visual_msg->header.frame_id = stereonet_frame_id_;
  visual_msg->height = visual_img.rows;
  visual_msg->width = visual_img.cols;
  visual_msg->encoding = "bgr8";
  visual_msg->is_bigendian = false;
  visual_msg->step = visual_img.cols * visual_img.elemSize();
  size_t size = visual_msg->step * visual_msg->height;
  visual_msg->data.resize(size);
  std::memcpy(visual_msg->data.data(), visual_img.data, size);

  visual_image_pub_->publish(*visual_msg);
}

void StereoNetNode::save_result(const std::shared_ptr<PubData> &pub_data) {
  if (!rclcpp::ok()) return;
  if (!save_result_flag_) return;
  // save enough images
  if (save_total_ > 0 && pub_data->count / save_freq_ >= (save_total_ - 1)) {
    // last one
    RCLCPP_WARN(this->get_logger(), "\033[32m=> save total %d images, stop saving\033[0m", save_total_);
    save_result_flag_ = false;
  }
  // do save if count is divisible by save_freq
  bool do_save = false;
  if (pub_data->count % save_freq_ == 0) do_save = true;
  if (!do_save) return;

  int current_count = pub_data->count / save_freq_;
  // save camera intrinsic
  if (current_count == 0) {
    std::string intrinsic_path = fs::path(save_dir_) / fs::path("camera_intrinsic.txt");
    std::ofstream ofs(intrinsic_path);
    if (ofs.is_open()) {
      ofs << std::fixed << std::setprecision(6);
      ofs << "# fx fy cx cy baseline(m)" << std::endl;
      ofs << camera_intrinsic_->fx << " " << camera_intrinsic_->fy << " " << camera_intrinsic_->cx << " "
          << camera_intrinsic_->cy << " " << camera_intrinsic_->baseline << std::endl;
      ofs.close();
      RCLCPP_WARN(this->get_logger(), "\033[32m=> save camera intrinsic to %s\033[0m", intrinsic_path.c_str());
    }
    if (save_origin_flag_) {
      std::string intrinsic_path = fs::path(save_dir_) / fs::path("origin_camera_intrinsic.txt");
      std::ofstream ofs(intrinsic_path);
      if (ofs.is_open()) {
        ofs << std::fixed << std::setprecision(6);
        ofs << "# fx fy cx cy baseline(m)" << std::endl;
        ofs << orignal_camera_intrinsic_->fx << " " << orignal_camera_intrinsic_->fy << " "
            << orignal_camera_intrinsic_->cx << " " << orignal_camera_intrinsic_->cy << " "
            << orignal_camera_intrinsic_->baseline << std::endl;
        ofs.close();
        RCLCPP_WARN(this->get_logger(), "\033[32m=> save origin camera intrinsic to %s\033[0m", intrinsic_path.c_str());
      }
    }
  }

  // save images & point cloud
  std::stringstream ss;
  ss << std::setw(6) << std::setfill('0') << current_count << "_";

  if (save_stereo_flag_) {
    std::string left_image_path = fs::path(save_dir_) / fs::path(ss.str() + "left.png");
    std::string right_image_path = fs::path(save_dir_) / fs::path(ss.str() + "right.png");
    cv::imwrite(left_image_path, pub_data->left_bgr);
    cv::imwrite(right_image_path, pub_data->right_bgr);
  }
  if (save_disp_flag_) {
    std::string disp_image_path = fs::path(save_dir_) / fs::path(ss.str() + "disp.pfm");
    cv::imwrite(disp_image_path, pub_data->disp);
  }
  if (save_uncert_flag_ && !pub_data->uncert.empty()) {
    std::string uncert_image_path = fs::path(save_dir_) / fs::path(ss.str() + "uncert.pfm");
    cv::imwrite(uncert_image_path, pub_data->uncert);
  }
  if (save_depth_flag_) {
    std::string depth_image_path = fs::path(save_dir_) / fs::path(ss.str() + "depth.png");
    cv::imwrite(depth_image_path, pub_data->depth);
  }
  if (save_visual_flag_ && !pub_data->visual_img.empty()) {
    std::string visual_image_path = fs::path(save_dir_) / fs::path(ss.str() + "visual.jpg");
    cv::imwrite(visual_image_path, pub_data->visual_img);
  }
  if (save_pcd_flag_ && pub_data->pointcloud && !pub_data->pointcloud->points.empty()) {
    std::string pointcloud_path = fs::path(save_dir_) / fs::path(ss.str() + "pointcloud.pcd");
    pcl::io::savePCDFileBinary(pointcloud_path, *(pub_data->pointcloud));
  }
  if (save_origin_flag_ && !pub_data->origin_left.empty() && !pub_data->origin_right.empty()) {
    std::string origin_left_image_path = fs::path(save_dir_) / fs::path(ss.str() + "origin_L.png");
    std::string origin_right_image_path = fs::path(save_dir_) / fs::path(ss.str() + "origin_R.png");
    cv::imwrite(origin_left_image_path, pub_data->origin_left);
    cv::imwrite(origin_right_image_path, pub_data->origin_right);
  }

  RCLCPP_WARN(this->get_logger(), "\033[32m=> save result to %s, pub count: %d, save count: %d\033[0m",
              save_dir_.c_str(), pub_data->count, current_count);
}

void StereoNetNode::save_result_once(const std::shared_ptr<PubData> &pub_data) {
  int current_count = pub_data->count;
  // save camera intrinsic
  {
    std::string intrinsic_path = fs::path(save_dir_) / fs::path("camera_intrinsic.txt");
    std::ofstream ofs(intrinsic_path);
    if (ofs.is_open()) {
      ofs << std::fixed << std::setprecision(6);
      ofs << "# fx fy cx cy baseline(m)" << std::endl;
      ofs << camera_intrinsic_->fx << " " << camera_intrinsic_->fy << " " << camera_intrinsic_->cx << " "
          << camera_intrinsic_->cy << " " << camera_intrinsic_->baseline << std::endl;
      ofs.close();
    }
  }
  {
    std::string origin_camera_intrinsic_path = fs::path(save_dir_) / fs::path("origin_camera_intrinsic.txt");
    std::ofstream ofs(origin_camera_intrinsic_path);
    if (ofs.is_open()) {
      ofs << std::fixed << std::setprecision(6);
      ofs << "# fx fy cx cy baseline(m)" << std::endl;
      ofs << orignal_camera_intrinsic_->fx << " " << orignal_camera_intrinsic_->fy << " "
          << orignal_camera_intrinsic_->cx << " " << orignal_camera_intrinsic_->cy << " "
          << orignal_camera_intrinsic_->baseline << std::endl;
      ofs.close();
    }
  }

  // save images & point cloud
  std::stringstream ss;
  ss << std::setw(6) << std::setfill('0') << current_count << "_";

  std::string left_image_path = fs::path(save_dir_) / fs::path(ss.str() + "left.png");
  std::string right_image_path = fs::path(save_dir_) / fs::path(ss.str() + "right.png");
  cv::imwrite(left_image_path, pub_data->left_bgr);
  cv::imwrite(right_image_path, pub_data->right_bgr);
  std::string disp_image_path = fs::path(save_dir_) / fs::path(ss.str() + "disp.pfm");
  cv::imwrite(disp_image_path, pub_data->disp);
  if (!pub_data->uncert.empty()) {
    std::string uncert_image_path = fs::path(save_dir_) / fs::path(ss.str() + "uncert.pfm");
    cv::imwrite(uncert_image_path, pub_data->uncert);
  }
  std::string depth_image_path = fs::path(save_dir_) / fs::path(ss.str() + "depth.png");
  cv::imwrite(depth_image_path, pub_data->depth);
  std::string visual_image_path = fs::path(save_dir_) / fs::path(ss.str() + "visual.jpg");
  cv::imwrite(visual_image_path, pub_data->visual_img);
  if (pub_data->pointcloud && !pub_data->pointcloud->points.empty()) {
    std::string pointcloud_path = fs::path(save_dir_) / fs::path(ss.str() + "pointcloud.pcd");
    pcl::io::savePCDFileBinary(pointcloud_path, *(pub_data->pointcloud));
  }
  if (!pub_data->origin_left.empty() && !pub_data->origin_right.empty()) {
    std::string origin_left_image_path = fs::path(save_dir_) / fs::path(ss.str() + "origin_L.png");
    std::string origin_right_image_path = fs::path(save_dir_) / fs::path(ss.str() + "origin_R.png");
    cv::imwrite(origin_left_image_path, pub_data->origin_left);
    cv::imwrite(origin_right_image_path, pub_data->origin_right);
  }
  if (!pub_data->left_bgr.empty()) {
    cv::Mat gray;
    cv::cvtColor(pub_data->left_bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat gray_f;
    gray.convertTo(gray_f, CV_32F, 1.0 / 255.0);

    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(gray_f, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray_f, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);

    cv::imwrite((fs::path(save_dir_) / fs::path(ss.str() + "left_grad_x.pfm")).string(), grad_x);
    cv::imwrite((fs::path(save_dir_) / fs::path(ss.str() + "left_grad_y.pfm")).string(), grad_y);
    cv::imwrite((fs::path(save_dir_) / fs::path(ss.str() + "left_grad.pfm")).string(), grad_mag);
  }

  RCLCPP_WARN(this->get_logger(), "\033[32m=> save result to %s, pub count: %d, save count: %d\033[0m",
              save_dir_.c_str(), pub_data->count, current_count);
}

void StereoNetNode::infer_offline() {
  auto img_paths = FileUtils::find_pairs(local_image_dir_);
  RCLCPP_WARN(this->get_logger(), "\033[32m=> found %zu image pairs in %s\033[0m", img_paths.size(),
              local_image_dir_.c_str());

  // read camera intrinsic
  if (calib_method_ == "none") {
    std::string camera_intrinsic_path = fs::path(local_image_dir_) / fs::path("camera_intrinsic.txt");
    if (fs::exists(camera_intrinsic_path)) {
      bool read_success =
          FileUtils::read_camera_intrinsic(camera_intrinsic_path, camera_intrinsic_->fx, camera_intrinsic_->fy,
                                           camera_intrinsic_->cx, camera_intrinsic_->cy, camera_intrinsic_->baseline);
      if (read_success) {
        RCLCPP_WARN_ONCE(
            this->get_logger(),
            "\033[31m=> read camera intrinsic from %s, [fx, fy, cx, cy, baseline(m)] : [%f, %f, %f, %f, %f]\033[0m",
            camera_intrinsic_path.c_str(), camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx,
            camera_intrinsic_->cy, camera_intrinsic_->baseline);
      } else {
        RCLCPP_ERROR(this->get_logger(), "\033[31m=> read camera intrinsic from %s failed\033[0m",
                     camera_intrinsic_path.c_str());
        rclcpp::shutdown();
      }
    } else if (camera_intrinsic_->is_valid()) {
      RCLCPP_WARN_ONCE(
          this->get_logger(),
          "\033[31m=> use camera intrinsic from launch params, [fx, fy, cx, cy, baseline(m), doffs] : [%f, %f, "
          "%f, %f, %f, %f]\033[0m",
          camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy,
          camera_intrinsic_->baseline, camera_intrinsic_->doffs);
    } else {
      RCLCPP_ERROR(this->get_logger(),
                   "\033[31m=> camera intrinsic is not set, please provide camera_intrinsic.txt in %s or set the "
                   "camera parameters [camera_fx, camera_fy, camera_cx, camera_cy, baseline] in the launch file\033[0m",
                   local_image_dir_.c_str());
      rclcpp::shutdown();
    }
  }

  int cnt = 0;
  for (auto &img_pair : img_paths) {
    if (rclcpp::ok() == false) break;
    RCLCPP_WARN_STREAM(this->get_logger(), "\033[33m=> processing image pair: [" << img_pair.first << ", "
                                                                                 << img_pair.second << "]\033[0m");
    cv::Mat left_img_bgr = cv::imread(img_pair.first, cv::IMREAD_COLOR);
    cv::Mat right_img_bgr = cv::imread(img_pair.second, cv::IMREAD_COLOR);
    if (left_img_bgr.empty() || right_img_bgr.empty()) {
      RCLCPP_ERROR(this->get_logger(), "=> failed to read image pair: %s and %s", img_pair.first.c_str(),
                   img_pair.second.c_str());
      continue;
    }
    int model_input_w = 0, model_input_h = 0;
    stereonet_process_->get_model_input_size(model_input_w, model_input_h);
    if (calib_method_ == "custom") {
      stereo_rectifier_->build_undistmap(left_img_bgr.cols, left_img_bgr.rows, model_input_w, model_input_h);
      stereo_rectifier_->get_intrinsic(camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx,
                                       camera_intrinsic_->cy, camera_intrinsic_->baseline);
    }

    cv::Mat combine_img_bgr;
    cv::vconcat(left_img_bgr, right_img_bgr, combine_img_bgr);
    cv::Mat combine_img_nv12;
    ImgConvertUtils::bgr_mat_to_nv12_mat(combine_img_bgr, combine_img_nv12);

    auto stereo_msg = std::make_shared<sensor_msgs::msg::Image>();
    stereo_msg->header.stamp = this->get_clock()->now();
    stereo_msg->header.frame_id = stereonet_frame_id_;
    stereo_msg->height = combine_img_bgr.rows;
    stereo_msg->width = combine_img_bgr.cols;
    stereo_msg->encoding = "nv12";
    stereo_msg->is_bigendian = false;
    stereo_msg->step = combine_img_bgr.cols; // Y plane step
    size_t size = combine_img_bgr.cols * combine_img_bgr.rows * 3 / 2;
    stereo_msg->data.resize(size);
    std::memcpy(stereo_msg->data.data(), combine_img_nv12.data, size);
    while (input_image_queue_.size_approx() >= 10) {
      // if task is not finished, wait
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    input_image_queue_.enqueue(stereo_msg);
    std::this_thread::sleep_for(std::chrono::milliseconds(image_sleep_));
    cnt++;
  }

  RCLCPP_WARN(this->get_logger(), "\033[32m=> all %d images in %s have been processed\033[0m", cnt,
              local_image_dir_.c_str());
}

bool StereoNetNode::judge_top_is_left_by_ORB(const cv::Mat &top_bgr, const cv::Mat &bottom_bgr) {
  // convert to gray
  cv::Mat top_gray, bottom_gray;
  if (top_bgr.channels() == 3) {
    cv::cvtColor(top_bgr, top_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bottom_bgr, bottom_gray, cv::COLOR_BGR2GRAY);
  } else {
    top_gray = top_bgr;
    bottom_gray = bottom_bgr;
  }
  // orb detect
  cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);
  std::vector<cv::KeyPoint> kp_top, kp_bottom;
  cv::Mat des_top, des_bottom;
  orb->detectAndCompute(top_gray, cv::Mat(), kp_top, des_top);
  orb->detectAndCompute(bottom_gray, cv::Mat(), kp_bottom, des_bottom);
  if (des_top.empty() || des_bottom.empty()) {
    // orb detect failed
    return true;
  }
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<cv::DMatch> matches;
  matcher.match(des_top, des_bottom, matches);

  // calculate positive and negative disparity
  int positive_disp = 0;
  int negative_disp = 0;
  for (const auto &m : matches) {
    const auto &pt_top = kp_top[m.queryIdx].pt;
    const auto &pt_bottom = kp_bottom[m.trainIdx].pt;
    float disparity = pt_top.x - pt_bottom.x;
    if (disparity > 1.0)
      positive_disp++;
    else if (disparity < -1.0)
      negative_disp++;
  }

  // judge: positive disparity > negative disparity → top is left image
  if (positive_disp > negative_disp)
    return true;
  else
    return false;
}

} // namespace stereonet

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(stereonet::StereoNetNode)
