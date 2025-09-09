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
                     "=> ===================== init " << this->get_name() << "=====================" << std::endl);
  this->declare_parameter<std::string>("stereonet_model_file_path", "");
  stereonet_model_file_path_ = this->get_parameter("stereonet_model_file_path").as_string();

  this->declare_parameter<std::string>("stereo_image_topic", "/image_combine_raw");
  stereo_image_topic_ = this->get_parameter("stereo_image_topic").as_string();
  this->declare_parameter<std::string>("camera_info_topic", "/image_right_raw/camera_info");
  camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();

  this->declare_parameter<std::string>("depth_image_topic", "~/stereonet_depth");
  depth_image_topic_ = this->get_parameter("depth_image_topic").as_string();
  this->declare_parameter<std::string>("depth_camera_info_topic", "~/stereonet_depth/camera_info");
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
  depth_camera_info_topic_ = this->get_parameter("depth_camera_info_topic").as_string();
  this->declare_parameter<std::string>("pointcloud2_topic", "~/stereonet_pointcloud2");
  pointcloud2_topic_ = this->get_parameter("pointcloud2_topic").as_string();
  this->declare_parameter<std::string>("visual_image_topic", "~/stereonet_visual");
  visual_image_topic_ = this->get_parameter("visual_image_topic").as_string();
  this->declare_parameter<bool>("render_perf", true);
  render_perf_ = this->get_parameter("render_perf").as_bool();
  if (render_perf_) {
    performance_writer::Get();
  }

  this->declare_parameter<std::string>("postprocess", "convex_upsampling");
  postprocess_ = this->get_parameter("postprocess").as_string();
  auto is_valid_postprocess = [](const std::string &postprocess) {
    return postprocess == "convex_upsampling" || postprocess == "convex_upsampling_with_uncert" ||
           postprocess == "convex_upsampling_with_interp";
  };
  if (!is_valid_postprocess(postprocess_)) {
    RCLCPP_ERROR(this->get_logger(), "\033[32m=> postprocess parameter invalid, should be one of [convex_upsampling, "
                                     "convex_upsampling_with_uncert, convex_upsampling_with_interp]\033[0m");
    rclcpp::shutdown();
  }

  this->declare_parameter<double>("uncertainty_th", 0.0);
  uncertainty_th_ = this->get_parameter("uncertainty_th").as_double();

  camera_intrinsic_ = std::make_shared<CameraIntrinsic>();
  this->declare_parameter<double>("camera_fx", 0.0);
  this->declare_parameter<double>("camera_fy", 0.0);
  this->declare_parameter<double>("camera_cx", 0.0);
  this->declare_parameter<double>("camera_cy", 0.0);
  this->declare_parameter<double>("baseline", 0.0);
  camera_intrinsic_->fx = this->get_parameter("camera_fx").as_double();
  camera_intrinsic_->fy = this->get_parameter("camera_fy").as_double();
  camera_intrinsic_->cx = this->get_parameter("camera_cx").as_double();
  camera_intrinsic_->cy = this->get_parameter("camera_cy").as_double();
  camera_intrinsic_->baseline = this->get_parameter("baseline").as_double();

  this->declare_parameter<double>("pointcloud_height_min", -5.0);
  this->declare_parameter<double>("pointcloud_height_max", 5.0);
  this->declare_parameter<double>("pointcloud_depth_max", 5.0);
  pointcloud_height_min_ = this->get_parameter("pointcloud_height_min").as_double();
  pointcloud_height_max_ = this->get_parameter("pointcloud_height_max").as_double();
  pointcloud_depth_max_ = this->get_parameter("pointcloud_depth_max").as_double();

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
  if (infer_thread_num_ <= 0) infer_thread_num_ = 1;
  if (infer_thread_num_ > 4) infer_thread_num_ = 4;
  if (save_thread_num_ <= 0) save_thread_num_ = 1;
  if (save_thread_num_ > 8) save_thread_num_ = 8;

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
    }
    if (image_sleep_ < 0) image_sleep_ = 0;
    infer_thread_num_ = 1;
    calib_method_ = "none";
    render_perf_ = false;
  }

  this->declare_parameter<bool>("speckle_filter_enable", false);
  speckle_filter_enable_ = this->get_parameter("speckle_filter_enable").as_bool();
  this->declare_parameter<int>("max_speckle_size", 100);
  max_speckle_size_ = this->get_parameter("max_speckle_size").as_int();
  this->declare_parameter<double>("max_disp_diff", 1.0);
  max_disp_diff_ = this->get_parameter("max_disp_diff").as_double();

  RCLCPP_WARN_STREAM(this->get_logger(),
                     std::endl
                         << "stereonet_model_file_path: " << stereonet_model_file_path_ << std::endl
                         << "stereo_image_topic: " << stereo_image_topic_ << std::endl
                         << "camera_info_topic: " << camera_info_topic_ << std::endl
                         << "depth_image_topic: " << depth_image_topic_ << std::endl
                         << "depth_camera_info_topic: " << depth_camera_info_topic_ << std::endl
                         << "rectify_left_image_topic: " << rectify_left_image_topic_ << std::endl
                         << "rectify_right_image_topic: " << rectify_right_image_topic_ << std::endl
                         << "publish_rectify_bgr: " << publish_rectify_bgr_ << std::endl
                         << "origin_left_image_topic: " << origin_left_image_topic_ << std::endl
                         << "origin_right_image_topic: " << origin_right_image_topic_ << std::endl
                         << "pointcloud2_topic: " << pointcloud2_topic_ << std::endl
                         << "visual_image_topic: " << visual_image_topic_ << std::endl
                         << "render_perf: " << render_perf_ << std::endl
                         << "postprocess: " << postprocess_ << std::endl
                         << "uncertainty_th: " << uncertainty_th_ << std::endl
                         << "[camera_fx, camera_fy, camera_cx, camera_cy, baseline]: [" << camera_intrinsic_->fx << ", "
                         << camera_intrinsic_->fy << ", " << camera_intrinsic_->cx << ", " << camera_intrinsic_->cy
                         << ", " << camera_intrinsic_->baseline << "(m)]" << std::endl
                         << "[pointcloud_height_min, pointcloud_height_max, pointcloud_depth_max]: ["
                         << pointcloud_height_min_ << "(m), " << pointcloud_height_max_ << "(m), "
                         << pointcloud_depth_max_ << "(m)]" << std::endl
                         << "[use_local_image_flag, local_image_dir, image_sleep]: [" << use_local_image_flag_ << ", "
                         << local_image_dir_ << ", " << image_sleep_ << "]" << std::endl
                         << "[save_result_flag, save_dir, save_freq, save_total]: [" << save_result_flag_ << ", "
                         << save_dir_ << ", " << save_freq_ << ", " << save_total_ << "]" << std::endl
                         << "[calib_method, stereo_calib_file_path]: [" << calib_method_ << ", "
                         << stereo_calib_file_path_ << "]" << std::endl
                         << "[speckle_filter_enable, max_speckle_size, max_disp_diff]: [" << speckle_filter_enable_
                         << ", " << max_speckle_size_ << ", " << max_disp_diff_ << "]" << std::endl
                         << "[infer_thread_num, save_thread_num]: [" << infer_thread_num_ << ", " << save_thread_num_
                         << "]" << std::endl
                         << "=> ==================================================================" << std::endl);

  if (save_result_flag_) {
    if (!fs::exists(save_dir_)) {
      if (fs::create_directories(save_dir_)) {
        RCLCPP_INFO(this->get_logger(), "\033[32m=> create save_dir: %s\033[0m", save_dir_.c_str());
      } else {
        RCLCPP_ERROR(this->get_logger(), "\033[31m=> create save_dir: %s failed\033[0m", save_dir_.c_str());
        rclcpp::shutdown();
      }
    }
    save_thread_pool_ptr_ = std::make_unique<BS::thread_pool<>>(save_thread_num_);
    if (save_freq_ <= 0) save_freq_ = 1;
  }
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
        stereo_image_topic_, 10, std::bind(&StereoNetNode::stereo_image_callback, this, std::placeholders::_1));
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, 10, std::bind(&StereoNetNode::camera_info_callback, this, std::placeholders::_1));
  }

  // Set up publishers
  visual_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(visual_image_topic_, 10);
  depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(depth_image_topic_, 10);
  depth_camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(depth_camera_info_topic_, 10);
  pointcloud2_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(pointcloud2_topic_, 10);
  rectify_left_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(rectify_left_image_topic_, 10);
  rectify_right_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(rectify_right_image_topic_, 10);
  origin_left_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(origin_left_image_topic_, 10);
  origin_right_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(origin_right_image_topic_, 10);
}

void StereoNetNode::set_dnn_model() {
  stereonet_process_ = std::make_shared<StereonetProcess>(this->get_logger());
  int ret_code = stereonet_process_->init(stereonet_model_file_path_);
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
}

void StereoNetNode::publish_static_tf() {
  static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
  geometry_msgs::msg::TransformStamped t;
  t.header.stamp = now();
  t.header.frame_id = "camera_link";
  t.child_frame_id = "camera_depth_frame";

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

  while (input_image_queue_.size_approx() >= 1) {
    sensor_msgs::msg::Image::SharedPtr drop;
    input_image_queue_.try_dequeue(drop);
  }
  input_image_queue_.enqueue(msg);
}

void StereoNetNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  if (sub_camera_info_flag_) return;
  // only subscribe once
  camera_intrinsic_->fx = msg->p[0];
  camera_intrinsic_->fy = msg->p[5];
  camera_intrinsic_->cx = msg->p[2];
  camera_intrinsic_->cy = msg->p[6];
  camera_intrinsic_->baseline = msg->p[3] / camera_intrinsic_->fx;

  if (camera_intrinsic_->baseline > 1) camera_intrinsic_->baseline *= 0.001f; // convert mm to m

  RCLCPP_WARN(this->get_logger(),
              "\033[31m=> sub rectified [fx, fy, cx, cy, baseline(m)] : [%f, %f, %f, %f, %f]\033[0m",
              camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy,
              camera_intrinsic_->baseline);
  sub_camera_info_flag_ = true;
}

void StereoNetNode::infer_function(const int &thread_id) {
  while (rclcpp::ok()) {
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

      // ================================== Inference ==================================
      cv::Mat disp, uncert;
      stereonet_process_->forward(rectify_left_img_data, rectify_right_img_data, uncertainty_th_, postprocess_, disp,
                                  uncert);
      cv::Mat depth;
      {
        ScopeProcessTime t(this->get_logger(), "disp_to_depth");
        if (camera_intrinsic_->is_valid()) {
          StereonetProcess::disp_to_depth(disp, depth, camera_intrinsic_->fx, camera_intrinsic_->baseline);
        } else {
          if (calib_method_ == "none") {
            RCLCPP_ERROR(this->get_logger(),
                         "\033[31m=> unable to receive topic %s to obtain camera intrinsic parameters, and the camera "
                         "intrinsic parameters [camera_fx, camera_fy, camera_cx, camera_cy, baseline] are not manually "
                         "set, please confirm whether the topic is correct or manually set the camera intrinsic "
                         "parameters. when calib_method is none\033[0m",
                         camera_info_topic_.c_str());
          } else if (calib_method_ == "custom") {
            RCLCPP_ERROR(
                this->get_logger(),
                "\033[31m=> calib_method is custom, camera intrinsic should be set from stereo_calib_file_path: "
                "%s\033[0m",
                stereo_calib_file_path_.c_str());
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

      // ================================== Publish ====================================
      auto pub_data = std::make_shared<PubData>();
      pub_data->timestamp = static_cast<uint64_t>(stereo_msg->header.stamp.sec) * 1'000'000'000 +
                            static_cast<uint64_t>(stereo_msg->header.stamp.nanosec);
      pub_data->header = stereo_msg->header;
      pub_data->origin_stereo_msg = stereo_msg;
      pub_data->disp = disp;
      pub_data->uncert = uncert;
      pub_data->depth = depth;

      if (render_perf_) {
        auto now = this->get_clock()->now();
        auto latency = (now - stereo_msg->header.stamp).seconds() * 1000;
        // if latency > 1000ms, maybe the time stamp is not correct, set latency to 0
        pub_data->latency = latency > 1000 ? 0 : latency;
        performance_writer::Get()->record_performance(pub_data->latency);
        pub_data->fps = performance_writer::Get()->get_fps();
        pub_data->cpu_usage = performance_writer::Get()->get_cpu_usage();
        pub_data->bpu_usage = performance_writer::Get()->get_bpu_usage();
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "=> fps: %d, cpu_usage: %d%%, bpu_usage: %d%%", pub_data->fps, pub_data->cpu_usage,
                             pub_data->bpu_usage);
      }

      pub_data->rectify_left_img_data = rectify_left_img_data;
      pub_data->rectify_right_img_data = rectify_right_img_data;

      pub_data_queue_.put(pub_data->timestamp, pub_data);
    }
  }
}

void StereoNetNode::preprocess(const sensor_msgs::msg::Image::SharedPtr &stereo_msg, const int &model_input_w,
                               const int &model_input_h, std::vector<uint8_t> &left_img_data,
                               std::vector<uint8_t> &right_img_data) {
  if (stereo_msg->encoding == "nv12") {
    int single_img_w = stereo_msg->width;
    int single_img_h = stereo_msg->height / 2;
    size_t single_nv12_size = single_img_w * single_img_h * 3 / 2;

    left_img_data.resize(single_nv12_size);
    right_img_data.resize(single_nv12_size);

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
        if (camera_info_updated_ == false) {
          camera_info_updated_ = true;
          camera_intrinsic_->cx = camera_intrinsic_->cx * model_input_w / single_img_w;
          camera_intrinsic_->cy = camera_intrinsic_->cy * model_input_h / single_img_h;
          camera_intrinsic_->fx = camera_intrinsic_->fx * model_input_w / single_img_w;
          camera_intrinsic_->fy = camera_intrinsic_->fy * model_input_h / single_img_h;
          RCLCPP_WARN(this->get_logger(),
                      "\033[31m=> after resize, update camera intrinsic: fx: %f, fy: %f, cx: %f, cy: %f\033[0m",
                      camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy);
        }
        single_img_w = model_input_w;
        single_img_h = model_input_h;
        ImgConvertUtils::bgr_mat_to_nv12(left_bgr, left_img_data.data());
        ImgConvertUtils::bgr_mat_to_nv12(right_bgr, right_img_data.data());
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
                      "\033[31m=> after resize, update camera intrinsic: fx: %f, fy: %f, cx: %f, cy: %f\033[0m",
                      camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy);
        }
        single_img_w = model_input_w;
        single_img_h = model_input_h;
      }

      size_t single_nv12_size = single_img_w * single_img_h * 3 / 2;
      left_img_data.resize(single_nv12_size);
      right_img_data.resize(single_nv12_size);
      ImgConvertUtils::bgr_mat_to_nv12(left_bgr, left_img_data.data());
      ImgConvertUtils::bgr_mat_to_nv12(right_bgr, right_img_data.data());
    } else if (calib_method_ == "custom") {
      ScopeProcessTime t(this->get_logger(), "stereo rectify");
      // rectify
      cv::Mat left_bgr_rectify, right_bgr_rectify;
      stereo_rectifier_->rectify(left_bgr, right_bgr, left_bgr_rectify, right_bgr_rectify);
      size_t single_nv12_size = left_bgr_rectify.cols * left_bgr_rectify.rows * 3 / 2;
      left_img_data.resize(single_nv12_size);
      right_img_data.resize(single_nv12_size);
      ImgConvertUtils::bgr_mat_to_nv12(left_bgr_rectify, left_img_data.data());
      ImgConvertUtils::bgr_mat_to_nv12(right_bgr_rectify, right_img_data.data());
    }

  } else {
    RCLCPP_ERROR(this->get_logger(), "=> unsupported image encoding: %s", stereo_msg->encoding.c_str());
  }
}

void StereoNetNode::publish_function() {
  while (rclcpp::ok()) {
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
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "=> publish result, stamp: %u.%u, latency: %.2f ms", pub_data->header.stamp.sec,
                           pub_data->header.stamp.nanosec, latency);

      // publish depth image
      {
        ScopeProcessTime t(this->get_logger(), "publish_depth_image");
        publish_depth_image(pub_data);
        publish_depth_camera_info(pub_data);
      }
      // pub rectified images
      {
        ScopeProcessTime t(this->get_logger(), "publish_rectified_image");
        publish_rectified_left_image(pub_data);
        publish_rectified_right_image(pub_data);
      }
      // publish pointcloud2
      {
        ScopeProcessTime t(this->get_logger(), "publish_pointcloud2");
        publish_pointcloud2(pub_data);
      }
      {
        // publish origin images
        ScopeProcessTime t(this->get_logger(), "publish_origin_image");
        publish_origin_left_image(pub_data);
        publish_origin_right_image(pub_data);
      }
      // publish visual image
      {
        ScopeProcessTime t(this->get_logger(), "publish_visual_image");
        publish_visual_image(pub_data);
      }
      {
        // save result
        if (save_result_flag_) save_thread_pool_ptr_->detach_task([this, pub_data]() { save_result(pub_data); });
      }
    }
  }
}

void StereoNetNode::publish_depth_image(const std::shared_ptr<PubData> &pub_data) {
  if (depth_image_pub_->get_subscription_count() == 0) return;

  auto depth_msg = std::make_shared<sensor_msgs::msg::Image>();
  depth_msg->header = pub_data->header;
  depth_msg->header.frame_id = "camera_depth_frame";
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
  depth_camera_info_msg->header.frame_id = "camera_depth_frame";
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

  // Set projection matrix
  depth_camera_info_msg->p[0] = camera_intrinsic_->fx; // fx
  depth_camera_info_msg->p[2] = camera_intrinsic_->cx; // cx
  depth_camera_info_msg->p[5] = camera_intrinsic_->fy; // fy
  depth_camera_info_msg->p[6] = camera_intrinsic_->cy; // cy
  depth_camera_info_msg->p[10] = 1.0;

  depth_camera_info_pub_->publish(*depth_camera_info_msg);
}

void StereoNetNode::publish_rectified_left_image(const std::shared_ptr<PubData> &pub_data) {
  if (rectify_left_image_pub_->get_subscription_count() == 0) return;
  auto left_msg = std::make_shared<sensor_msgs::msg::Image>();
  left_msg->header = pub_data->header;
  left_msg->header.frame_id = "camera_depth_frame";
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
  right_msg->header.frame_id = "camera_right_frame";
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
  if (pointcloud2_pub_->get_subscription_count() == 0 && save_result_flag_ == false) return;
  cv::Mat bgr;
  ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_left_img_data.data(), bgr, pub_data->disp.cols,
                                   pub_data->disp.rows);

  const int step = 2; // downsample
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
      float z = depth_row[u] * 0.001f;
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

  if (save_result_flag_) pub_data->pointcloud = pcl_cloud;

  sensor_msgs::msg::PointCloud2 cloud_msg;
  pcl::toROSMsg(*pcl_cloud, cloud_msg);
  cloud_msg.header = pub_data->header;
  cloud_msg.header.frame_id = "camera_link";
  cloud_msg.is_dense = false;
  cloud_msg.is_bigendian = false;
  pointcloud2_pub_->publish(cloud_msg);
}

void StereoNetNode::publish_origin_left_image(const std::shared_ptr<PubData> &pub_data) {
  if (origin_left_image_pub_->get_subscription_count() == 0 && save_result_flag_ == false) return;
  if (pub_data->origin_stereo_msg->encoding == "nv12") {
    auto left_msg = std::make_shared<sensor_msgs::msg::Image>();
    left_msg->header = pub_data->header;
    left_msg->header.frame_id = "camera_left_frame";
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
    pub_data->origin_left_msg = left_msg;
  } else if (pub_data->origin_stereo_msg->encoding == "rgb8" || pub_data->origin_stereo_msg->encoding == "bgr8") {
    auto left_msg = std::make_shared<sensor_msgs::msg::Image>();
    left_msg->header = pub_data->header;
    left_msg->header.frame_id = "camera_left_frame";
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
      pub_data->origin_left_msg = left_msg;
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
      pub_data->origin_left_msg = left_msg;
    }
  }
}

void StereoNetNode::publish_origin_right_image(const std::shared_ptr<PubData> &pub_data) {
  if (origin_right_image_pub_->get_subscription_count() == 0 && save_result_flag_ == false) return;
  if (pub_data->origin_stereo_msg->encoding == "nv12") {
    auto right_msg = std::make_shared<sensor_msgs::msg::Image>();
    right_msg->header = pub_data->header;
    right_msg->header.frame_id = "camera_right_frame";
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
    pub_data->origin_right_msg = right_msg;
  } else if (pub_data->origin_stereo_msg->encoding == "rgb8" || pub_data->origin_stereo_msg->encoding == "bgr8") {
    auto right_msg = std::make_shared<sensor_msgs::msg::Image>();
    right_msg->header = pub_data->header;
    right_msg->header.frame_id = "camera_right_frame";
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
      pub_data->origin_right_msg = right_msg;
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
      pub_data->origin_right_msg = right_msg;
    }
  }
}

void StereoNetNode::publish_visual_image(const std::shared_ptr<PubData> &pub_data) {
  if (visual_image_pub_->get_subscription_count() == 0) return;
  // ===================================== render visual image ==============================================
  int width = pub_data->disp.cols;
  int height = pub_data->disp.rows;
  cv::Mat left_bgr;
  ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_left_img_data.data(), left_bgr, width, height);

  cv::Mat visual_img;
  double minVal, maxVal;
  cv::minMaxLoc(pub_data->disp, &minVal, &maxVal);
  pub_data->disp.convertTo(visual_img, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
  cv::applyColorMap(visual_img, visual_img, cv::COLORMAP_JET);
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
      depth_text << std::fixed << std::setprecision(2) << depth_value << "m";
      cv::putText(visual_img, depth_text.str(), cv::Point(x + 5, y - 5), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                  CV_RGB(255, 255, 255), 2);
      cv::putText(visual_img, depth_text.str(), cv::Point(x + 5, left_bgr.rows + y - 5), cv::FONT_HERSHEY_SIMPLEX,
                  font_scale, CV_RGB(255, 255, 255), 2);
    }
  }
  // ===================================== render performance metrics =======================================
  if (render_perf_) {
    std::stringstream perf_text;
    perf_text << "FPS: " << pub_data->fps << " Latency: " << pub_data->latency << "ms CPU: " << pub_data->cpu_usage
              << "% BPU: " << pub_data->bpu_usage << "%";
    cv::putText(visual_img, perf_text.str(), cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, font_scale, CV_RGB(0, 0, 255),
                2);
  }

  // ===================================== publish visual image ============================================
  pub_data->visual_img = visual_img;
  // Convert cv::Mat to sensor_msgs::msg::Image
  auto visual_msg = std::make_shared<sensor_msgs::msg::Image>();
  visual_msg->header = pub_data->header;
  visual_msg->header.frame_id = "camera_link";
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
  if (rclcpp::ok() == false) return;
  bool do_save = false;
  int current_count = 0;
  {
    std::lock_guard<std::mutex> lock(save_mutex_);
    if (!save_result_flag_) return;
    if (save_total_ > 0 && save_count_ / save_freq_ >= save_total_) {
      RCLCPP_WARN(this->get_logger(), "\033[31m=> save total %d images, stop saving\033[0m", save_total_);
      save_result_flag_ = false;
      return;
    }
    if (save_count_ % save_freq_ != 0) {
      save_count_++;
      return;
    }
    current_count = save_count_;
    save_count_++;
    do_save = true;
  }
  if (!do_save) return;

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
  }

  std::stringstream ss;
  ss << std::setw(6) << std::setfill('0') << current_count << "_";

  std::string depth_image_path = fs::path(save_dir_) / fs::path(ss.str() + "depth.png");
  std::string disp_image_path = fs::path(save_dir_) / fs::path(ss.str() + "disp.pfm");
  std::string uncert_image_path = fs::path(save_dir_) / fs::path(ss.str() + "uncert.png");
  std::string left_image_path = fs::path(save_dir_) / fs::path(ss.str() + "left.png");
  std::string right_image_path = fs::path(save_dir_) / fs::path(ss.str() + "right.png");
  std::string pointcloud_path = fs::path(save_dir_) / fs::path(ss.str() + "pointcloud.pcd");
  std::string visual_image_path = fs::path(save_dir_) / fs::path(ss.str() + "visual.png");
  std::string origin_left_image_path = fs::path(save_dir_) / fs::path(ss.str() + "origin_L.png");
  std::string origin_right_image_path = fs::path(save_dir_) / fs::path(ss.str() + "origin_R.png");

  cv::Mat left_bgr, right_bgr;
  ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_left_img_data.data(), left_bgr, pub_data->disp.cols,
                                   pub_data->disp.rows);
  ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_right_img_data.data(), right_bgr, pub_data->disp.cols,
                                   pub_data->disp.rows);

  cv::imwrite(depth_image_path, pub_data->depth);
  cv::imwrite(disp_image_path, pub_data->disp);
  if (!pub_data->uncert.empty()) cv::imwrite(uncert_image_path, pub_data->uncert);
  cv::imwrite(left_image_path, left_bgr);
  cv::imwrite(right_image_path, right_bgr);
  if (pub_data->pointcloud && !pub_data->pointcloud->points.empty())
    pcl::io::savePCDFileBinary(pointcloud_path, *(pub_data->pointcloud));
  if (!pub_data->visual_img.empty()) cv::imwrite(visual_image_path, pub_data->visual_img);
  if (pub_data->origin_left_msg && use_local_image_flag_ == false) {
    cv::Mat origin_left;
    if (pub_data->origin_left_msg->encoding == "nv12") {
      ImgConvertUtils::nv12_to_bgr_mat(pub_data->origin_left_msg->data.data(), origin_left,
                                       pub_data->origin_left_msg->width, pub_data->origin_left_msg->height);
    } else if (pub_data->origin_left_msg->encoding == "rgb8") {
      cv::Mat rgb(pub_data->origin_left_msg->height, pub_data->origin_left_msg->width, CV_8UC3,
                  const_cast<uint8_t *>(pub_data->origin_left_msg->data.data()), pub_data->origin_left_msg->step);
      cv::cvtColor(rgb, origin_left, cv::COLOR_RGB2BGR);
    } else if (pub_data->origin_left_msg->encoding == "bgr8") {
      origin_left =
          cv::Mat(pub_data->origin_left_msg->height, pub_data->origin_left_msg->width, CV_8UC3,
                  const_cast<uint8_t *>(pub_data->origin_left_msg->data.data()), pub_data->origin_left_msg->step)
              .clone();
    }
    if (!origin_left.empty()) cv::imwrite(origin_left_image_path, origin_left);
  }
  if (pub_data->origin_right_msg && use_local_image_flag_ == false) {
    cv::Mat origin_right;
    if (pub_data->origin_right_msg->encoding == "nv12") {
      ImgConvertUtils::nv12_to_bgr_mat(pub_data->origin_right_msg->data.data(), origin_right,
                                       pub_data->origin_right_msg->width, pub_data->origin_right_msg->height);
    } else if (pub_data->origin_right_msg->encoding == "rgb8") {
      cv::Mat rgb(pub_data->origin_right_msg->height, pub_data->origin_right_msg->width, CV_8UC3,
                  const_cast<uint8_t *>(pub_data->origin_right_msg->data.data()), pub_data->origin_right_msg->step);
      cv::cvtColor(rgb, origin_right, cv::COLOR_RGB2BGR);
    } else if (pub_data->origin_right_msg->encoding == "bgr8") {
      origin_right =
          cv::Mat(pub_data->origin_right_msg->height, pub_data->origin_right_msg->width, CV_8UC3,
                  const_cast<uint8_t *>(pub_data->origin_right_msg->data.data()), pub_data->origin_right_msg->step)
              .clone();
    }
    if (!origin_right.empty()) cv::imwrite(origin_right_image_path, origin_right);
  }

  RCLCPP_WARN(this->get_logger(), "\033[31m=> save result to %s, save count: %d\033[0m", save_dir_.c_str(),
              current_count);
}

void StereoNetNode::infer_offline() {
  auto img_paths = FileUtils::find_pairs(local_image_dir_);
  RCLCPP_INFO(this->get_logger(), "\033[32m=> found %zu image pairs in %s\033[0m", img_paths.size(),
              local_image_dir_.c_str());

  std::string camera_intrinsic_path = fs::path(local_image_dir_) / fs::path("camera_intrinsic.txt");
  if (fs::exists(camera_intrinsic_path)) {
    bool read_success =
        FileUtils::read_camera_intrinsic(camera_intrinsic_path, camera_intrinsic_->fx, camera_intrinsic_->fy,
                                         camera_intrinsic_->cx, camera_intrinsic_->cy, camera_intrinsic_->baseline);
    if (read_success) {
      RCLCPP_WARN_ONCE(
          this->get_logger(),
          "\033[31m=> read camera intrinsic from %s: fx: %f, fy: %f, cx: %f, cy: %f, baseline(m): %f\033[0m",
          camera_intrinsic_path.c_str(), camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx,
          camera_intrinsic_->cy, camera_intrinsic_->baseline);
    } else {
      RCLCPP_ERROR(this->get_logger(), "\033[31m=> read camera intrinsic from %s failed\033[0m",
                   camera_intrinsic_path.c_str());
      rclcpp::shutdown();
    }
  } else if (camera_intrinsic_->is_valid()) {
    RCLCPP_WARN_ONCE(this->get_logger(),
                     "\033[33m=> use camera intrinsic from parameter: fx: %f, fy: %f, cx: %f, cy: %f, baseline(m): "
                     "%f\033[0m",
                     camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy,
                     camera_intrinsic_->baseline);
  } else {
    RCLCPP_ERROR(this->get_logger(),
                 "\033[31m=> camera intrinsic is not set, please provide camera_intrinsic.txt in %s or set the "
                 "camera parameters [camera_fx, camera_fy, camera_cx, camera_cy, baseline] in the launch file\033[0m",
                 local_image_dir_.c_str());
    rclcpp::shutdown();
  }

  bool intrinsic_updated = false;
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

    if (left_img_bgr.cols != model_input_w || left_img_bgr.rows != model_input_h) {
      if (!intrinsic_updated) {
        camera_intrinsic_->cx = camera_intrinsic_->cx * model_input_w / left_img_bgr.cols;
        camera_intrinsic_->cy = camera_intrinsic_->cy * model_input_h / left_img_bgr.rows;
        camera_intrinsic_->fx = camera_intrinsic_->fx * model_input_w / left_img_bgr.cols;
        camera_intrinsic_->fy = camera_intrinsic_->fy * model_input_h / left_img_bgr.rows;
        RCLCPP_WARN(this->get_logger(),
                    "\033[33m=> update camera intrinsic: fx: %f, fy: %f, cx: %f, cy: %f, baseline(m): %f\033[0m",
                    camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy,
                    camera_intrinsic_->baseline);
        intrinsic_updated = true;
      }

      RCLCPP_WARN(this->get_logger(),
                  "\033[33m=> resize input image to fit model input size, [%d x %d] -> [%d x %d]\033[0m",
                  left_img_bgr.cols, left_img_bgr.rows, model_input_w, model_input_h);
      RCLCPP_WARN(this->get_logger(),
                  "\033[33m=> using camera intrinsic: fx: %f, fy: %f, cx: %f, cy: %f, baseline(m): %f\033[0m",
                  camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy,
                  camera_intrinsic_->baseline);
      cv::resize(left_img_bgr, left_img_bgr, cv::Size(model_input_w, model_input_h));
      cv::resize(right_img_bgr, right_img_bgr, cv::Size(model_input_w, model_input_h));
    }

    cv::Mat combine_img_bgr;
    cv::vconcat(left_img_bgr, right_img_bgr, combine_img_bgr);
    cv::Mat combine_img_nv12;
    ImgConvertUtils::bgr_mat_to_nv12_mat(combine_img_bgr, combine_img_nv12);

    auto stereo_msg = std::make_shared<sensor_msgs::msg::Image>();
    stereo_msg->header.stamp = this->get_clock()->now();
    stereo_msg->header.frame_id = "camera_link";
    stereo_msg->height = combine_img_bgr.rows;
    stereo_msg->width = combine_img_bgr.cols;
    stereo_msg->encoding = "nv12";
    stereo_msg->is_bigendian = false;
    stereo_msg->step = combine_img_bgr.cols; // Y plane step
    size_t size = combine_img_bgr.cols * combine_img_bgr.rows * 3 / 2;
    stereo_msg->data.resize(size);
    std::memcpy(stereo_msg->data.data(), combine_img_nv12.data, size);
    input_image_queue_.enqueue(stereo_msg);
    std::this_thread::sleep_for(std::chrono::milliseconds(image_sleep_));
    cnt++;
  }

  RCLCPP_INFO(this->get_logger(), "\033[32m=> all %d images in %s have been processed\033[0m", cnt,
              local_image_dir_.c_str());
}

} // namespace stereonet

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(stereonet::StereoNetNode)
