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
#include <string>
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
}

void StereoNetNode::set_node_params() {
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

  this->declare_parameter<int>("infer_thread_num", 2);
  infer_thread_num_ = this->get_parameter("infer_thread_num").as_int();

  RCLCPP_WARN_STREAM(this->get_logger(),
                     "=> params:" << std::endl
                                  << "stereonet_model_file_path: " << stereonet_model_file_path_ << std::endl
                                  << "stereo_image_topic: " << stereo_image_topic_ << std::endl
                                  << "camera_info_topic: " << camera_info_topic_ << std::endl
                                  << "depth_image_topic: " << depth_image_topic_ << std::endl
                                  << "depth_camera_info_topic: " << depth_camera_info_topic_ << std::endl
                                  << "rectify_left_image_topic: " << rectify_left_image_topic_ << std::endl
                                  << "rectify_right_image_topic: " << rectify_right_image_topic_ << std::endl
                                  << "pointcloud2_topic: " << pointcloud2_topic_ << std::endl
                                  << "visual_image_topic: " << visual_image_topic_ << std::endl
                                  << "render_perf: " << render_perf_ << std::endl
                                  << "postprocess: " << postprocess_ << std::endl
                                  << "uncertainty_th: " << uncertainty_th_ << std::endl
                                  << "camera_fx: " << camera_intrinsic_->fx << std::endl
                                  << "camera_fy: " << camera_intrinsic_->fy << std::endl
                                  << "camera_cx: " << camera_intrinsic_->cx << std::endl
                                  << "camera_cy: " << camera_intrinsic_->cy << std::endl
                                  << "baseline: " << camera_intrinsic_->baseline << std::endl
                                  << "pointcloud [height min, heght max, depth_max] m: " << pointcloud_height_min_
                                  << "," << pointcloud_height_max_ << "," << pointcloud_depth_max_);
}

void StereoNetNode::set_subscription_publisher() {
  // Set up subscriptions
  stereo_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_image_topic_, 10, std::bind(&StereoNetNode::stereo_image_callback, this, std::placeholders::_1));
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, 10, std::bind(&StereoNetNode::camera_info_callback, this, std::placeholders::_1));

  // Set up publishers
  visual_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(visual_image_topic_, 10);
  depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(depth_image_topic_, 10);
  depth_camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(depth_camera_info_topic_, 10);
  pointcloud2_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(pointcloud2_topic_, 10);
  rectify_left_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(rectify_left_image_topic_, 10);
  rectify_right_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(rectify_right_image_topic_, 10);
}

void StereoNetNode::set_dnn_model() {
  stereonet_process_ = std::make_shared<StereonetProcess>(this->get_logger());
  stereonet_process_->init(stereonet_model_file_path_);
}

void StereoNetNode::set_worker_threads() {
  for (int i = 0; i < infer_thread_num_; ++i) {
    infer_threads_.emplace_back(&StereoNetNode::infer_function, this, i);
  }
  publish_thread_ = std::thread(&StereoNetNode::publish_function, this);
}

void StereoNetNode::stereo_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  auto now = this->get_clock()->now();
  auto latency = (now - msg->header.stamp).seconds() * 1000;
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                       "=> recv stereo image, stamp: %u.%u, latency: %.2f ms, queue size: %zu", msg->header.stamp.sec,
                       msg->header.stamp.nanosec, latency, input_image_queue_.size_approx());

  while (input_image_queue_.size_approx() >= 1) {
    sensor_msgs::msg::Image::SharedPtr drop;
    input_image_queue_.try_dequeue(drop);
  }
  input_image_queue_.enqueue(msg);
}

void StereoNetNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  camera_intrinsic_->fx = msg->p[0];
  camera_intrinsic_->fy = msg->p[5];
  camera_intrinsic_->cx = msg->p[2];
  camera_intrinsic_->cy = msg->p[6];
  camera_intrinsic_->baseline = msg->p[3] / camera_intrinsic_->fx;

  if (camera_intrinsic_->baseline > 1) camera_intrinsic_->baseline *= 0.001f; // convert mm to m

  RCLCPP_WARN_ONCE(this->get_logger(), "\033[31m=> sub rectified fx: %f, fy: %f, cx: %f, cy: %f, base_line: :%f\033[0m",
                   camera_intrinsic_->fx, camera_intrinsic_->fy, camera_intrinsic_->cx, camera_intrinsic_->cy,
                   camera_intrinsic_->baseline);
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

      // ================================== Publish ====================================
      auto pub_data = std::make_shared<PubData>();
      pub_data->timestamp = static_cast<uint64_t>(stereo_msg->header.stamp.sec) * 1'000'000'000 +
                            static_cast<uint64_t>(stereo_msg->header.stamp.nanosec);
      pub_data->header = stereo_msg->header;
      pub_data->disp = disp;
      pub_data->uncert = uncert;

      cv::Mat depth;
      StereonetProcess::disp_to_depth(disp, depth, camera_intrinsic_->fx, camera_intrinsic_->baseline);
      pub_data->depth = depth;

      if (render_perf_) {
        auto now = this->get_clock()->now();
        pub_data->latency = (now - stereo_msg->header.stamp).seconds() * 1000;
        performance_writer::Get()->record_performance(pub_data->latency);
        pub_data->fps = performance_writer::Get()->get_fps();
        pub_data->cpu_usage = performance_writer::Get()->get_cpu_usage();
        pub_data->bpu_usage = performance_writer::Get()->get_bpu_usage();
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

    std::memcpy(left_img_data.data(), stereo_msg->data.data(), single_img_w * single_img_h);
    std::memcpy(left_img_data.data() + single_img_w * single_img_h,
                stereo_msg->data.data() + stereo_msg->width * stereo_msg->height, single_img_w * single_img_h / 2);
    std::memcpy(right_img_data.data(), stereo_msg->data.data() + single_img_w * single_img_h,
                single_img_w * single_img_h);
    std::memcpy(right_img_data.data() + single_img_w * single_img_h,
                stereo_msg->data.data() + stereo_msg->width * stereo_msg->height + single_img_w * single_img_h / 2,
                single_img_w * single_img_h / 2);

  } else if (stereo_msg->encoding == "rgb8" || stereo_msg->encoding == "bgr8") {
    int single_img_w = stereo_msg->width / 2;
    int single_img_h = stereo_msg->height;
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

    cv::Mat left_bgr = stereo_bgr(cv::Rect(0, 0, single_img_w, single_img_h));
    cv::Mat right_bgr = stereo_bgr(cv::Rect(single_img_w, 0, single_img_w, single_img_h));

    left_img_data.resize(single_bgr_size);
    right_img_data.resize(single_bgr_size);
    ImgConvertUtils::bgr_mat_to_nv12(left_bgr, left_img_data.data());
    ImgConvertUtils::bgr_mat_to_nv12(right_bgr, right_img_data.data());

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
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
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
      // publish visual image
      {
        ScopeProcessTime t(this->get_logger(), "publish_visual_image");
        publish_visual_image(pub_data);
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
  left_msg->encoding = "nv12";
  left_msg->is_bigendian = false;
  left_msg->step = width; // Y plane step
  size_t size = width * height * 3 / 2;
  left_msg->data.resize(size);
  std::memcpy(left_msg->data.data(), pub_data->rectify_left_img_data.data(), size);
  rectify_left_image_pub_->publish(*left_msg);
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
  right_msg->encoding = "nv12";
  right_msg->is_bigendian = false;
  right_msg->step = width; // Y plane step
  size_t size = width * height * 3 / 2;
  right_msg->data.resize(size);
  std::memcpy(right_msg->data.data(), pub_data->rectify_right_img_data.data(), size);
  rectify_right_image_pub_->publish(*right_msg);
}

/*
void StereoNetNode::publish_pointcloud2(const std::shared_ptr<PubData> &pub_data) {
  // Convert depth image to point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  cv::Mat bgr;
  ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_left_img_data.data(), bgr, pub_data->disp.cols,
                                   pub_data->disp.rows);
  pcl_cloud->points.reserve(pub_data->depth.rows * pub_data->depth.cols / 4); // reserve for 2x2 downsample
  for (int v = 0; v < pub_data->depth.rows; v += 2) {
    for (int u = 0; u < pub_data->depth.cols; u += 2) {
      float z = pub_data->depth.at<uint16_t>(v, u) * 0.001f; // convert mm to m
      if (z <= 0 || z > pointcloud_depth_max_) continue;
      float x = (u - camera_intrinsic_->cx) * z / camera_intrinsic_->fx;
      float y = (v - camera_intrinsic_->cy) * z / camera_intrinsic_->fy;
      if (y < 0 && abs(y) > pointcloud_height_max_) continue;
      if (y > 0 && y > abs(pointcloud_height_min_)) continue;
      auto r = bgr.at<cv::Vec3b>(v, u)[2];
      auto g = bgr.at<cv::Vec3b>(v, u)[1];
      auto b = bgr.at<cv::Vec3b>(v, u)[0];
      pcl_cloud->points.emplace_back(z, -x, -y, r, g, b);
    }
  }
  pcl_cloud->width = pcl_cloud->points.size();
  pcl_cloud->height = 1;

  sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg(new sensor_msgs::msg::PointCloud2());
  pcl::toROSMsg(*pcl_cloud, *cloud_msg);
  cloud_msg->header = pub_data->header;
  cloud_msg->header.frame_id = "camera_link";
  cloud_msg->is_dense = false;
  cloud_msg->is_bigendian = false;
  pointcloud2_pub_->publish(*cloud_msg);
}
*/

void StereoNetNode::publish_pointcloud2(const std::shared_ptr<PubData> &pub_data) {
  if (pointcloud2_pub_->get_subscription_count() == 0) return;
  cv::Mat bgr;
  ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_left_img_data.data(), bgr, pub_data->disp.cols,
                                   pub_data->disp.rows);

  sensor_msgs::msg::PointCloud2 cloud_msg;
  cloud_msg.header = pub_data->header;
  cloud_msg.header.frame_id = "camera_link";
  cloud_msg.is_dense = false;
  cloud_msg.is_bigendian = false;

  // 定义字段：x, y, z, rgb
  sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
  modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

  // 预估点数（2x2 下采样）
  size_t reserve_size = pub_data->depth.rows * pub_data->depth.cols / 4;
  modifier.resize(reserve_size);

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud_msg, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud_msg, "b");

  int valid_count = 0;
  for (int v = 0; v < pub_data->depth.rows; v += 2) {
    const uint16_t *depth_ptr = pub_data->depth.ptr<uint16_t>(v);
    const cv::Vec3b *color_ptr = bgr.ptr<cv::Vec3b>(v);

    for (int u = 0; u < pub_data->depth.cols; u += 2) {
      float z = depth_ptr[u] * 0.001f; // mm → m
      if (z <= 0 || z > pointcloud_depth_max_) continue;
      float x = (u - camera_intrinsic_->cx) * z / camera_intrinsic_->fx;
      float y = (v - camera_intrinsic_->cy) * z / camera_intrinsic_->fy;
      if (y < 0 && std::abs(y) > pointcloud_height_max_) continue;
      if (y > 0 && y > std::abs(pointcloud_height_min_)) continue;
      *iter_x = z;
      *iter_y = -x;
      *iter_z = -y;
      *iter_r = color_ptr[u][2];
      *iter_g = color_ptr[u][1];
      *iter_b = color_ptr[u][0];
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_r;
      ++iter_g;
      ++iter_b;
      valid_count++;
    }
  }

  // resize to actual valid points
  cloud_msg.width = valid_count;
  cloud_msg.height = 1;
  cloud_msg.row_step = cloud_msg.point_step * valid_count;
  cloud_msg.data.resize(valid_count * cloud_msg.point_step);

  pointcloud2_pub_->publish(cloud_msg);
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

} // namespace stereonet

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(stereonet::StereoNetNode)
