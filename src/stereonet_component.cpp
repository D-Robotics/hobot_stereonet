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
  set_worker_threads();
}

StereoNetNode::~StereoNetNode() {
  RCLCPP_INFO(this->get_logger(), "Node is being destroyed!");
  for (auto &t : infer_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void StereoNetNode::set_node_params() {
  this->declare_parameter<std::string>("stereo_image_topic", "/image_combine_raw");
  stereo_image_topic_ = this->get_parameter("stereo_image_topic").as_string();

  this->declare_parameter<std::string>("stereonet_model_file_path", "");
  stereonet_model_file_path_ = this->get_parameter("stereonet_model_file_path").as_string();

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

  RCLCPP_INFO_STREAM(this->get_logger(), "=> params" << std::endl
                                                     << "stereo_image_topic: " << stereo_image_topic_ << std::endl
                                                     << "stereonet_model_file_path: " << stereonet_model_file_path_
                                                     << std::endl
                                                     << "uncertainty_th: " << uncertainty_th_ << std::endl
                                                     << "camera_fx: " << camera_intrinsic_->fx << std::endl
                                                     << "camera_fy: " << camera_intrinsic_->fy << std::endl
                                                     << "camera_cx: " << camera_intrinsic_->cx << std::endl
                                                     << "camera_cy: " << camera_intrinsic_->cy << std::endl
                                                     << "baseline: " << camera_intrinsic_->baseline << std::endl);
}

void StereoNetNode::set_subscription_publisher() {
  // Set up subscriptions
  stereo_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_image_topic_, 10, std::bind(&StereoNetNode::stereo_image_callback, this, std::placeholders::_1));
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, 10, std::bind(&StereoNetNode::camera_info_callback, this, std::placeholders::_1));

  // Set up publishers
  visual_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(visual_topic_, 10);
}

void StereoNetNode::set_dnn_model() {
  stereonet_process_ = std::make_shared<StereonetProcess>(this->get_logger());
  stereonet_process_->init(stereonet_model_file_path_);
}

void StereoNetNode::set_worker_threads() {
  for (int i = 0; i < 2; ++i) {
    infer_threads_.emplace_back(&StereoNetNode::infer_function, this, i);
  }
  publish_thread_ = std::thread(&StereoNetNode::publish_function, this);
}

void StereoNetNode::stereo_image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
  while (input_image_queue_.size_approx() >= 2) {
    sensor_msgs::msg::Image::SharedPtr drop;
    input_image_queue_.try_dequeue(drop);
    RCLCPP_WARN(this->get_logger(), "=> drop one message to avoid queue too long");
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

      std::vector<uint8_t> left_img_data, right_img_data;
      int single_img_w = 0, single_img_h = 0;
      {
        ScopeProcessTime t(this->get_logger(), "preprocess");
        preprocess(stereo_msg, left_img_data, right_img_data, single_img_w, single_img_h);
      }

      cv::Mat disp, uncert;
      stereonet_process_->forward(left_img_data, right_img_data, single_img_w, single_img_h, uncertainty_th_, disp,
                                  uncert);

      auto pub_data = std::make_shared<PubData>();
      pub_data->header = stereo_msg->header;
      pub_data->disp = disp;
      pub_data->fps = 0;
      pub_data->latency = 0;
      pub_data->cpu_usage = 0;
      pub_data->bpu_usage = 0;
      uint64_t timestamp = static_cast<uint64_t>(stereo_msg->header.stamp.sec) * 1'000'000'000 +
                           static_cast<uint64_t>(stereo_msg->header.stamp.nanosec);
      pub_data_queue_.put(timestamp, pub_data);
    }
  }
}

void StereoNetNode::preprocess(const sensor_msgs::msg::Image::SharedPtr &stereo_msg,
                               std::vector<uint8_t> &left_img_data, std::vector<uint8_t> &right_img_data,
                               int &single_img_w, int &single_img_h) {
  if (stereo_msg->encoding == "nv12") {
    single_img_w = stereo_msg->width;
    single_img_h = stereo_msg->height / 2;
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
    single_img_w = stereo_msg->width / 2;
    single_img_h = stereo_msg->height;
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
      // Publish visual image
      {
        ScopeProcessTime t(this->get_logger(), "publish_visual_image");
        publish_visual_image(pub_data);
      }
      // Publish point cloud
    }
  }
}

void StereoNetNode::publish_visual_image(const std::shared_ptr<PubData> &pub_data) {
  cv::Mat disp_vis;
  double minVal, maxVal;
  cv::minMaxLoc(pub_data->disp, &minVal, &maxVal);
  pub_data->disp.convertTo(disp_vis, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
  cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);

  // Convert cv::Mat to sensor_msgs::msg::Image
  auto visual_msg = std::make_shared<sensor_msgs::msg::Image>();
  visual_msg->header = pub_data->header;
  visual_msg->height = disp_vis.rows;
  visual_msg->width = disp_vis.cols;
  visual_msg->encoding = "bgr8";
  visual_msg->is_bigendian = false;
  visual_msg->step = disp_vis.cols * disp_vis.elemSize();
  size_t size = visual_msg->step * visual_msg->height;
  visual_msg->data.resize(size);
  std::memcpy(visual_msg->data.data(), disp_vis.data, size);

  visual_image_pub_->publish(*visual_msg);
}

} // namespace stereonet

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(stereonet::StereoNetNode)
