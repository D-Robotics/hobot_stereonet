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

#ifndef HOBOT_STEREONET_INCLUDE_STEREONET_COMPONENT_H_
#define HOBOT_STEREONET_INCLUDE_STEREONET_COMPONENT_H_

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "blockingconcurrentqueue.h"
#include "img_convert_utils.h"
#include "stereonet_process.h"
#include "order_blockqueue.hpp"

namespace stereonet {
/**
 * @struct CameraIntrinsic
 * @brief Structure to hold camera intrinsic parameters.
 */
struct CameraIntrinsic {
  double cx = 0.0;
  double cy = 0.0;
  double fx = 0.0;
  double fy = 0.0;
  double baseline = 0.0; // in meters
};

/**
 * @class StereoNetNode
 * @brief A ROS2 node that performs stereo depth estimation using the StereoNet model.
 */
class StereoNetNode : public rclcpp::Node {
public:
  explicit StereoNetNode(const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions(),
                         const std::string &node_name = "StereoNetNode");
  ~StereoNetNode();

private:
  struct PubData {
    std_msgs::msg::Header header;
    cv::Mat disp;
    int fps, latency;
    int cpu_usage, bpu_usage;
  };

  // ============================================ member functions ============================================
  /**
   * @brief Set parameters for the node
   */
  void set_node_params();

  /**
   * @brief Set up subscriptions and publishers for the node
   */
  void set_subscription_publisher();

  /**
   * @brief Load and set up the DNN model for inference
   */
  void set_dnn_model();

  /**
   * @brief Set up worker threads for processing
   */
  void set_worker_threads();

  /**
   * @brief Callback function for stereo image subscription
   * @param msg The received stereo image message
   */
  void stereo_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

  /**
   * @brief Callback function for camera info subscription
   * @param msg The received camera info message
   */
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  /**
   * @brief Inference function to process stereo images and generate disparity maps
   * This function runs in a separate thread and continuously processes images from the input queue.
   * @param thread_id The ID of the thread for logging purposes
   */
  void infer_function(const int &thread_id);

  /**
   * @brief Preprocess function to convert stereo image message to left and right image data
   * @param stereo_msg The received stereo image message
   * @param left_img_data Output shared pointer to the left image data
   * @param right_img_data Output shared pointer to the right image data
   * @param single_img_w Output width of a single image
   * @param single_img_h Output height of a single image
   */
  void preprocess(const sensor_msgs::msg::Image::SharedPtr &stereo_msg, std::vector<uint8_t> &left_img_data,
                  std::vector<uint8_t> &right_img_data, int &single_img_w, int &single_img_h);

  /**
   * @brief Publish function to publish the processed disparity maps
   * This function runs in a separate thread and continuously publishes disparity maps.
   */
  void publish_function();

  /**
   * @brief Publish the visual image based on the disparity map
   * @param pub_data The processed data containing the disparity map and metadata
   * @return int Status code (0 for success, non-zero for failure)
   */
  void publish_visual_image(const std::shared_ptr<PubData> &pub_data);

  // ============================================ member variables ============================================
  std::string stereo_image_topic_ = "/image_combine_raw";
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr stereo_image_sub_ = nullptr;
  std::string camera_info_topic_ = "/image_right_raw/camera_info";
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_ = nullptr;
  std::string visual_topic_ = "~/stereonet_visual";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visual_image_pub_ = nullptr;

  std::string stereonet_model_file_path_ = "";

  double uncertainty_th_ = 0.0;
  std::shared_ptr<CameraIntrinsic> camera_intrinsic_ = nullptr;

  // DNN model processing class
  std::shared_ptr<StereonetProcess> stereonet_process_ = nullptr;

  moodycamel::BlockingConcurrentQueue<sensor_msgs::msg::Image::SharedPtr> input_image_queue_;
  std::vector<std::thread> infer_threads_;
  order_blockqueue<std::shared_ptr<PubData>> pub_data_queue_;
  std::thread publish_thread_;
};
} // namespace stereonet
#endif // HOBOT_STEREONET_INCLUDE_STEREONET_COMPONENT_H_
