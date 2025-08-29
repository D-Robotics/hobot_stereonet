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

#include <filesystem>
#include <sstream>
#include <mutex>
#include <omp.h>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/common/transforms.h"
#include "pcl_conversions/pcl_conversions.h"
#include "blockingconcurrentqueue.h"
#include "BS_thread_pool.hpp"
#include "img_convert_utils.h"
#include "stereonet_process.h"
#include "order_blockqueue.hpp"
#include "performance_record.h"

namespace fs = std::filesystem;
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
    uint64_t timestamp;
    std_msgs::msg::Header header;
    cv::Mat disp;
    cv::Mat depth;
    cv::Mat uncert;
    std::vector<uint8_t> rectify_left_img_data;  // nv12
    std::vector<uint8_t> rectify_right_img_data; // nv12
    int fps, latency;
    int cpu_usage, bpu_usage;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud = nullptr;
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
   * @param model_input_w model input width
   * @param model_input_h model input height
   * @param left_img_data Output shared pointer to the left image data
   * @param right_img_data Output shared pointer to the right image data
   */
  void preprocess(const sensor_msgs::msg::Image::SharedPtr &stereo_msg, const int &model_input_w,
                  const int &model_input_h, std::vector<uint8_t> &left_img_data, std::vector<uint8_t> &right_img_data);

  /**
   * @brief Publish function to publish the processed disparity maps
   * This function runs in a separate thread and continuously publishes disparity maps.
   */
  void publish_function();

  /**
   * @brief Publish the depth image based on the disparity map
   * @param pub_data The processed data containing the disparity map and metadata
   */
  void publish_depth_image(const std::shared_ptr<PubData> &pub_data);
  /**
   * @brief Publish the depth camera info based on the original camera info
   */
  void publish_depth_camera_info(const std::shared_ptr<PubData> &pub_data);
  /**
   * @brief Publish the rectified left image
   */
  void publish_rectified_left_image(const std::shared_ptr<PubData> &pub_data);

  /**
   * @brief Publish the rectified right image
   */
  void publish_rectified_right_image(const std::shared_ptr<PubData> &pub_data);

  /**
   * @brief Publish the point cloud based on the disparity map
   * @param pub_data The processed data containing the disparity map and metadata
   */
  void publish_pointcloud2(const std::shared_ptr<PubData> &pub_data);

  /**
   * @brief Publish the visual image based on the disparity map
   * @param pub_data The processed data containing the disparity map and metadata
   */
  void publish_visual_image(const std::shared_ptr<PubData> &pub_data);

  /**
   * @brief Publish static TF for the stereo camera setup
   */
  void publish_static_tf();

  /**
   * @
   */
  void save_result(const std::shared_ptr<PubData> &pub_data);

  // ============================================ member variables ============================================
  // sub
  std::string stereo_image_topic_ = "/image_combine_raw";
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr stereo_image_sub_ = nullptr;
  std::string camera_info_topic_ = "/image_right_raw/camera_info";
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_ = nullptr;

  // pub
  std::string visual_image_topic_ = "~/stereonet_visual";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visual_image_pub_ = nullptr;
  bool render_perf_ = true;
  std::string depth_image_topic_ = "~/stereonet_depth";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_ = nullptr;
  std::string depth_camera_info_topic_ = "~/stereonet_depth/camera_info";
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_camera_info_pub_ = nullptr;
  std::string pointcloud2_topic_ = "~/stereonet_pointcloud2";
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud2_pub_ = nullptr;
  std::string rectify_left_image_topic_ = "~/rectify_left_image";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rectify_left_image_pub_ = nullptr;
  std::string rectify_right_image_topic_ = "~/rectify_right_image";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rectify_right_image_pub_ = nullptr;
  bool publish_rectify_bgr_ = false;

  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_ = nullptr;

  // model params
  std::shared_ptr<StereonetProcess> stereonet_process_ = nullptr;
  std::string stereonet_model_file_path_ = "";
  std::string postprocess_ = "convex_upsampling";
  double uncertainty_th_ = 0.0;
  std::shared_ptr<CameraIntrinsic> camera_intrinsic_ = nullptr;

  double pointcloud_height_min_ = -5.0;
  double pointcloud_height_max_ = 5.0;
  double pointcloud_depth_max_ = 5.0;

  // save params
  bool save_result_flag_ = false;
  std::string save_dir_ = "./stereonet_result";
  int save_freq_ = 1;
  int save_total_ = -1;
  int save_count_ = 0;
  std::mutex save_mutex_;

  // thread
  moodycamel::BlockingConcurrentQueue<sensor_msgs::msg::Image::SharedPtr> input_image_queue_;
  std::vector<std::thread> infer_threads_;
  int infer_thread_num_ = 2;
  order_blockqueue<std::shared_ptr<PubData>> pub_data_queue_;
  std::thread publish_thread_;
  uint64_t last_frame_timestamp_ = 0;
  BS::thread_pool<> save_thread_pool_;
};
} // namespace stereonet
#endif // HOBOT_STEREONET_INCLUDE_STEREONET_COMPONENT_H_
