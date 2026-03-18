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
#include "file_utils.h"
#include "stereonet_process.h"
#include "order_blockqueue.hpp"
#include "performance_record.h"
#include "speckle_filter.h"
#include "stereo_rectify.h"
#include "pcl_filter.h"
#include "camera_intrinsic.h"
#include "pub_data.h"
#include "epipolar_align.h"
#include "feature_epipolar_align.h"

namespace fs = std::filesystem;

using RoiVec = std::vector<uint16_t>;

struct PreProcessData {
  PreProcessData(sensor_msgs::msg::Image::SharedPtr stereo_msg, std::vector<uint8_t> rectify_left_img_data,
                 std::vector<uint8_t> rectify_right_img_data)
      : stereo_msg(stereo_msg), rectify_left_img_data(rectify_left_img_data),
        rectify_right_img_data(rectify_right_img_data) {
  }
  sensor_msgs::msg::Image::SharedPtr stereo_msg;
  std::vector<uint8_t> rectify_left_img_data;
  std::vector<uint8_t> rectify_right_img_data;
};

namespace stereonet {

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
   * @brief Callback function for left camera info subscription
   * @param msg The received left camera info message
   */
  void left_camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

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
   * @brief Publish the depth camera info
   */
  void publish_depth_camera_info(const std::shared_ptr<PubData> &pub_data);
  /**
   * @brief Publish the rectify left camera info
   */
  void publish_rectify_left_camera_info(const std::shared_ptr<PubData> &pub_data);
  /**
   * @brief Publish the rectify right camera info
   */
  void publish_rectify_right_camera_info(const std::shared_ptr<PubData> &pub_data);
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
   * @brief Publish the epipolar aligned image
   * @param pub_data The processed data containing the disparity map and metadata
   */
  void publish_epipolar_image(const std::shared_ptr<PubData> &pub_data);

  /**
   * @brief Publish the feature epipolar aligned image
   * @param pub_data The processed data containing the disparity map and metadata
   */
  void publish_feature_epipolar_image(const std::shared_ptr<PubData> &pub_data);

  /**
   * @brief Publish the original left and right images
   * @param pub_data The processed data containing the disparity map and metadata
   */
  void publish_origin_left_image(const std::shared_ptr<PubData> &pub_data);

  /**
   * @brief Publish the original left and right images
   * @param pub_data The processed data containing the disparity map and metadata
   */
  void publish_origin_right_image(const std::shared_ptr<PubData> &pub_data);

  /**
   * @brief Publish static TF for the stereo camera setup
   */
  void publish_static_tf();

  /**
   * @brief Inference by local image
   */
  void infer_offline();

  /**
   * @brief save result to local dir
   * @param pub_data The processed data containing the disparity map and metadata
   */
  void save_result(const std::shared_ptr<PubData> &pub_data);

  /**
   * @brief save result to local dir once
   * @param pub_data The processed data containing the disparity map and metadata
   */
  void save_result_once(const std::shared_ptr<PubData> &pub_data);

  // ============================================ member variables ============================================
  // sub
  std::string stereo_image_topic_ = "/image_combine_raw";
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr stereo_image_sub_ = nullptr;
  std::string camera_info_topic_ = "/image_combine_raw/right/camera_info";
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_ = nullptr;
  std::string left_camera_info_topic_ = "/image_combine_raw/left/camera_info";
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_camera_info_sub_ = nullptr;
  sensor_msgs::msg::CameraInfo::SharedPtr origin_camera_info_ = nullptr;
  sensor_msgs::msg::CameraInfo::SharedPtr origin_left_camera_info_ = nullptr;

  // pub
  std::string visual_image_topic_ = "~/stereonet_visual";
  bool publish_visual_enabled_ = true;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr visual_image_pub_ = nullptr;
  std::string depth_image_topic_ = "~/stereonet_depth";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_ = nullptr;
  std::string depth_camera_info_topic_ = "~/stereonet_depth/camera_info";
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_camera_info_pub_ = nullptr;
  std::string rectify_left_camera_info_topic_ = "~/rectify_left_image/camera_info";
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rectify_left_camera_info_pub_ = nullptr;
  std::string rectify_right_camera_info_topic_ = "~/rectify_right_image/camera_info";
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rectify_right_camera_info_pub_ = nullptr;
  std::string pointcloud2_topic_ = "~/stereonet_pointcloud2";
  bool publish_pcd_enabled_ = true;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud2_pub_ = nullptr;
  std::string rectify_left_image_topic_ = "~/rectify_left_image";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rectify_left_image_pub_ = nullptr;
  std::string rectify_right_image_topic_ = "~/rectify_right_image";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rectify_right_image_pub_ = nullptr;
  bool publish_rectify_bgr_ = false;
  std::string origin_left_image_topic_ = "~/origin_left_image";
  std::string origin_right_image_topic_ = "~/origin_right_image";
  bool publish_origin_enable_ = true;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr origin_left_image_pub_ = nullptr;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr origin_right_image_pub_ = nullptr;
  std::string stereonet_frame_id_ = "camera_link";
  std::string stereonet_frame_id_right_ = "camera_link_right";

  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_ = nullptr;

  // model params
  std::shared_ptr<StereonetProcess> stereonet_process_ = nullptr;
  std::string stereonet_model_file_path_ = "";
  double uncertainty_th_ = 0.0;

  int pointcloud_downsample_step_ = 2;
  double pointcloud_height_min_ = -5.0;
  double pointcloud_height_max_ = 5.0;
  double pointcloud_depth_max_ = 5.0;

  // postprocess params
  bool speckle_filter_enable_ = false;
  int max_speckle_size_ = 100;
  float max_disp_diff_ = 1.0f;
  bool left_img_mask_enable_ = false;

  bool pcl_filter_enable_ = false;
  // float voxel_leaf_size_ = 0.05f; // m
  // int mean_k_ = 10;
  // double std_thresh_ = 1.0;
  float grid_size_ = 0.1f; // m
  int grid_min_point_count_ = 5;

  // offline infer
  bool use_local_image_flag_ = false;
  std::string local_image_dir_ = "./offline_image";
  rclcpp::TimerBase::SharedPtr infer_offline_timer_ = nullptr;
  int image_sleep_ = 0; // ms

  // save params
  bool save_result_flag_ = false;
  std::string save_dir_ = "./stereonet_result";
  int save_freq_ = 1;
  int save_total_ = -1;
  bool save_stereo_flag_ = true;
  bool save_origin_flag_ = false;
  bool save_disp_flag_ = true;
  bool save_uncert_flag_ = false;
  bool save_depth_flag_ = true;
  bool save_visual_flag_ = true;
  bool save_pcd_flag_ = false;
  bool save_result_once_ = false;
  bool do_save_result_once_ = false;

  // calib params
  std::shared_ptr<CameraIntrinsic> orignal_camera_intrinsic_ = nullptr;
  std::shared_ptr<CameraIntrinsic> camera_intrinsic_ = nullptr;
  std::string calib_method_ = "none"; // none, custom
  std::string stereo_calib_file_path_ = "";
  std::atomic<bool> camera_info_updated_{false};
  std::atomic<bool> calc_fov_flag_{false};
  std::shared_ptr<StereoRectify> stereo_rectifier_ = nullptr;

  // render
  std::string render_type_ = "distance";
  bool render_perf_ = true;
  int depth_decimal_num_ = 2;
  int render_max_disp_ = 80;
  double render_z_near_ = -1.0;
  double render_z_range_ = 3.0;

  // measure mode
  bool measure_mode_ = false;
  int roi_size_ = 10;
  std::deque<RoiVec> roi_buffer;
  double gt_depth_ = 0.0;

  // epipolar mode
  bool epipolar_mode_ = false;
  std::string epipolar_img_ = "rect";
  int chessboard_per_rows_ = 20;
  int chessboard_per_cols_ = 11;
  double chessboard_square_size_ = 0.06;
  bool feature_epipolar_mode_ = false;

  // thread
  moodycamel::BlockingConcurrentQueue<sensor_msgs::msg::Image::SharedPtr> input_image_queue_;
  std::vector<std::thread> infer_threads_;
  int infer_thread_num_ = 2;
  order_blockqueue<std::shared_ptr<PubData>> pub_data_queue_;
  std::thread publish_thread_;
  uint64_t last_frame_timestamp_ = 0;
  std::unique_ptr<BS::thread_pool<>> save_thread_pool_ptr_ = nullptr;
  int save_thread_num_ = 4;
  int max_save_task_ = 50;
  moodycamel::BlockingConcurrentQueue<std::shared_ptr<PreProcessData>> pre_process_queue_;

  std::string post_version_ = "auto";

  // param callback
  OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};
} // namespace stereonet
#endif // HOBOT_STEREONET_INCLUDE_STEREONET_COMPONENT_H_
