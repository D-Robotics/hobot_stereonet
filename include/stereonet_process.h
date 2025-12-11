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

#ifndef HOBOT_STEREONET_INCLUDE_STEREONET_PROCESS_H_
#define HOBOT_STEREONET_INCLUDE_STEREONET_PROCESS_H_

#include <atomic>
#include <deque>
#include <string>
#include <opencv2/opencv.hpp>
#include "log_macros.h"
#include "camera_intrinsic.h"
#include "Eigen/Dense"
#include "magic_enum/magic_enum.hpp"
#include "dnn_platform.h"
#include "timer_utils.h"
#if HOBOT_HAS_RCLCPP
#include "order_blockqueue.hpp"
#include "BS_thread_pool.hpp"
#include "pub_data.h"
#endif

namespace stereonet {
// =================================================================================================================================
#define HB_CHECK_SUCCESS(logger, ret_code, errmsg)                                                                     \
  do {                                                                                                                 \
    /*value can be call of function*/                                                                                  \
    if (ret_code != 0) {                                                                                               \
      LOG_ERROR(logger, "=> [BPU ERROR]: " << errmsg << ", error code: " << ret_code);                                 \
    }                                                                                                                  \
  } while (0);

// =================================================================================================================================

/**
 * @brief StereonetProcess class for StereoNet model inference
 * This class is used to initialize and manage the StereoNet model inference process.
 */
class StereonetProcess {
public:
  StereonetProcess() = default;
  explicit StereonetProcess(const rclcpp::Logger &logger);
  ~StereonetProcess();

  /**
   * @brief Initialize the StereoNet model
   * @param model_path Path to the StereoNet model file
   * @param max_memory_count Maximum number of memory buffers to allocate
   * @return 0 on success, -1 on failure
   */
  int init(const std::string &model_path, const int &max_memory_count = 5);

  /**
   * @brief Perform forward inference using the StereoNet model asynchronously
   * @param left_img_data Pointer to the left image data in NV12 format
   * @param right_img_data Pointer to the right image data in NV12 format
   * @param idle_tensor_id Output tensor id
   * @return 0 on success, -1 on failure
   */
  int forward(uint8_t *left_img_data, uint8_t *right_img_data, int &idle_tensor_id);

  /**
   * @brief Perform forward inference using the StereoNet model
   * @param left_img_data Pointer to the left image data in NV12 format
   * @param right_img_data Pointer to the right image data in NV12 format
   * @param uncertainty_th Uncertainty threshold for postprocessing
   * @param postprocess Postprocessing method to apply (e.g., "convex_upsampling")
   * @param disp Output disparity map
   * @param uncert Output uncertainty map
   * @return 0 on success, -1 on failure
   */
  int forward_sync(std::vector<uint8_t> &left_img_data, std::vector<uint8_t> &right_img_data,
                   const double &uncertainty_th, cv::Mat &disp, cv::Mat &uncert);

#if HOBOT_HAS_RCLCPP
  /**
   * @brief Perform forward inference using the StereoNet model asynchronously
   * @param left_img_data Pointer to the left image data in NV12 format
   * @param right_img_data Pointer to the right image data in NV12 format
   * @param uncertainty_th Uncertainty threshold for postprocessing
   * @param camera_intrinsic Pointer to the camera intrinsic parameters
   * @param stereo_msg Pointer to the stereo image message
   * @param pub_data_queue Output queue for publishing processed data
   * @return 0 on success, -1 on failure
   */
  int forward_async(std::vector<uint8_t> &left_img_data, std::vector<uint8_t> &right_img_data,
                    const double &uncertainty_th, std::shared_ptr<CameraIntrinsic> camera_intrinsic,
                    const sensor_msgs::msg::Image::SharedPtr &stereo_msg,
                    order_blockqueue<std::shared_ptr<PubData>> &pub_data_queue);
#endif

  /**
   * @brief Postprocess the output tensors using convex upsampling
   * @param idle_tensor_id Output tensor id
   * @param uncertainty_th Uncertainty threshold for postprocessing
   * @param disp Output disparity map
   * @param uncert Output uncertainty map
   */
  int postprocess(const int idle_tensor_id, const double &uncertainty_th, cv::Mat &disp, cv::Mat &uncert,
                  const std::string &post_version = "auto");

  /**
   * @brief Postprocess and output disparity map, uncertainty map and depth map
   * @param idle_tensor_id Output tensor id
   * @param uncertainty_th Uncertainty threshold for postprocessing
   * @param disp Output disparity map
   * @param uncert Output uncertainty map
   * @param fx Focal length in x direction
   * @param baseline Baseline distance between the two cameras
   * @param depth Output depth map
   * @return 0 on success, -1 on failure
   */
  int postprocess_out_disp_depth(const int idle_tensor_id, const double &uncertainty_th, float *disp, float *uncert,
                                 const double fx, const double baseline, uint16_t *depth);

  /**
   * @brief Get the input size required by the model
   * @param w Width of the input image
   * @param h Height of the input image
   */
  void get_model_input_size(int &w, int &h) const;

  /**
   * @brief Convert disparity map to depth map
   * @param disp Input disparity map
   * @param depth Output depth map
   * @param fx Focal length in x direction
   * @param baseline Baseline distance between the two cameras
   */
  static void disp_to_depth(const cv::Mat &disp, cv::Mat &depth, const double fx, const double baseline);

private:
  // ===================================== member functions =======================================
  /**
   * @brief prepare input tensor for model inference
   * @param input_tensors vector to hold the prepared input tensors
   * @return 0 on success, -1 on failure
   */
  int prepare_input_tensor(std::vector<hbDNNTensor> &input_tensors);

  /**
   * @brief prepare output tensor for model inference
   * @param output_tensors vector to hold the prepared output tensors
   * @return 0 on success, -1 on failure
   */
  int prepare_output_tensor(std::vector<hbDNNTensor> &output_tensors);

  /**
   * @brief Get an idle tensor index for processing
   * @return Index of an idle tensor, or -1 if none are available
   */
  int get_idle_tensor();

  /**
   * @brief Set a tensor as idle after processing
   * @param tensor_id Index of the tensor to set as idle
   * @return 0 on success, -1 on failure
   */
  int set_tensor_idle(int tensor_id);

  /**
   * @brief Fill image data into the input tensor
   * @param input_tensors Vector of input tensors to fill
   * @param left_img_data Pointer to the left image data
   * @param right_img_data Pointer to the right image data
   * @return 0 on success, -1 on failure
   */
  int fill_img_to_input_tensor(std::vector<hbDNNTensor> &input_tensors, uint8_t *left_img_data,
                               uint8_t *right_img_data);

  /**
   * @brief Postprocess the output tensors using convex upsampling
   * @param tensors Vector of output tensors from the model
   * @param out_mat Output matrix to hold the processed result
   * @return 0 on success, -1 on failure
   */
  int postprocess_convex_upsampling(const std::vector<hbDNNTensor> &tensors, cv::Mat &out_mat);

  /**
   * @brief Postprocess the output tensors using convex upsampling with interpolation
   * @param tensors Vector of output tensors from the model
   * @param out_mat Output matrix to hold the processed result
   * @return 0 on success, -1 on failure
   */
  int postprocess_convex_upsampling_with_interp(const std::vector<hbDNNTensor> &tensors, cv::Mat &out_mat);

  // ===================================== member variables =======================================
  rclcpp::Logger logger_;
  std::string model_path_;
  hbPackedDNNHandle_t packed_dnn_handle_;
  const char **model_name_list_;
  int model_count_ = 0;
  hbDNNHandle_t dnn_handle_;
  int input_count_ = 0;
  int output_count_ = 0;

  int32_t input_tensor_type_;

  int max_memory_count_ = 5;
  std::deque<std::atomic_bool> idle_tensor_;
  std::vector<std::vector<hbDNNTensor>> batch_output_tensors_;
  std::vector<std::vector<hbDNNTensor>> batch_input_tensors_;

  int model_input_w_;
  int model_input_h_;
  // int model_output_w_;
  // int model_output_h_;

  std::string postprocess_;

  int max_disp_ = 192;
  float uncertainty_th_ = 0.10;

#if HOBOT_HAS_RCLCPP
  std::unique_ptr<BS::thread_pool<>> postprocess_thread_pool_ptr_ = nullptr;
#endif
};
} // namespace stereonet

#endif // HOBOT_STEREONET_INCLUDE_STEREONET_PROCESS_H_
