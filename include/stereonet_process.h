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

#include <atomic>
#include <deque>
#include <string>
#include <opencv2/opencv.hpp>

#include "dnn_platform/dnn_platform.h"

#include "image_conversion.h"

#include "Eigen/Dense"

#ifndef STEREO_INCLUDE_STEREONET_PROCESS_H_
#define STEREO_INCLUDE_STEREONET_PROCESS_H_

#define ALIGN_16(v) ((v + 15) & ~15)
#define ALIGN(value, alignment) (((value) + ((alignment)-1)) & ~((alignment)-1))
#define ALIGN_32(value) ALIGN(value, 32)

#define HB_CHECK_SUCCESS(value, errmsg)                          \
do {                                                             \
  /*value can be call of function*/                              \
  int32_t ret_code = value;                                      \
  if (ret_code != 0) {                                           \
      std::cout << "[BPU ERROR]" << errmsg << "error code: " << std::endl; \
  }                                                              \
} while (0);

#if __has_include(<rclcpp/rclcpp.hpp>)
#include <rclcpp/rclcpp.hpp>
#endif

struct ScopeProcessTime {
 public:
  ScopeProcessTime(const std::string &name) : name_(name) {
    //std::cout << name << " START" << std::endl;
    begin_ = std::chrono::system_clock::now();
  }
  ~ScopeProcessTime() {
    auto end = std::chrono::system_clock::now();
    const std::chrono::duration<float, std::milli> d = end - begin_;
#if __has_include(<rclcpp/rclcpp.hpp>)
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger(""), name_ << ", consume: "
                                                      << std::fixed << std::setprecision(3)
                                                      << d.count() << "ms");
#endif
  }

 private:
  std::string name_;
  std::chrono::system_clock::time_point begin_;
};

struct StereonetProcess {

  enum StereonetErrorCode {
    OK = 0,
    TENSOR_BUSY = -1,
    DNN_ERROR = -2,
    INPUT_ERROR = -3
  };

  const int MAX_PROCESS_COUNT = 5;
  StereonetProcess();

  int stereonet_init(const std::string &model_file_name,
                     int max_disp, const std::string &postprocess, float uncertainty_th);
  int stereonet_deinit();

  int stereonet_inference(const cv::Mat &left_img,
                          const cv::Mat &right_img,
                          bool is_nv12,
                          std::vector<float> &points);

  void get_depth_width_height(int &width, int &height) const {
    width = model_output_w_;
    height = model_output_h_;
  }

  void get_input_width_height(int &width, int &height) const {
    width = model_input_w_;
    height = model_input_h_;
  }

  int set_uncertainty_th(float uncertainty_th) {
    if (uncertainty_th > 1) return StereonetErrorCode::INPUT_ERROR;
    uncertainty_th_ = uncertainty_th;
    return StereonetErrorCode::OK;
  }

  void get_blind_area(float fx, float base_line, float& blind_area) {
    blind_area = base_line * fx / max_disp_;
  }

 private:
  int get_idle_tensor();
  int set_tensor_idle(int tensor_id);
  int32_t prepare_input_tensor(std::vector<hbDNNTensor> &input_tensor, hbDNNHandle_t dnn_handle);

 private:
  hbDNNHandle_t dnn_handle_;
  hbPackedDNNHandle_t packed_dnn_handle_;

  std::deque<std::atomic_bool> idle_tensor_;
  std::vector<std::vector<hbDNNTensor>> output_tensors_;
  std::vector<std::vector<hbDNNTensor>> input_tensors_;
  int32_t input_tensor_type_;

  std::string postprocess_;

  int model_input_w_, model_input_h_;
  int model_output_w_, model_output_h_;
  int output_count_;

  int max_disp_ = 192;

  float focal_, baseline_;
  float uncertainty_th_ = 0.09;
};

#endif //STEREO_INCLUDE_STEREONET_PROCESS_H_
