//
// Created by zhy on 7/1/24.
//

#include <atomic>
#include <deque>
#include <string>
#include <opencv2/opencv.hpp>
#include <dnn/hb_dnn.h>

#include "image_conversion.h"

#ifndef STEREO_INCLUDE_STEREONET_PROCESS_H_
#define STEREO_INCLUDE_STEREONET_PROCESS_H_

#define ALIGN_16(v) ((v + 15) & ~15)

#define HB_CHECK_SUCCESS(value, errmsg)                          \
do {                                                             \
  /*value can be call of function*/                              \
  int32_t ret_code = value;                                      \
  if (ret_code != 0) {                                           \
      std::cout << "[BPU ERROR]" << errmsg << "error code: " << std::endl; \
  }                                                              \
} while (0);

#include <rclcpp/rclcpp.hpp>

struct ScopeProcessTime {
 public:
  ScopeProcessTime(const std::string &name) : name_(name) {
    //std::cout << name << " START" << std::endl;
    begin_ = std::chrono::system_clock::now();
  }
  ~ScopeProcessTime() {
    auto end = std::chrono::system_clock::now();
    const std::chrono::duration<float, std::milli> d = end - begin_;
    RCLCPP_DEBUG_STREAM(rclcpp::get_logger(""), name_ << ", consume: "
              << std::fixed << std::setprecision(3)
              << d.count() << "ms");
  }

 private:
  std::string name_;
  std::chrono::system_clock::time_point begin_;
};

struct StereonetProcess {

  enum StereonetErrorCode {
    OK = 0,
    TENSOR_BUSY = -1,
    DNN_ERROR = -2
  };

  const int MAX_PROCESS_COUNT = 5;
  StereonetProcess();

  int stereonet_init(const std::string &model_file_name);
  int stereonet_deinit ();

  int stereonet_inference(const cv::Mat &left_img,
                      const cv::Mat &right_img,
                      bool is_nv12,
                      std::vector<float> &points);

  void get_depth_width_height(int &width, int &height) const {
    width  = model_output_w_;
    height = model_output_h_;
  }

  void get_input_width_height(int &width, int &height) const {
    width  = model_input_w_;
    height = model_input_h_;
  }

 private:
  int get_idle_tensor();
  int set_tensor_idle(int tensor_id);

 private:
  hbDNNHandle_t		dnn_handle_;
  hbPackedDNNHandle_t packed_dnn_handle_;

  std::deque<std::atomic_bool> idle_tensor_;
  std::vector<std::vector<hbDNNTensor>> output_tensors_;
  std::vector<std::vector<hbDNNTensor>> input_tensors_;

  int model_input_w_, model_input_h_;
  int model_output_w_, model_output_h_;
  int output_count_;

  float focal_, baseline_;
};

#endif //STEREO_INCLUDE_STEREONET_PROCESS_H_
