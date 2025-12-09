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

#include <iostream>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "blockingconcurrentqueue.h"
// =============== stereonet ===============
#include "log_macros.h"
#include "camera_intrinsic.h"
#include "stereonet_process.h"
#include "img_convert_utils.h"
// =============== stereonet ===============

class StereoNetNode {
public:
  StereoNetNode(const std::string &model_path) {
    // stereonet process
    stereonet_process_ = std::make_shared<stereonet::StereonetProcess>();
    stereonet_process_->init(model_path);

    // thread
    capture_thread_ = std::thread(&StereoNetNode::capture_function, this);
    for (int i = 0; i < infer_thread_num_; ++i) {
      infer_threads_.emplace_back(&StereoNetNode::infer_function, this);
    }
    publish_thread_ = std::thread(&StereoNetNode::publish_function, this);
  }
  ~StereoNetNode() = default;

private:
  void capture_function() {
    // read left and right images
    cv::Mat left_img = cv::imread("./img/000001-left.png", cv::IMREAD_COLOR);
    cv::Mat right_img = cv::imread("./img/000001-right.png", cv::IMREAD_COLOR);

    // read model input size
    int model_input_w = 0, model_input_h = 0;
    stereonet_process_->get_model_input_size(model_input_w, model_input_h);

    // resize left and right images to model input size
    LOG_INFO(nullptr, "=> left img size [" << left_img.cols << ", " << left_img.rows << "], right img size ["
                                           << right_img.cols << ", " << right_img.rows << "], need reszie to ["
                                           << model_input_w << ", " << model_input_h << "]");
    cv::Mat left_img_resize, right_img_resize;
    cv::resize(left_img, left_img_resize, cv::Size(model_input_w, model_input_h));
    cv::resize(right_img, right_img_resize, cv::Size(model_input_w, model_input_h));

    // read camera intrinsic
    if (readCameraIntrinsicFromFile("./img/camera_intrinsic.txt", camera_intrinsic_)) {
      LOG_INFO(nullptr, "=> before resize, cam intrinsic [fx, fy, cx, cy, baseline]: ["
                            << camera_intrinsic_.fx << ", " << camera_intrinsic_.fy << ", " << camera_intrinsic_.cx
                            << ", " << camera_intrinsic_.cy << ", " << camera_intrinsic_.baseline << "]");
      camera_intrinsic_.fx = camera_intrinsic_.fx * model_input_w / left_img.cols;
      camera_intrinsic_.fy = camera_intrinsic_.fy * model_input_h / left_img.rows;
      camera_intrinsic_.cx = camera_intrinsic_.cx * model_input_w / left_img.cols;
      camera_intrinsic_.cy = camera_intrinsic_.cy * model_input_h / left_img.rows;
      LOG_INFO(nullptr, "=> after resize, cam intrinsic [fx, fy, cx, cy, baseline]: ["
                            << camera_intrinsic_.fx << ", " << camera_intrinsic_.fy << ", " << camera_intrinsic_.cx
                            << ", " << camera_intrinsic_.cy << ", " << camera_intrinsic_.baseline << "]");
    }

    // convert to nv12
    std::vector<uint8_t> left_img_nv12, right_img_nv12;
    size_t model_input_nv12_size = model_input_w * model_input_h * 3 / 2;
    left_img_nv12.resize(model_input_nv12_size);
    right_img_nv12.resize(model_input_nv12_size);
    ImgConvertUtils::bgr_mat_to_nv12(left_img_resize, left_img_nv12.data());
    ImgConvertUtils::bgr_mat_to_nv12(right_img_resize, right_img_nv12.data());

    // enqueue
    while (true) {
      while (input_image_queue_.size_approx() >= 10) {
        std::pair<std::vector<uint8_t>, std::vector<uint8_t>> drop;
        input_image_queue_.try_dequeue(drop);
      }
      input_image_queue_.enqueue(std::make_pair(left_img_nv12, right_img_nv12));
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  void infer_function() {
    while (true) {
      std::pair<std::vector<uint8_t>, std::vector<uint8_t>> input_data;
      if (input_image_queue_.wait_dequeue_timed(input_data, std::chrono::milliseconds(100))) {
        // infer by multi-thread
        auto left_img_nv12 = input_data.first;
        auto right_img_nv12 = input_data.second;
        cv::Mat disp, uncert;
        stereonet_process_->forward(left_img_nv12, right_img_nv12, uncertainty_th_, disp, uncert);
        cv::Mat depth;
        stereonet_process_->disp_to_depth(disp, depth, camera_intrinsic_.fx, camera_intrinsic_.baseline);

        // enquque
        while (pub_data_queue_.size_approx() >= infer_thread_num_) {
          std::pair<cv::Mat, cv::Mat> drop;
          pub_data_queue_.try_dequeue(drop);
        }
        pub_data_queue_.enqueue(std::make_pair(disp, depth));
      }
    }
  }

  void publish_function() {
    // calc time cost
    auto now = std::chrono::system_clock::now();
    int count = 0;
    while (true) {
      // dequeue
      std::pair<cv::Mat, cv::Mat> pub_data;
      if (pub_data_queue_.wait_dequeue_timed(pub_data, std::chrono::milliseconds(100))) {
        cv::Mat &disp = pub_data.first;
        cv::Mat &depth = pub_data.second;
        ++count;
        if (count == 100) {
          auto end = std::chrono::system_clock::now();
          auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - now).count();
          now = end;
          count = 0;
          // cv::imwrite("disp.pfm", disp);
          // cv::imwrite("depth.png", depth);
          LOG_INFO(nullptr, "=> save disp.pfm and depth.png, time cost: "
                                << std::fixed << std::setprecision(3) << (duration / 100.0)
                                << "ms, fps: " << std::setprecision(3) << 1000.0 / (duration / 100.0));
        }
      }
    }
  }

  bool readCameraIntrinsicFromFile(const std::string &file_path, stereonet::CameraIntrinsic &intrinsic) {
    std::ifstream infile(file_path);
    if (!infile.is_open()) {
      std::cerr << "Failed to open file: " << file_path << std::endl;
      return false;
    }

    std::string line;
    while (std::getline(infile, line)) {
      if (line.empty() || line[0] == '#') continue;

      std::istringstream ss(line);
      double fx, fy, cx, cy, baseline;
      if (ss >> fx >> fy >> cx >> cy >> baseline) {
        intrinsic.fx = fx;
        intrinsic.fy = fy;
        intrinsic.cx = cx;
        intrinsic.cy = cy;
        intrinsic.baseline = baseline;
        return intrinsic.is_valid();
      } else {
        std::cerr << "Failed to parse line: " << line << std::endl;
        return false;
      }
    }

    std::cerr << "No valid data found in file: " << file_path << std::endl;
    return false;
  }

  std::shared_ptr<stereonet::StereonetProcess> stereonet_process_;

  // thread
  std::thread capture_thread_;
  moodycamel::BlockingConcurrentQueue<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> input_image_queue_;
  std::vector<std::thread> infer_threads_;
  int infer_thread_num_ = 2;
  moodycamel::BlockingConcurrentQueue<std::pair<cv::Mat, cv::Mat>> pub_data_queue_;
  std::thread publish_thread_;

  // camera intrinsic
  stereonet::CameraIntrinsic camera_intrinsic_;

  float uncertainty_th_ = -0.10;
};

int main(int argc, char **argv) {
  std::string model_path = "./DStereoV2.6_int8.bin";
  if (argc > 1) {
    model_path = argv[1];
  }
  if (!std::filesystem::exists(model_path)) {
    LOG_ERROR(nullptr, "=> model file not exist: " << model_path);
    return -1;
  }

  auto stereonet_node = std::make_shared<StereoNetNode>(model_path);

  while (true) {
  }

  return 0;
}