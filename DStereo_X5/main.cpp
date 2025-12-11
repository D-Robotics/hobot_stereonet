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
#include "BS_thread_pool.hpp"
#include <atomic>
#include <csignal>
// =============== stereonet ===============
#include "log_macros.h"
#include "camera_intrinsic.h"
#include "stereonet_process.h"
#include "img_convert_utils.h"
#include "performance_record.h"
// =============== stereonet ===============

static std::atomic<bool> g_running{true};

void signal_handler(int) {
  g_running = false;
}

struct InputData {
  InputData(uint64_t timestamp, const cv::Mat &left_img, const std::vector<uint8_t> &left_img_nv12,
            const std::vector<uint8_t> &right_img_nv12)
      : timestamp(timestamp), left_img(left_img), left_img_nv12(left_img_nv12), right_img_nv12(right_img_nv12) {
  }
  // timestamp
  uint64_t timestamp; // ms
  cv::Mat left_img;
  std::vector<uint8_t> left_img_nv12;
  std::vector<uint8_t> right_img_nv12;
};

struct PubData {
  PubData(uint64_t timestamp, const cv::Mat &left_img, const cv::Mat &disp, const cv::Mat &depth)
      : timestamp(timestamp), left_img(left_img), disp(disp), depth(depth) {
  }
  // timestamp
  uint64_t timestamp; // ms
  cv::Mat left_img;
  cv::Mat disp;
  cv::Mat depth;
};

class StereoNetNode {
public:
  StereoNetNode(const std::string &model_path, int infer_thread_num, float uncertainty_th = -0.10) {
    infer_thread_num_ = infer_thread_num;
    uncertainty_th_ = uncertainty_th;
    // stereonet process
    stereonet_process_ = std::make_shared<stereonet::StereonetProcess>();
    stereonet_process_->init(model_path);

    // thread
    capture_thread_ = std::thread(&StereoNetNode::capture_function, this);
    for (int i = 0; i < infer_thread_num_; ++i) {
      infer_threads_.emplace_back(&StereoNetNode::infer_function, this);
    }
    publish_thread_ = std::thread(&StereoNetNode::publish_function, this);
    postprocess_thread_pool_ptr_ = std::make_unique<BS::thread_pool<>>(1);
    save_thread_pool_ptr_ = std::make_unique<BS::thread_pool<>>(1);

    // performance
    performance_writer::Get();
  }

  ~StereoNetNode() {
    LOG_INFO(nullptr, "=> release StereoNetNode");
    g_running = false;
    if (capture_thread_.joinable()) capture_thread_.join();
    for (auto &t : infer_threads_)
      if (t.joinable()) t.join();
    if (publish_thread_.joinable()) publish_thread_.join();
    if (postprocess_thread_pool_ptr_) {
      postprocess_thread_pool_ptr_->wait();
      postprocess_thread_pool_ptr_.reset();
    }
    if (save_thread_pool_ptr_) {
      save_thread_pool_ptr_->wait();
      save_thread_pool_ptr_.reset();
    }
  }

private:
  void capture_function() {
    // read left and right images
    cv::Mat left_img = cv::imread("./img/left000000.png", cv::IMREAD_COLOR);
    cv::Mat right_img = cv::imread("./img/right000000.png", cv::IMREAD_COLOR);

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

    // enqueue: simulate 30fps camera
    while (g_running) {
      while (input_image_queue_.size_approx() >= 1) {
        // LOG_INFO(nullptr, "=> drop one input image");
        std::shared_ptr<InputData> drop;
        input_image_queue_.try_dequeue(drop);
      }
      // timestamp ms
      auto timestamp =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
              .count();
      std::shared_ptr<InputData> input_data =
          std::make_shared<InputData>(timestamp, left_img_resize, left_img_nv12, right_img_nv12);
      input_image_queue_.enqueue(input_data);
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
  }

  void infer_function() {
    while (g_running) {
      std::shared_ptr<InputData> input_data;
      if (input_image_queue_.wait_dequeue_timed(input_data, std::chrono::milliseconds(100))) {
        // infer by multi-thread
        if (infer_thread_num_ > 1) {
          LOG_INFO_ONCE(nullptr, "=> infer by multi-thread: " << infer_thread_num_);

          // infer
          std::vector<uint8_t> left_img_nv12 = input_data->left_img_nv12;
          std::vector<uint8_t> right_img_nv12 = input_data->right_img_nv12;
          cv::Mat disp, uncert;
          stereonet_process_->forward_sync(left_img_nv12, right_img_nv12, uncertainty_th_, disp, uncert);
          cv::Mat depth;
          stereonet_process_->disp_to_depth(disp, depth, camera_intrinsic_);

          // enquque
          while (pub_data_queue_.size_approx() >= infer_thread_num_) {
            LOG_INFO(nullptr, "=> drop one pub data");
            std::shared_ptr<PubData> drop;
            pub_data_queue_.try_dequeue(drop);
          }
          pub_data_queue_.enqueue(std::make_shared<PubData>(input_data->timestamp, input_data->left_img, disp, depth));
        } else {
          LOG_INFO_ONCE(nullptr, "=> infer by single thread");

          // infer
          uint8_t *left_img_nv12 = input_data->left_img_nv12.data();
          uint8_t *right_img_nv12 = input_data->right_img_nv12.data();
          int idle_tensor_id = 0;
          stereonet_process_->forward(left_img_nv12, right_img_nv12, idle_tensor_id);
          postprocess_thread_pool_ptr_->detach_task([this, idle_tensor_id, input_data]() {
            // postprocess
            int width, height;
            stereonet_process_->get_model_input_size(width, height);
            std::vector<float> disp, uncert;
            std::vector<uint16_t> depth;
            disp.resize(width * height);
            uncert.resize(width * height);
            depth.resize(width * height);
            stereonet_process_->postprocess_out_disp_depth(idle_tensor_id, uncertainty_th_, camera_intrinsic_,
                                                           disp.data(), uncert.data(), depth.data());
            cv::Mat disp_mat(height, width, CV_32FC1);
            memcpy(disp_mat.data, disp.data(), width * height * sizeof(float));
            cv::Mat depth_mat(height, width, CV_16UC1);
            memcpy(depth_mat.data, depth.data(), width * height * sizeof(uint16_t));

            // enquque
            while (pub_data_queue_.size_approx() >= infer_thread_num_) {
              LOG_INFO(nullptr, "=> drop one pub data");
              std::shared_ptr<PubData> drop;
              pub_data_queue_.try_dequeue(drop);
            }
            pub_data_queue_.enqueue(
                std::make_shared<PubData>(input_data->timestamp, input_data->left_img, disp_mat, depth_mat));
          });
        }
      }
    }
  }

  void publish_function() {
    int count = 0;
    while (g_running) {
      // dequeue
      std::shared_ptr<PubData> pub_data;
      if (pub_data_queue_.wait_dequeue_timed(pub_data, std::chrono::milliseconds(100))) {
        uint64_t now_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        uint64_t latency = now_ms - pub_data->timestamp;
        performance_writer::Get()->record_performance(latency);
        ++count;
        if (count == 100) {
          count = 0;

          // print performance
          auto fps = performance_writer::Get()->get_fps();
          auto cpu_usage = performance_writer::Get()->get_cpu_usage();
          auto bpu_usage = performance_writer::Get()->get_bpu_usage();
          LOG_INFO(nullptr, "=> fps: " << std::fixed << std::setprecision(2) << fps << ", latency: " << latency
                                       << "ms, cpu_usage: " << cpu_usage << "%, bpu_usage: " << bpu_usage << "%");

          // save result async
          save_thread_pool_ptr_->detach_task([this, pub_data]() {
            // LOG_INFO(nullptr, "=> save disp.pfm / depth.png / pointcloud.pcd");
            cv::imwrite("disp_" + std::to_string(pub_data->timestamp) + ".pfm", pub_data->disp);
            cv::imwrite("depth_" + std::to_string(pub_data->timestamp) + ".png", pub_data->depth);
            cv::Mat visual_img;
            stereonet_process_->convert_visual_img(pub_data->left_img, pub_data->disp, pub_data->depth, visual_img);
            cv::imwrite("visual_" + std::to_string(pub_data->timestamp) + ".png", visual_img);
            // std::vector<stereonet::PointXYZ> pointcloud;
            // stereonet_process_->depth_to_pointcloud(pub_data->depth, camera_intrinsic_, pointcloud);
            // stereonet_process_->dump_pcd_file("pointcloud.pcd", pointcloud);
            std::vector<stereonet::PointXYZRGB> pointcloud;
            stereonet_process_->depth_to_pointcloud_rgb(pub_data->depth, pub_data->left_img, camera_intrinsic_,
                                                        pointcloud);
            stereonet_process_->dump_pcd_file_rgb("pointcloud_" + std::to_string(pub_data->timestamp) + ".pcd",
                                                  pointcloud);
          });
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
  moodycamel::BlockingConcurrentQueue<std::shared_ptr<InputData>> input_image_queue_;
  std::vector<std::thread> infer_threads_;
  int infer_thread_num_ = 1;
  moodycamel::BlockingConcurrentQueue<std::shared_ptr<PubData>> pub_data_queue_;
  std::thread publish_thread_;
  std::unique_ptr<BS::thread_pool<>> postprocess_thread_pool_ptr_ = nullptr;
  std::unique_ptr<BS::thread_pool<>> save_thread_pool_ptr_ = nullptr;

  // camera intrinsic
  stereonet::CameraIntrinsic camera_intrinsic_;

  float uncertainty_th_ = -0.10;
};

int main(int argc, char **argv) {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  std::string model_path = "./model/DStereoV2.4_int16.bin";
  int infer_thread_num = 1;
  float uncertainty_th = -0.10;
  if (argc > 1) {
    model_path = argv[1];
  }
  if (argc > 2) {
    infer_thread_num = std::stoi(argv[2]);
  }
  if (argc > 3) {
    uncertainty_th = std::stof(argv[3]);
  }

  if (!std::filesystem::exists(model_path)) {
    LOG_ERROR(nullptr, "=> model file not exist: " << model_path);
    return -1;
  }

  auto stereonet_node = std::make_shared<StereoNetNode>(model_path, infer_thread_num, uncertainty_th);

  // ctrl + c signal handler
  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  // release StereoNetNode
  stereonet_node.reset();

  return 0;
}