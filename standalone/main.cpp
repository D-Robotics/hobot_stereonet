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

/**
 * @brief global variable for control c signal
 *  if set to false, the program will exit
 */
static std::atomic<bool> g_running{true};

/**
 * @brief control c signal handler
 * @param sig Signal number
 * @return void
 */
void signal_handler(int) {
  g_running = false;
}

/**
 * @brief Input data structure used for input data queue
 */
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

/**
 * @brief Publish data structure used for pub data queue
 */
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

/**
 * @brief StereoNetNode class for StereoNet model inference
 * This class is used to initialize and manage the StereoNet model inference process.
 */
class StereoNetNode {
public:
  /**
   * @brief Constructor
   * @param model_path Model path
   * @param infer_thread_num Inference thread number
   * @param algo_fps Algorithm fps
   * @param uncertainty_th Uncertainty threshold
   * @param post_version Postprocess version
   * @return void
   */
  StereoNetNode(const std::string &model_path, int infer_thread_num, double algo_fps = 30.0f,
                float uncertainty_th = -0.10, std::string post_version = "auto") {
    // member variables
    infer_thread_num_ = infer_thread_num;
    algo_fps_ = algo_fps;
    uncertainty_th_ = uncertainty_th;
    post_version_ = post_version;

    // stereonet process
    stereonet_process_ = std::make_shared<stereonet::StereonetProcess>();
    stereonet_process_->init(model_path, post_version);

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

  /**
   * @brief Destructor
   */
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
  /**
   * @brief This function is used to simulate the camera's acquisition of left and right images
   * @return void
   */
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
      std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(1000.0 / algo_fps_)));
    }
  }

  /**
   * @brief This function is used to infer the model
   * @return void
   */
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
          InferenceHandle handle;
          stereonet_process_->forward(left_img_nv12, right_img_nv12, handle);
          postprocess_thread_pool_ptr_->detach_task([this, handle, input_data]() {
            // postprocess
            cv ::Mat disp, uncert, depth;
            stereonet_process_->postprocess_out_disp_depth(handle, uncertainty_th_, camera_intrinsic_, disp, uncert,
                                                           depth);

            // enquque
            while (pub_data_queue_.size_approx() >= infer_thread_num_) {
              LOG_INFO(nullptr, "=> drop one pub data");
              std::shared_ptr<PubData> drop;
              pub_data_queue_.try_dequeue(drop);
            }
            pub_data_queue_.enqueue(
                std::make_shared<PubData>(input_data->timestamp, input_data->left_img, disp, depth));
          });
        }
      }
    }
  }

  /**
   * @brief This function is used to count resource consumption and save the results
   * @return void
   */
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
            // save disp
            cv::imwrite("./result/disp_" + std::to_string(pub_data->timestamp) + ".pfm", pub_data->disp);
            // save depth
            cv::imwrite("./result/depth_" + std::to_string(pub_data->timestamp) + ".png", pub_data->depth);
            // save visual
            cv::Mat visual_img;
            stereonet_process_->convert_visual_img(pub_data->left_img, pub_data->disp, pub_data->depth,
                                                   camera_intrinsic_, visual_img);
            cv::imwrite("./result/visual_" + std::to_string(pub_data->timestamp) + ".jpg", visual_img);
            // save pointcloud
            std::vector<stereonet::PointXYZRGB> pointcloud;
            stereonet_process_->depth_to_pointcloud_rgb(pub_data->depth, pub_data->left_img, camera_intrinsic_,
                                                        pointcloud);
            stereonet_process_->dump_pcd_file_rgb("./result/pointcloud_" + std::to_string(pub_data->timestamp) + ".pcd",
                                                  pointcloud);
          });
        }
      }
    }
  }

  /**
   * @brief Read camera intrinsic from file
   * @param file_path File path
   * @param intrinsic Camera intrinsic
   * @return true if read successfully, false otherwise
   */
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

  // stereonet process
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

  // uncertainty threshold
  float uncertainty_th_ = -0.10;
  // postprocess version
  std::string post_version_ = "auto";

  // algorithm fps
  double algo_fps_ = 30.0f;
};

void print_help(const char *prog_name) {
  std::cout << R"(Usage:)" << prog_name << R"( [model_path] [infer_thread_num] [uncertainty_th] [post_version]

Arguments:
  model_path         Path to stereo model (.bin)
                     default: ./model/DStereoV2.4_int16.bin
  infer_thread_num   Inference thread number
                     default: 1
  algo_fps           Algorithm fps
                     default: 30.0
  uncertainty_th     Uncertainty threshold
                     default: -0.10
  post_version       Postprocess version: auto | v2.0 | v2.1 | v2.2 | v2.3 | v2.4 | v2.4_uncert
                     default: auto

Examples: )" << prog_name
            << R"( ./model/DStereoV2.4_int16.bin)" << std::endl;
}

int main(int argc, char **argv) {
  // help
  if (argc > 1) {
    std::string arg1(argv[1]);
    if (arg1 == "-h" || arg1 == "--help") {
      print_help(argv[0]);
      return 0;
    }
  }

  // control c signal
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  // parse arguments
  std::string model_path = "./model/DStereoV2.4_int16.bin";
  int infer_thread_num = 1;
  float uncertainty_th = -0.10;
  std::string post_version = "auto";
  double algo_fps = 30.0f;
  if (argc > 1) model_path = argv[1];
  if (argc > 2) infer_thread_num = std::stoi(argv[2]);
  if (argc > 3) algo_fps = std::stod(argv[3]);
  if (argc > 4) uncertainty_th = std::stof(argv[4]);
  if (argc > 5) post_version = argv[5];

  if (!std::filesystem::exists(model_path)) {
    LOG_ERROR(nullptr, "=> model file not exist: " << model_path);
    return -1;
  }

  // init StereoNetNode
  auto stereonet_node =
      std::make_shared<StereoNetNode>(model_path, infer_thread_num, algo_fps, uncertainty_th, post_version);

  // spin
  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  // release StereoNetNode
  stereonet_node.reset();

  return 0;
}