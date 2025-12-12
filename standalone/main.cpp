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

#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <memory>
#include <thread>
#include <csignal>
#include <filesystem>

#include "stereonet_process.h"
#include "image_conversion.h"
#include "performance_record.h"
#include "blockqueue.h"
#include "metrics_process.h"

#include "data_loader.h"

std::atomic_bool stop_flag{false};

struct CameraParameter {
  float camera_cx, camera_cy, camera_fx, camera_fy, base_line;
};

struct Point {
  float X, Y, Z;
};

struct StereoResult {
  uint64_t ts;
  cv::Mat model_depth;
  std::vector<Point> Points;
};

struct InferenceData {
  uint64_t ts;
  cv::Mat left_image, right_image;
  InferenceData(uint64 timestamp,
                const std::string &left_file, const std::string &right_file) {
    ts = timestamp;
    left_image = cv::imread(left_file);
    right_image = cv::imread(right_file);
    if (left_image.empty() || right_image.empty()) {
      std::cerr << "left_file: " << left_file << ", or right_file: "
                << right_file << " is not exist!"<< std::endl;
      return;
    }
  }

  bool is_valid() {
    return !left_image.empty() && !right_image.empty();
  }

  InferenceData() {}
};

struct DataWrapper {
  DataWrapper() {}
  DataWrapper(InferenceData &infer_data,
              std::vector<float> &disp_points,
              StereoResult &stereo_res) {
    inference_data = infer_data;
    disparity_points = std::move(disp_points);
    stereo_result = stereo_res;
  }
  std::vector<float> disparity_points;
  StereoResult stereo_result;
  InferenceData inference_data;
};

struct StereoDemo {
  int init(const std::string &stereonet_model_file_path,
           const std::string &post_version,
           int max_disp,
           float uncertainty_th = 0.1);
  int deinit();

  int get_image(InferenceData &infer_data);
  int get_inference_result(InferenceData &infer_data,
                           StereoResult &stereo_result,
                           std::vector<float> &points,
                           CameraParameter &camera_parameter);

  void get_blind_area(float fx, float base_line, float& blind_area) {
    stereonet_process_->get_blind_area(fx, base_line, blind_area);
  }

 private:
  std::shared_ptr<StereonetProcess> stereonet_process_;

 private:
  int model_input_h_, model_input_w_;
  int model_output_h_, model_output_w_;

};

int StereoDemo::init(const std::string &stereonet_model_file_path,
                     const std::string &post_version,
                     int max_disp,
                     float uncertainty_th) {
  int ret = 0;
  stereonet_process_ = std::make_shared<StereonetProcess>();
  //  The `uncertainty_th` ranges from 0.0 to 1.0
  //  — the closer it is to 0.0, the more aggressive the filtering.
  ret = stereonet_process_->stereonet_init(
      stereonet_model_file_path, max_disp, post_version, uncertainty_th);
  if (ret != 0) {
    std::cerr << "stereonet init failed!" << std::endl;
    return -1;
  }

  stereonet_process_->get_input_width_height(
      model_input_w_, model_input_h_);
  stereonet_process_->get_depth_width_height(
      model_output_w_, model_output_h_);
  std::cout << "model_input_w: " << model_input_w_ << ", model_input_h: " << model_input_h_ << std::endl;

  return 0;
}

std::string get_pure_file_name(const std::string &file_name) {
  std::string file;
  std::filesystem::path path(file_name);
  if (path.extension() == ".png") {
    file = path.stem().string();
  } else {
    file = file_name;
  }
  return file;
}

int StereoDemo::deinit() {
  stereonet_process_->stereonet_deinit();
  return 0;
}

int StereoDemo::get_inference_result(
    InferenceData &infer_data,
    StereoResult &stereo_result,
    std::vector<float> &points,
    CameraParameter &camera_parameter) {
  int img_origin_height, img_origin_width;


  if (infer_data.left_image.empty() || infer_data.right_image.empty()) {
    std::cerr << "InferenceData is empty" << std::endl;
    return -1;
  }

  img_origin_height = infer_data.left_image.rows;
  img_origin_width = infer_data.left_image.cols;

  if (stereonet_process_->stereonet_inference(
      infer_data.left_image, infer_data.right_image, false, points) != 0) {
    std::cerr << "Inferenc Failed" << std::endl;
    return -1;
  }

  cv::Mat model_depth_img = cv::Mat(model_output_h_, model_output_w_, CV_16UC1);
  uint16_t *depth_data = (uint16_t *)model_depth_img.data;
  float factor = 1000 * (camera_parameter.camera_fx * camera_parameter.base_line);
  float min_disparity = 50000 / factor;
  float32x4_t min_disparity_vec = vdupq_n_f32(min_disparity);
  //uint16x4_t zero_vec_u16 = vget_low_u16(vdupq_n_u16(0));
  float32x4_t factor_vector = vdupq_n_f32(factor);

  for (uint32_t i = 0; i < points.size(); i += 4) {
    float32x4_t points_vec = vmaxq_f32(vld1q_f32(&points[i]), min_disparity_vec);
    float32x4_t depth_vec = vdivq_f32(factor_vector, points_vec);
    uint16x4_t depth_int16_vec = vmovn_u32(vcvtq_u32_f32(depth_vec));
    vst1_u16(&depth_data[i], depth_int16_vec);
  }

  stereo_result.model_depth = model_depth_img;
  stereo_result.ts = infer_data.ts;
  stereo_result.Points.clear();
  stereo_result.Points.reserve(model_output_h_ * model_output_w_ / 4);
  for (int y = 0; y < model_output_h_; y += 2) {
    float fy = (camera_parameter.camera_cy  - y) / camera_parameter.camera_fy;
    for (int x = 0; x < model_output_w_; x += 2) {
      float depth = depth_data[y * model_output_w_ + x] / 1000.0f;
      if (depth > 5) continue;
      float X = (camera_parameter.camera_cx - x) / camera_parameter.camera_fx * depth;
      float Y = fy * depth;
      Point point;
      point.X = X;
      point.Y = Y;
      point.Z = depth;
      stereo_result.Points.push_back(point);
    }
  }
  return 0;
}

void signal_handler(int signo) {
  if (signo == SIGINT) {
    stop_flag = true;
    std::cout << "\nrecv SIGINT, exit!" << std::endl;
  }
}

int dump_depth_in_mm(StereoResult &stereo_result, const std::string &file_name = "") {
  auto ts = stereo_result.ts;
  cv::Mat depth;
  switch (stereo_result.model_depth.type()) {
    case CV_64FC1:
    case CV_32FC1:
      depth = (stereo_result.model_depth * 1000);
      depth.convertTo(depth, CV_16UC1);
      break;
    case CV_16UC1:
      depth = stereo_result.model_depth;
      break;
  }
  if (file_name.empty()) {
    cv::imwrite("./result/" + std::to_string(ts) +"_depth.png", depth);
  } else {
    cv::imwrite("./result/" + get_pure_file_name(file_name) +"_depth.png", depth);
  }
  return 0;
}

int dump_disparity(StereoResult &stereo_result, cv::Mat& disparity, const std::string &file_name = "") {
  auto ts = stereo_result.ts;
  if (file_name.empty()) {
    cv::imwrite("./result/" + std::to_string(ts) +"_disparity.pfm", disparity);
  } else {
    cv::imwrite("./result/" + get_pure_file_name(file_name) +"_disparity.pfm", disparity);
  }
  return 0;
}

void dump_one_point_disparity(
    InferenceData &infer_data,
    StereoResult &stereo_result,
    std::vector<float>&points,
    int x, int y) {
  auto ts = stereo_result.ts;
  cv::Mat left_image = infer_data.left_image.clone();
  cv::Mat right_image = infer_data.right_image.clone();
  cv::Mat combine;
  auto disparity = points[y * left_image.cols + x];
  cv::circle(left_image, cv::Point(x, y), 10,
             cv::Scalar(255, 0, 0), 3);
  cv::circle(right_image, cv::Point(x - disparity, y), 10,
             cv::Scalar(255, 0, 0), 3);
  cv::putText(left_image, "diff: " + std::to_string(disparity),
              cv::Point2i(20,20),
              cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(255, 255, 255), 2);
  cv::hconcat(left_image, right_image, combine);
  cv::imwrite("./result/" +  std::to_string(ts)  + "_one_point_disparity.jpg", combine);
}

void dump_pcd_file(StereoResult &stereo_result) {
  auto ts = stereo_result.ts;
  const std::string& filename =  "./result/" + std::to_string(ts) + ".pcd";
  std::ofstream ofs(filename);
  const std::vector<Point> & points = stereo_result.Points;
  if (!ofs.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
  ofs << "VERSION 0.7\n";
  ofs << "FIELDS x y z\n";
  ofs << "SIZE 4 4 4\n";
  ofs << "TYPE F F F\n";
  ofs << "COUNT 1 1 1\n";
  ofs << "WIDTH " << points.size() << "\n";
  ofs << "HEIGHT 1\n";
  ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
  ofs << "POINTS " << points.size() << "\n";
  ofs << "DATA ascii\n";
  for (const auto& point : points) {
    ofs << point.X << " " << point.Y << " " << point.Z << "\n";
  }
  ofs.close();
}

int dump_visual_image(InferenceData &infer_data,
                      StereoResult &stereo_result,
                      std::vector<float>&points,
                      const std::string &file_name = "") {
  auto ts = stereo_result.ts;
  const cv::Mat &depth_img = stereo_result.model_depth;
  cv::Mat bgr_image = infer_data.left_image;
  cv::Mat visual_img(bgr_image.rows * 2, bgr_image.cols, CV_8UC3);
  bgr_image.copyTo(visual_img(cv::Rect(0, 0, bgr_image.cols, bgr_image.rows)));

  cv::Mat feat_mat(bgr_image.rows, bgr_image.cols, CV_32F, const_cast<float *>(points.data()));
  dump_disparity(stereo_result, feat_mat, file_name);
  cv::Mat feat_visual;
  feat_mat.convertTo(feat_visual, CV_8U, 4, 0);
  //  cv::convertScaleAbs(feat_visual, feat_visual, 2);
  cv::applyColorMap(feat_visual,
                    visual_img(cv::Rect(0, bgr_image.rows, bgr_image.cols, bgr_image.rows)),
                    cv::COLORMAP_JET);

  int step_num = 6;
  int x_step = bgr_image.cols / step_num;
  int y_step = bgr_image.rows / step_num;

  for (int i = 1; i < step_num; i++) {
    for (int j = 1; j < step_num; j++) {
      cv::line(visual_img, cv::Point2i(0, bgr_image.rows + i * y_step),
               cv::Point2i(bgr_image.cols, bgr_image.rows + i * y_step),
               cv::Scalar(255, 255, 255), 1);
      cv::line(visual_img, cv::Point2i(j * x_step, bgr_image.rows),
               cv::Point2i(j * x_step, bgr_image.rows * 2),
               cv::Scalar(255, 255, 255), 1);

      cv::line(visual_img, cv::Point2i(0, i * y_step),
               cv::Point2i(bgr_image.cols, i * y_step),
               cv::Scalar(255, 255, 255), 1);
      cv::line(visual_img, cv::Point2i(j * x_step, 0),
               cv::Point2i(j * x_step, bgr_image.rows),
               cv::Scalar(255, 255, 255), 1);
      uint16_t Z;
      double distance;
      switch (depth_img.type()) {
        case CV_32FC1:
          distance = depth_img.at<float>(i * y_step, j * x_step);
          Z = distance;
          break;
        case CV_16UC1:
          Z = depth_img.at<uint16_t>(i * y_step, j * x_step);
          distance = static_cast<double>(Z) / 1000.0;
          break;
        case CV_64FC1:
          distance = depth_img.at<double>(i * y_step, j * x_step);
          Z = distance;
          break;
      }

      // distance = points[i * y_step * bgr_image.cols + j * x_step];

      std::ostringstream ss;
      ss << std::fixed << std::setprecision(2) << distance << "m";
      cv::putText(visual_img, ss.str(), cv::Point2i(j * x_step + 3,
                                                    bgr_image.rows + i * y_step - 3),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(255, 255, 255), 2);

      cv::putText(visual_img, ss.str(), cv::Point2i(j * x_step + 3,
                                                    i * y_step - 3),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(255, 255, 255), 2);
    }
  }
  if (file_name.empty()) {
    cv::imwrite("./result/" + std::to_string(ts) +"_visual.jpg", visual_img);
  } else {
    cv::imwrite("./result/" + get_pure_file_name(file_name) +"_visual.jpg", visual_img);
  }
  return 0;
}

int main_V2_4(int argc, char **argv) {
  int ret;
  std::string stereonet_model_file_path = "./config/DStereoV2.4_int16.bin";
  std::string left_file = "./left000000.png", right_file = "./right000000.png";
  StereoDemo stereo_demo;
  StereoResult stereo_result;
  CameraParameter camera_parameter;
  std::vector<float> disparity_points;

  InferenceData infer_data(std::chrono::high_resolution_clock::now().time_since_epoch().count(),
                           left_file, right_file);

  camera_parameter.camera_fx = 208.503;
  camera_parameter.camera_fy = 208.503;
  camera_parameter.camera_cx = 316.668;
  camera_parameter.camera_cy = 175.107;
  camera_parameter.base_line = 0.0804746;

  signal(SIGINT, signal_handler);

  ret = stereo_demo.init(stereonet_model_file_path, "v2.4", 192);
  if (ret != 0) {
    std::cerr << "model init failed!" << std::endl;
    return -1;
  }
  std::cout << "model init succeed!" << std::endl;

  ret = stereo_demo.get_inference_result(infer_data,
                                         stereo_result, disparity_points, camera_parameter);
  if (ret == 0) {
    dump_visual_image(infer_data, stereo_result, disparity_points);
    dump_pcd_file(stereo_result);
    dump_depth_in_mm(stereo_result);
  } else {
    std::cerr << "get_inference_result failed!" << std::endl;
  }

  stereo_demo.deinit();
  return 0;
}

int main_V2_4_uncertainty(int argc, char **argv) {
  int ret;
  float blind_area;
  int print_count = 0;
  std::shared_ptr<std::thread> save_thread;
  blockqueue<DataWrapper> data_que;
  std::string stereonet_model_file_path = "./config/DStereoV2.4_int16_uncertainty.bin";
  std::string left_file = "./left000000.png", right_file = "./right000000.png";
  StereoDemo stereo_demo;
  DataWrapper data_wrapper;
  CameraParameter camera_parameter;

  InferenceData infer_data(std::chrono::high_resolution_clock::now().time_since_epoch().count(),
                           left_file, right_file);
  performance_writer::Get();

  camera_parameter.camera_fx = 208.503;
  camera_parameter.camera_fy = 208.503;
  camera_parameter.camera_cx = 316.668;
  camera_parameter.camera_cy = 175.107;
  camera_parameter.base_line = 0.0804746;

  signal(SIGINT, signal_handler);

  ret = stereo_demo.init(stereonet_model_file_path, "v2.4", 192);
  if (ret != 0) {
    std::cerr << "model init failed!" << std::endl;
    return -1;
  }
  std::cout << "model init succeed!" << std::endl;

  stereo_demo.get_blind_area(camera_parameter.camera_fx, camera_parameter.base_line, blind_area);
  std::cout << "blind area is " << blind_area << "m" << std::endl;

  save_thread = std::make_shared<std::thread>(
      [&]() {
        while (!stop_flag) {
          DataWrapper data_wrapper;
          if (data_que.get(data_wrapper)) {
            dump_visual_image(data_wrapper.inference_data,
                              data_wrapper.stereo_result, data_wrapper.disparity_points);
            dump_pcd_file(data_wrapper.stereo_result);
            dump_depth_in_mm(data_wrapper.stereo_result);
          }
        }
      });

  while (!stop_flag) {
    data_wrapper.inference_data = infer_data;
    data_wrapper.disparity_points.clear();
    auto start = std::chrono::high_resolution_clock::now();
    ret = stereo_demo.get_inference_result(infer_data, data_wrapper.stereo_result,
                                           data_wrapper.disparity_points, camera_parameter);
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (ret == 0) {
      if (data_que.size() > 1) {
        data_que.pop_front();
      }
      data_que.put(data_wrapper);
      performance_writer::Get()->record_performance(latency);
      if (++print_count > 10) {
        std::cout << "fps: " << performance_writer::Get()->get_fps()
                  << ", latency: " << latency
                  <<"ms, cpu_usage: " << performance_writer::Get()->get_cpu_usage()
                  <<"%, bpu_usage: " << performance_writer::Get()->get_bpu_usage() << "%." << std::endl;
        print_count = 0;
      }
    } else {
      std::cerr << "get_inference_result failed!" << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }

  save_thread->join();
  data_que.clear();
  stereo_demo.deinit();
  return 0;
}

void calculate_metrics(const std::string &json_file, const std::string &data_path) {
  int index = 0;
  MetricsJsonWriter metrics_json_writer;
  StereoDemo stereo_demo;
  std::vector<StereoImageSet> stereo_image_sets;
  std::string stereonet_model_file_path = "./config/DStereoV23_int16_1110.bin";

  std::vector<double> epes, ffprs, fnprs, infinity_metrics,
                      bad_pixels_2, bad_pixels_4,
                      a99_0, a99_1, a99_2;

  std::vector<double> a99s_max(3, std::numeric_limits<double>::min());
  std::vector<int> a99s_max_index(3, -1);

  stereo_demo.init(stereonet_model_file_path, "v2.4", 192, -0.1);
  stereo_image_sets = StereoDataLoader::load(json_file);
  std::cout << "stereo_image_sets count: " << stereo_image_sets.size() << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  for (auto &stereo_image : stereo_image_sets) {
    std::string left_image_file = data_path + "/" + stereo_image.left_image_file;
    std::string right_image_file = data_path + "/" + stereo_image.right_image_file;
    std::string disparity_image_file = data_path + "/" + stereo_image.disparity_image_file;
    InferenceData infer_data(std::chrono::high_resolution_clock::now().time_since_epoch().count(),
                             left_image_file, right_image_file);
    cv::Mat gt_disparity = cv::imread(disparity_image_file, cv::IMREAD_UNCHANGED);
    if (stereo_image.is_valid() && infer_data.is_valid() && !gt_disparity.empty()) {
      CameraParameter camera_parameter;
      StereoResult stereo_result;
      std::vector<float> infer_disparity_points;
      cv::Mat infer_disparity;
      stereo_image.get_camera_parameter(
          camera_parameter.camera_fx, camera_parameter.camera_fy,
          camera_parameter.camera_cx, camera_parameter.camera_cy,
          camera_parameter.base_line);
      if (stereo_demo.get_inference_result(infer_data, stereo_result,
                                           infer_disparity_points, camera_parameter) == 0) {
        infer_disparity = cv::Mat(
            stereo_result.model_depth.rows, stereo_result.model_depth.cols,
            CV_32FC1, infer_disparity_points.data());

        double epe = MetricsProcess::calculateEPE(gt_disparity, infer_disparity);

        std::pair<double, double> bad_pixel = MetricsProcess::calculateBadPixels(
            gt_disparity, infer_disparity);

        double ffpr = MetricsProcess::calculateFFPR(gt_disparity, infer_disparity,
                                                    camera_parameter.camera_fx, camera_parameter.base_line);

        double fnpr = MetricsProcess::calculateFNPR(gt_disparity, infer_disparity,
                                                    camera_parameter.camera_fx, camera_parameter.base_line);

        std::vector<std::pair<double, double>> ranges;
        ranges.push_back(std::make_pair(0.15, 1));
        ranges.push_back(std::make_pair(1, 2));
        ranges.push_back(std::make_pair(2, 3));
        std::vector<double> a99 = MetricsProcess::calculateA99DepthRelativeError(
            stereo_image.left_image_file,
            gt_disparity, infer_disparity, ranges,
            camera_parameter.camera_fx, camera_parameter.base_line);

        double infinity_metric = MetricsProcess::calculateInfinityMetric(
            gt_disparity, infer_disparity,
            camera_parameter.camera_fx, camera_parameter.base_line);

        std::cout << "file: " << left_image_file
                  << ", process(" << index + 1 << "/" << stereo_image_sets.size() << ")"
                  << ", epe: " << epe << ", bad_pixel_2|4: " << bad_pixel.first << " | " << bad_pixel.second
                  << ", fnpr: " << fnpr << ", ffpr: " << ffpr
                  << ", infinity_metric: " << infinity_metric << std::endl;
        std::cout << "A99 range [0.15, 1): " << a99[0] << std::endl;
        std::cout << "A99 range [1, 2): "    << a99[1] << std::endl;
        std::cout << "A99 range [2, 3): "    << a99[2] << std::endl;

        metrics_json_writer.add_image_metrics(stereo_image.left_image_file,
                                              epe, bad_pixel.first, bad_pixel.second,
                                              fnpr, ffpr, a99, infinity_metric);

        if (!std::isnan(epe) && !std::isinf(epe)) epes.push_back(epe);
        if (!std::isnan(bad_pixel.first) && !std::isinf(bad_pixel.first)) bad_pixels_2.push_back(bad_pixel.first);
        if (!std::isnan(bad_pixel.second) && !std::isinf(bad_pixel.second)) bad_pixels_4.push_back(bad_pixel.second);
        if (!std::isnan(fnpr) && !std::isinf(fnpr)) fnprs.push_back(fnpr);
        if (!std::isnan(ffpr) && !std::isinf(ffpr)) ffprs.push_back(ffpr);
        if (!std::isnan(infinity_metric) && !std::isinf(infinity_metric)) infinity_metrics.push_back(infinity_metric);
        if (!std::isnan(a99[0]) && !std::isinf(a99[0])) {
          a99_0.push_back(a99[0]);
          if (a99[0] > a99s_max[0]) {
            a99s_max[0] = a99[0];
            a99s_max_index[0] = index;
          }
        }
        if (!std::isnan(a99[1]) && !std::isinf(a99[1])) {
          a99_1.push_back(a99[1]);
          if (a99[1] > a99s_max[1]) {
            a99s_max[1] = a99[1];
            a99s_max_index[1] = index;
          }
        }
        if (!std::isnan(a99[2]) && !std::isinf(a99[2])) {
          a99_2.push_back(a99[2]);
          if (a99[2] > a99s_max[2]) {
            a99s_max[2] = a99[2];
            a99s_max_index[2] = index;
          }
        }
      } else {
        std::cerr << "inference failed" << std::endl;
      }
    } else {
      std::cerr << "stereo_image is invalid" << std::endl;
    }
    double epe_sum = std::accumulate(epes.begin(), epes.end(), 0.);
    double bad_pixels2_sum = std::accumulate(bad_pixels_2.begin(), bad_pixels_2.end(), 0.);
    double bad_pixels4_sum = std::accumulate(bad_pixels_4.begin(), bad_pixels_4.end(), 0.);
    double fnpr_sum = std::accumulate(fnprs.begin(), fnprs.end(), 0.);
    double ffpr_sum = std::accumulate(ffprs.begin(), ffprs.end(), 0.);
    double infinity_metric_sum = std::accumulate(infinity_metrics.begin(), infinity_metrics.end(), 0.);
    double a99_0_sum = std::accumulate(a99_0.begin(), a99_0.end(), 0.);
    double a99_1_sum = std::accumulate(a99_1.begin(), a99_1.end(), 0.);
    double a99_2_sum = std::accumulate(a99_2.begin(), a99_2.end(), 0.);

    std::cout << "average epe: " << epe_sum / epes.size() << ", bad_pixel_2|4: " << bad_pixels2_sum / bad_pixels_2.size()
              << " | " << bad_pixels4_sum / bad_pixels_4.size()
              << ", fnpr: " << fnpr_sum / fnprs.size() << ", ffpr: " << ffpr_sum /  ffprs.size()
              << ", infinity_metric: " << infinity_metric_sum / infinity_metrics.size() << std::endl;
    std::cout << "A99 range [0.15, 1): " << a99_0_sum / a99_0.size() << std::endl;
    std::cout << "A99 range [1, 2): "    << a99_1_sum / a99_1.size() << std::endl;
    std::cout << "A99 range [2, 3): "    << a99_2_sum / a99_2.size() << std::endl;

    index++;
  }

  index = 0;
  for (const auto &i : a99s_max_index) {
    std::string left_image_file = data_path + "/" + stereo_image_sets[i].left_image_file;
    std::string right_image_file = data_path + "/" + stereo_image_sets[i].right_image_file;
    std::string disparity_image_file = data_path + "/" + stereo_image_sets[i].disparity_image_file;

    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    InferenceData infer_data(now, left_image_file, right_image_file);

    std::cout << "a99 max index: " << index << ", value: " << a99s_max[index] << std::endl;
    std::cout << "file: " << left_image_file << std::endl;

    CameraParameter camera_parameter;
    StereoResult stereo_result;
    cv::Mat infer_disparity;
    cv::Mat gt_disparity = cv::imread(disparity_image_file, cv::IMREAD_UNCHANGED), gt_disparity_32FC;
    gt_disparity.convertTo(gt_disparity_32FC, CV_32FC1);
    std::vector<float> infer_disparity_points;
    std::vector<float> gt_disparity_points(gt_disparity_32FC.begin<float>(), gt_disparity_32FC.end<float>());
    stereo_image_sets[i].get_camera_parameter(
        camera_parameter.camera_fx, camera_parameter.camera_fy,
        camera_parameter.camera_cx, camera_parameter.camera_cy,
        camera_parameter.base_line);

    if (stereo_demo.get_inference_result(infer_data, stereo_result,
                                         infer_disparity_points, camera_parameter) == 0) {
      dump_visual_image(infer_data, stereo_result, infer_disparity_points,
                        "infer_disp_" + stereo_image_sets[i].left_image_file);
      infer_disparity = cv::Mat(
          stereo_result.model_depth.rows, stereo_result.model_depth.cols,
          CV_32FC1, infer_disparity_points.data());
      stereo_result.model_depth = MetricsProcess::dispToDepth(infer_disparity,
                                                              camera_parameter.camera_fx, camera_parameter.base_line);
      dump_depth_in_mm(stereo_result, "infer_depth_" + stereo_image_sets[i].left_image_file);
      now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
      InferenceData gt_data(now, left_image_file, right_image_file);

      stereo_result.ts = now;
      stereo_result.model_depth = MetricsProcess::dispToDepth(gt_disparity,
          camera_parameter.camera_fx, camera_parameter.base_line);
      dump_visual_image(gt_data, stereo_result, gt_disparity_points,
                        "gt_disp_" + stereo_image_sets[i].left_image_file);
      dump_depth_in_mm(stereo_result, "gt_depth_" + stereo_image_sets[i].left_image_file);
    }
    index++;
  }

  metrics_json_writer.save_to_file("./X5_metrics.json");

  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "metric calculation finished! Consume: " << std::fixed << std::setprecision(4)
            << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
            << " second" << std::endl;
}

int main(int argc, char **argv) {
//  system("mkdir -p ./result/");
//  calculate_metrics("../testset/scene_flow_test/calib.json", "../testset/scene_flow_test/");
//  return 0;
//  return main_V2_4(argc, argv);

  return main_V2_4(argc, argv);
}
