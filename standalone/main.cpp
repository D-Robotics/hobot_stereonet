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

#include "stereonet_process.h"
#include "image_conversion.h"

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
};

struct StereoDemo {
  int init(const std::string &stereonet_model_file_path,
           const std::string &post_version,
           int max_disp);
  int deinit();

  int get_image(InferenceData &infer_data);
  int get_inference_result(InferenceData &infer_data,
          StereoResult &stereo_result,
          std::vector<float> &points,
          CameraParameter &camera_parameter);

private:
  std::shared_ptr<StereonetProcess> stereonet_process_;

private:
  int model_input_h_, model_input_w_;
  int model_output_h_, model_output_w_;

};

int StereoDemo::init(const std::string &stereonet_model_file_path,
                     const std::string &post_version,
                      int max_disp) {
  int ret = 0;
  stereonet_process_ = std::make_shared<StereonetProcess>();
  ret = stereonet_process_->stereonet_init(
          stereonet_model_file_path, max_disp, post_version, 0.7);
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
  float32x4_t zero_vec = vdupq_n_f32(0.f);
  //uint16x4_t zero_vec_u16 = vget_low_u16(vdupq_n_u16(0));
  float32x4_t factor_vector = vdupq_n_f32(factor);
  for (uint32_t i = 0; i < points.size(); i += 4) {
    float32x4_t points_vec = vld1q_f32(&points[i]);
    uint32x4_t mask = vcgtq_f32(points_vec, zero_vec);
    float32x4_t depth_vec = vdivq_f32(factor_vector, points_vec);
    uint16x4_t depth_int16_vec = vmovn_u32(vcvtq_u32_f32(vbslq_f32(mask, depth_vec, zero_vec)));
    vst1_u16(&depth_data[i], depth_int16_vec);
  }

  stereo_result.model_depth = model_depth_img;
  stereo_result.ts = infer_data.ts;
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
    std::cout << "\nrecv SIGINT, exit!" << std::endl;
  }
}

int dump_depth_in_mm(StereoResult &stereo_result) {
  auto ts = stereo_result.ts;
  cv::imwrite("./result/"+ std::to_string(ts) +"_depth.png", stereo_result.model_depth);
  return 0;
}

int dump_disparity(StereoResult &stereo_result, cv::Mat& disparity) {
  auto ts = stereo_result.ts;
  cv::imwrite("./result/"+ std::to_string(ts) +"_disparity.pfm", disparity);
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
                     std::vector<float>&points) {
  auto ts = stereo_result.ts;
  const cv::Mat &depth_img = stereo_result.model_depth;
  cv::Mat bgr_image = infer_data.left_image;
  cv::Mat visual_img(bgr_image.rows * 2, bgr_image.cols, CV_8UC3);
  bgr_image.copyTo(visual_img(cv::Rect(0, 0, bgr_image.cols, bgr_image.rows)));

  cv::Mat feat_mat(bgr_image.rows, bgr_image.cols, CV_32F, const_cast<float *>(points.data()));
  dump_disparity(stereo_result, feat_mat);
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
      uint16_t Z = depth_img.at<uint16_t>(i * y_step, j * x_step);
      // mm -> m
      double distance = static_cast<double>(Z) / 1000.0;

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
  cv::imwrite("./result/" + std::to_string(ts) +"_visual.jpg", visual_img);
  return 0;
}


int main(int argc, char **argv) {
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
  system("mkdir -p ./result/");

  ret = stereo_demo.init(stereonet_model_file_path, "v2.4", 192);
  if (ret != 0) {
    std::cerr << "model init failed!" << std::endl;
    return -1;
  }
  std::cout << "model init succeed!" << std::endl;

  ret = stereo_demo.get_inference_result(infer_data, stereo_result, disparity_points, camera_parameter);
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
