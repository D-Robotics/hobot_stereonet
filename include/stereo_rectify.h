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

#ifndef HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_
#define HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_

#include "rclcpp/rclcpp.hpp"
#include "opencv2/opencv.hpp"

class StereoRectify {
public:
  StereoRectify(const std::string &stereo_calib_file_path, const rclcpp::Logger &logger);

  // 构建undistort map（只能执行一次）
  int build_undistmap(const int &input_width, const int &input_height, const int &output_width,
                      const int &output_height);

  void rectify(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &rectified_left_image,
               cv::Mat &rectified_right_image);

  void get_intrinsic(double &fx, double &fy, double &cx, double &cy, double &baseline) const;

private:
  std::string stereo_calib_file_path_;
  rclcpp::Logger logger_;

  std::vector<cv::Mat> Kls_, Krs_, Dls_, Drs_, R_rls_, t_rls_;
  std::vector<std::vector<int>> cam_resolutions_;
  std::vector<std::string> distortion_models_;
  std::vector<float> fov_scales_;
  std::vector<float> alphas_;

  std::vector<cv::Mat> Qs_;
  std::vector<cv::Mat> undistmap1ls_, undistmap2ls_, undistmap1rs_, undistmap2rs_;

  bool undistmap_built_ = false;
};

#endif // HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_