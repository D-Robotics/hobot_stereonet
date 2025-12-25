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

  /**
   * @brief Build the undistort map (only execute once)
   * @param input_width Input image width
   * @param input_height Input image height
   * @param output_width Output image width
   * @param output_height Output image height
   * @return 0 on success, -1 on failure
   */
  int build_undistmap(const int &input_width, const int &input_height, const int &output_width,
                      const int &output_height);

  /**
   * @brief Rectify the left and right images
   * @param left_image The left image
   * @param right_image The right image
   * @param rectified_left_image The rectified left image
   * @param rectified_right_image The rectified right image
   */
  void rectify(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &rectified_left_image,
               cv::Mat &rectified_right_image);

  /**
   * @brief Get the intrinsic parameters
   * @param fx Focal length in x direction
   * @param fy Focal length in y direction
   * @param cx Center x coordinate
   * @param cy Center y coordinate
   * @param baseline Baseline distance between the two cameras
   */
  void get_intrinsic(double &fx, double &fy, double &cx, double &cy, double &baseline) const;

private:
  // logger
  rclcpp::Logger logger_;

  // stereo calibration file path
  std::string stereo_calib_file_path_;

  // camera intrinsic
  std::vector<cv::Mat> Kls_, Krs_, Dls_, Drs_, R_rls_, t_rls_;
  std::vector<std::vector<int>> cam_resolutions_;
  std::vector<std::string> distortion_models_;
  std::vector<float> fov_scales_;
  std::vector<float> alphas_;

  // stereo rectification
  std::vector<cv::Mat> Qs_;
  std::vector<cv::Mat> undistmap1ls_, undistmap2ls_, undistmap1rs_, undistmap2rs_;

  // build undistort map flag
  bool undistmap_built_ = false;
};

#endif // HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_