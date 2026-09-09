// Copyright (c) 2025,D-Robotics.
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
#include <fstream>
#include <opencv2/ccalib/omnidir.hpp>
#include <cmath>
#include <limits>
#include <cassert>

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
   * @brief Rectify the left and right images
   * @param left_nv12 The left image in NV12 format
   * @param right_nv12 The right image in NV12 format
   * @param output_w Output image width
   * @param output_h Output image height
   * @param rectified_left_nv12 The rectified left image in NV12 format
   * @param rectified_right_nv12 The rectified right image in NV12 format
   */
  void rectify_nv12(const uint8_t *left_nv12, const uint8_t *right_nv12, int input_w, int input_h,
                    uint8_t *rect_left_nv12, uint8_t *rect_right_nv12, int output_w, int output_h);

  /**
   * @brief Get the intrinsic parameters
   * @param fx Focal length in x direction
   * @param fy Focal length in y direction
   * @param cx Center x coordinate
   * @param cy Center y coordinate
   * @param baseline Baseline distance between the two cameras
   */
  void get_intrinsic(double &fx, double &fy, double &cx, double &cy, double &baseline) const;

  /**
   * @brief Get the rectification model name, e.g. "RECTIFY_PERSPECTIVE" or "RECTIFY_LONGLATI".
   */
  std::string get_rectify_model() const { return rectify_model_; }

private:
  // logger
  rclcpp::Logger logger_;

  // stereo calibration file path
  std::string stereo_calib_file_path_;

  // camera intrinsic
  cv::Mat Kl_, Kr_, Dl_, Dr_, R_rl_, t_rl_;
  std::vector<int> cam_resolution_;
  std::string distortion_model_;
  // Mei rectification model: "RECTIFY_PERSPECTIVE" (default) or "RECTIFY_LONGLATI".
  std::string rectify_model_ = "RECTIFY_PERSPECTIVE";
  float fov_scale_ = 0.8f;
  float alpha_ = 0.0f;

  // Mei perspective / fisheye (equidistant): target horizontal FOV in degrees
  // (<=0 means auto max-no-black). mei: focal derived directly from it;
  // equidistant: the fov_scale yielding this HFOV is searched.
  double target_hfov_ = 0.0;

  // stereo rectification
  cv::Mat Q_;
  cv::Mat undistmap1l_, undistmap2l_, undistmap1r_, undistmap2r_;

  // build undistort map flag
  bool undistmap_built_ = false;

  // mei model
  double xi_l_ = 0.0;
  double xi_r_ = 0.0;

  double rectify_fx_ = 0.0;
  double rectify_fy_ = 0.0;
  double rectify_cx_ = 0.0;
  double rectify_cy_ = 0.0;
  double rectify_baseline_ = 0.0;

  double mei_focal_scale_ = 1.0;
};

#endif // HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_