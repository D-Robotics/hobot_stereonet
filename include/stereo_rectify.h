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

#ifndef STEREO_HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_
#define STEREO_HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_

#include <opencv2/opencv.hpp>

namespace stereonet {

struct StereoRectify {
  template <typename T>
  StereoRectify(const T &fs, int model_input_w, int model_input_h,
                bool resize_before_rectify = false) {
    // Reading cam0 data
    std::vector<double> cam0_distortion_coeffs;
    std::vector<double> cam0_intrinsics;
    std::vector<int> cam0_resolution;
    float width_scale = 1.0f, height_scale = 1.0f;
    int stereo_input_width, stereo_input_height;

    fs["cam0"]["distortion_coeffs"] >> cam0_distortion_coeffs;
    fs["cam0"]["intrinsics"] >> cam0_intrinsics;
    fs["cam0"]["resolution"] >> cam0_resolution;
    fs["cam0"]["distortion_model"] >> distortion_model;

    // Reading cam1 data
    std::vector<std::vector<double>> cam1_T_cn_cnm1;
    std::vector<double> cam1_distortion_coeffs;
    std::vector<double> cam1_intrinsics;
    std::vector<int> cam1_resolution;

    fs["cam1"]["T_cn_cnm1"] >> cam1_T_cn_cnm1;
    fs["cam1"]["distortion_coeffs"] >> cam1_distortion_coeffs;
    fs["cam1"]["intrinsics"] >> cam1_intrinsics;
    fs["cam1"]["resolution"] >> cam1_resolution;

    if (resize_before_rectify) {
      stereo_input_width = model_input_w;
      stereo_input_height = model_input_h;
      width_scale = model_input_w * 1.0 / cam0_resolution[0];
      height_scale = model_input_h * 1.0 / cam0_resolution[1];
    } else {
      stereo_input_width = cam0_resolution[0];
      stereo_input_height = cam0_resolution[1];
    }

    Dl = cv::Mat(1, cam0_distortion_coeffs.size(), CV_64F, cam0_distortion_coeffs.data()).clone();
    Kl = cv::Mat::zeros(3, 3, CV_64F);
    Kl.at<double>(0, 0) = cam0_intrinsics[0] * width_scale;
    Kl.at<double>(0, 2) = cam0_intrinsics[2] * width_scale;
    Kl.at<double>(1, 1) = cam0_intrinsics[1] * height_scale;
    Kl.at<double>(1, 2) = cam0_intrinsics[3] * height_scale;
    Kl.at<double>(2, 2) = 1;

    R_rl = cv::Mat::zeros(3, 3, CV_64F);
    t_rl = cv::Mat::zeros(3, 1, CV_64F);

    R_rl.at<double>(0, 0) = cam1_T_cn_cnm1[0][0];
    R_rl.at<double>(0, 1) = cam1_T_cn_cnm1[0][1];
    R_rl.at<double>(0, 2) = cam1_T_cn_cnm1[0][2];
    R_rl.at<double>(1, 0) = cam1_T_cn_cnm1[1][0];
    R_rl.at<double>(1, 1) = cam1_T_cn_cnm1[1][1];
    R_rl.at<double>(1, 2) = cam1_T_cn_cnm1[1][2];
    R_rl.at<double>(2, 0) = cam1_T_cn_cnm1[2][0];
    R_rl.at<double>(2, 1) = cam1_T_cn_cnm1[2][1];
    R_rl.at<double>(2, 2) = cam1_T_cn_cnm1[2][2];

    t_rl.at<double>(0, 0) = cam1_T_cn_cnm1[0][3];
    t_rl.at<double>(1, 0) = cam1_T_cn_cnm1[1][3];
    t_rl.at<double>(2, 0) = cam1_T_cn_cnm1[2][3];

    Dr = cv::Mat(1, cam1_distortion_coeffs.size(), CV_64F, cam1_distortion_coeffs.data()).clone();

    Kr = cv::Mat::zeros(3, 3, CV_64F);
    Kr.at<double>(0, 0) = cam1_intrinsics[0] * width_scale;
    Kr.at<double>(0, 2) = cam1_intrinsics[2] * width_scale;
    Kr.at<double>(1, 1) = cam1_intrinsics[1] * height_scale;
    Kr.at<double>(1, 2) = cam1_intrinsics[3] * height_scale;
    Kr.at<double>(2, 2) = 1;

    float fov_scale = 0.8;
    if (distortion_model == "radtan" ||
        distortion_model == "rational_polynomial") {
      cv::stereoRectify(Kl, Dl, Kr, Dr,
                        cv::Size(stereo_input_width, stereo_input_height), R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                        cv::CALIB_ZERO_DISPARITY, 0, cv::Size(model_input_w, model_input_h));

      cv::initUndistortRectifyMap(Kl, Dl, Rl, Pl,
                                  cv::Size(model_input_w, model_input_h), CV_32FC1, undistmap1l, undistmap2l);
      cv::initUndistortRectifyMap(Kr, Dr, Rr, Pr,
                                  cv::Size(model_input_w, model_input_h), CV_32FC1, undistmap1r, undistmap2r);
    } else if (distortion_model == "equidistant") {
      if (!fs["cam1"]["fov_scale"].empty()) {
        fs["cam1"]["fov_scale"] >> fov_scale;
      }
      if (fov_scale == 0) fov_scale = 0.8;
      cv::fisheye::stereoRectify(Kl,
                                 Dl,
                                 Kr,
                                 Dr,
                                 cv::Size(stereo_input_width, stereo_input_height),
                                 R_rl,
                                 t_rl,
                                 Rl,
                                 Rr,
                                 Pl,
                                 Pr,
                                 Q,
                                 cv::fisheye::CALIB_ZERO_DISPARITY,
                                 cv::Size(model_input_w, model_input_h),
                                 0.0,
                                 fov_scale);
      cv::fisheye::initUndistortRectifyMap(Kl, Dl, Rl, Pl,
                                           cv::Size(model_input_w, model_input_h), CV_32FC1, undistmap1l, undistmap2l);
      cv::fisheye::initUndistortRectifyMap(Kr, Dr, Rr, Pr,
                                           cv::Size(model_input_w, model_input_h), CV_32FC1, undistmap1r, undistmap2r);
    }

    camera_fx = Q.at<double>(2, 3);
    camera_fy = Q.at<double>(2, 3);
    camera_cx = -Q.at<double>(0, 3);
    camera_cy = -Q.at<double>(1, 3);
    //  const cv::Mat t = Rr * t_rl;
    base_line = std::abs(1 / Q.at<double>(3, 2));

    std::cout << "distortion_model: " << distortion_model << std::endl
              << "Kl:" << std::endl
              << Kl << std::endl
              << "Dl:" << std::endl
              << Dl << std::endl
              << "Kr: " << std::endl
              << Kr << std::endl
              << "Dr:" << std::endl
              << Dr << std::endl
              << "R, t: " << std::endl
              << R_rl << std::endl
              << t_rl << std::endl
              << "calib file width, height: " << cam0_resolution[0] << ", " << cam0_resolution[1] << std::endl
              << "model input width, height: " << model_input_w << ", " << model_input_h << std::endl
              << "fov_scale: " << fov_scale << std::endl
              << "resize_before_rectify: " << resize_before_rectify << std::endl
              << "width_scale, height_scale: " << width_scale << ", " << height_scale
              << std::endl;
  }

  void Rectify(const cv::Mat &left_image,
               const cv::Mat &right_image,
               cv::Mat &rectified_left_image,
               cv::Mat &rectified_right_image) {
    cv::remap(left_image, rectified_left_image, undistmap1l, undistmap2l, cv::INTER_AREA);
    cv::remap(right_image, rectified_right_image, undistmap1r, undistmap2r, cv::INTER_AREA);
  }

  void GetIntrinsic(float &cx, float &cy, float &fx, float &fy, float &bl) {
    cx = camera_cx;
    cy = camera_cy;
    fx = camera_fx;
    fy = camera_fy;
    bl = base_line;
  }

  std::string distortion_model;
  cv::Mat Rl, Rr, Pl, Pr, Q;
  cv::Mat Kl, Kr, Dl, Dr, R_rl, t_rl;
  cv::Mat undistmap1l, undistmap2l, undistmap1r, undistmap2r;
  float camera_cx, camera_cy, camera_fx, camera_fy, base_line;
};

}

#endif //STEREO_HOBOT_STEREONET_INCLUDE_STEREO_RECTIFY_H_
