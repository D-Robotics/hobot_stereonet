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

#include "stereo_rectify.h"

StereoRectify::StereoRectify(const std::string &stereo_calib_file_path, const rclcpp::Logger &logger)
    : stereo_calib_file_path_(stereo_calib_file_path), logger_(logger) {
  int i = 0;
  RCLCPP_WARN_STREAM(logger_, "=> -------- init StereoRectify------------------");
  cv::FileStorage fs(stereo_calib_file_path_, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to open " << stereo_calib_file_path_);
    return;
  }

  auto extract_parameters = [&]<typename T>(const T &stereo_node) {
    // cam0
    std::vector<double> cam0_intrinsics;
    std::vector<double> cam0_distortion_coeffs;
    std::vector<int> cam0_resolution;
    std::string cam0_distortion_model;
    stereo_node["cam0"]["intrinsics"] >> cam0_intrinsics;
    stereo_node["cam0"]["distortion_coeffs"] >> cam0_distortion_coeffs;
    stereo_node["cam0"]["resolution"] >> cam0_resolution;
    stereo_node["cam0"]["distortion_model"] >> cam0_distortion_model;

    // cam1
    std::vector<double> cam1_intrinsics;
    std::vector<double> cam1_distortion_coeffs;
    std::vector<int> cam1_resolution;
    std::string cam1_distortion_model;
    stereo_node["cam1"]["intrinsics"] >> cam1_intrinsics;
    stereo_node["cam1"]["distortion_coeffs"] >> cam1_distortion_coeffs;
    stereo_node["cam1"]["resolution"] >> cam1_resolution;
    stereo_node["cam1"]["distortion_model"] >> cam1_distortion_model;

    // extrinsics
    std::vector<std::vector<double>> cam1_T_cn_cnm1;
    stereo_node["cam1"]["T_cn_cnm1"] >> cam1_T_cn_cnm1;

    float fov_scale = 0.8f;
    if (!stereo_node["cam1"]["fov_scale"].empty()) {
      stereo_node["cam1"]["fov_scale"] >> fov_scale;
    }

    float alpha = 0.0f;
    if (!stereo_node["cam1"]["alpha"].empty()) {
      stereo_node["cam1"]["alpha"] >> alpha;
    }

    // cv::Mat
    cv::Mat Kl = cv::Mat::zeros(3, 3, CV_64F);
    Kl.at<double>(0, 0) = cam0_intrinsics[0];
    Kl.at<double>(0, 2) = cam0_intrinsics[2];
    Kl.at<double>(1, 1) = cam0_intrinsics[1];
    Kl.at<double>(1, 2) = cam0_intrinsics[3];
    Kl.at<double>(2, 2) = 1;
    cv::Mat Dl = cv::Mat(1, cam0_distortion_coeffs.size(), CV_64F, cam0_distortion_coeffs.data()).clone();

    cv::Mat Kr = cv::Mat::zeros(3, 3, CV_64F);
    Kr.at<double>(0, 0) = cam1_intrinsics[0];
    Kr.at<double>(0, 2) = cam1_intrinsics[2];
    Kr.at<double>(1, 1) = cam1_intrinsics[1];
    Kr.at<double>(1, 2) = cam1_intrinsics[3];
    Kr.at<double>(2, 2) = 1;
    cv::Mat Dr = cv::Mat(1, cam1_distortion_coeffs.size(), CV_64F, cam1_distortion_coeffs.data()).clone();

    cv::Mat R_rl = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat t_rl = cv::Mat::zeros(3, 1, CV_64F);
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        R_rl.at<double>(r, c) = cam1_T_cn_cnm1[r][c];
      }
      t_rl.at<double>(r, 0) = cam1_T_cn_cnm1[r][3];
    }

    // save
    Kls_.push_back(Kl);
    Krs_.push_back(Kr);
    Dls_.push_back(Dl);
    Drs_.push_back(Dr);
    R_rls_.push_back(R_rl);
    t_rls_.push_back(t_rl);
    cam_resolutions_.push_back(cam0_resolution);
    distortion_models_.push_back(cam0_distortion_model);
    fov_scales_.push_back(fov_scale);
    alphas_.push_back(alpha);

    // print
    RCLCPP_WARN_STREAM(logger_, "=> load stereo calib from: " << stereo_calib_file_path_);
    RCLCPP_WARN_STREAM(logger_, "=> Kl: " << std::endl << Kl);
    RCLCPP_WARN_STREAM(logger_, "=> Dl: " << std::endl << Dl);
    RCLCPP_WARN_STREAM(logger_, "=> Kr: " << std::endl << Kr);
    RCLCPP_WARN_STREAM(logger_, "=> Dr: " << std::endl << Dr);
    RCLCPP_WARN_STREAM(logger_, "=> R_rl: " << std::endl << R_rl);
    RCLCPP_WARN_STREAM(logger_, "=> t_rl: " << std::endl << t_rl);
    RCLCPP_WARN_STREAM(logger_,
                       "=> cam0_resolution: " << "[" << cam0_resolution[0] << ", " << cam0_resolution[1] << "]");
    RCLCPP_WARN_STREAM(logger_, "=> cam0_distortion_model: " << cam0_distortion_model);
    if (cam0_distortion_model == "equidistant") RCLCPP_WARN_STREAM(logger_, "=> fov_scale: " << fov_scale);
    if (cam0_distortion_model == "radtan" || cam0_distortion_model == "rational_polynomial")
      RCLCPP_WARN_STREAM(logger_, "=> alpha: " << alpha);
    RCLCPP_WARN_STREAM(logger_, "=> ---------------------------------------------");
  };

  while (true) {
    std::string stereo_no = "stereo" + std::to_string(i);
    if (!fs[stereo_no].empty()) {
      cv::FileNode stereo_node = fs[stereo_no];
      extract_parameters(stereo_node);
      i++;
    } else {
      if (i == 0) {
        extract_parameters(fs);
      }
      break;
    }
  }
  fs.release();
}

int StereoRectify::build_undistmap(const int &input_width, const int &input_height, const int &output_width,
                                   const int &output_height) {
  if (undistmap_built_) return 0;
  RCLCPP_WARN_STREAM(logger_, "=> -------- build_undistmap --------------------");
  for (int i = 0; i < Kls_.size(); i++) {
    cv::Mat Kl = Kls_[i].clone();
    cv::Mat Kr = Krs_[i].clone();
    cv::Mat Dl = Dls_[i].clone();
    cv::Mat Dr = Drs_[i].clone();
    cv::Mat R_rl = R_rls_[i].clone();
    cv::Mat t_rl = t_rls_[i].clone();
    std::vector<int> cam_resolution = cam_resolutions_[i];
    std::string distortion_model = distortion_models_[i];
    float fov_scale = fov_scales_[i];
    float alpha = alphas_[i];

    int tmp_input_width = -1;
    int tmp_input_height = -1;
    int tmp_output_width = -1;
    int tmp_output_height = -1;
    if (i == 0) {
      tmp_input_width = input_width;
      tmp_input_height = input_height;
    } else if (i >= 1) {
      tmp_input_width = cam_resolution[0];
      tmp_input_height = cam_resolution[1];
    }

    if (i == Kls_.size() - 1) {
      tmp_output_width = output_width;
      tmp_output_height = output_height;
    } else if (i >= 0 && i < Kls_.size() - 1) {
      std::vector<int> cam_resolution_next = cam_resolutions_[i + 1];
      tmp_output_width = cam_resolution_next[0];
      tmp_output_height = cam_resolution_next[1];
    }

    double width_scale = static_cast<double>(tmp_input_width) / static_cast<double>(cam_resolution[0]);
    double height_scale = static_cast<double>(tmp_input_height) / static_cast<double>(cam_resolution[1]);
    Kl.at<double>(0, 0) *= width_scale;
    Kl.at<double>(0, 2) *= width_scale;
    Kl.at<double>(1, 1) *= height_scale;
    Kl.at<double>(1, 2) *= height_scale;
    Kr.at<double>(0, 0) *= width_scale;
    Kr.at<double>(0, 2) *= width_scale;
    Kr.at<double>(1, 1) *= height_scale;
    Kr.at<double>(1, 2) *= height_scale;

    cv::Mat Rl, Rr, Pl, Pr, Q;
    cv::Mat undistmap1l, undistmap2l, undistmap1r, undistmap2r;
    if (distortion_model == "radtan" || distortion_model == "rational_polynomial") {
      if (alpha > 1.0f) alpha = 1.0f;
      cv::stereoRectify(Kl, Dl, Kr, Dr, cv::Size(tmp_input_width, tmp_input_height), R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                        cv::CALIB_ZERO_DISPARITY, alpha, cv::Size(tmp_output_width, tmp_output_height));
      cv::initUndistortRectifyMap(Kl, Dl, Rl, Pl, cv::Size(tmp_output_width, tmp_output_height), CV_32FC1, undistmap1l,
                                  undistmap2l);
      cv::initUndistortRectifyMap(Kr, Dr, Rr, Pr, cv::Size(tmp_output_width, tmp_output_height), CV_32FC1, undistmap1r,
                                  undistmap2r);
    } else if (distortion_model == "equidistant") {
      if (fov_scale <= 0) fov_scale = 0.8f;
      cv::fisheye::stereoRectify(Kl, Dl, Kr, Dr, cv::Size(tmp_input_width, tmp_input_height), R_rl, t_rl, Rl, Rr, Pl,
                                 Pr, Q, cv::fisheye::CALIB_ZERO_DISPARITY,
                                 cv::Size(tmp_output_width, tmp_output_height), 0.0, fov_scale);
      cv::fisheye::initUndistortRectifyMap(Kl, Dl, Rl, Pl, cv::Size(tmp_output_width, tmp_output_height), CV_32FC1,
                                           undistmap1l, undistmap2l);
      cv::fisheye::initUndistortRectifyMap(Kr, Dr, Rr, Pr, cv::Size(tmp_output_width, tmp_output_height), CV_32FC1,
                                           undistmap1r, undistmap2r);
    } else {
      RCLCPP_ERROR_STREAM(logger_, "=> unsupported distortion_model: " << distortion_model);
      return -1;
    }

    undistmap1ls_.push_back(undistmap1l);
    undistmap2ls_.push_back(undistmap2l);
    undistmap1rs_.push_back(undistmap1r);
    undistmap2rs_.push_back(undistmap2r);
    Qs_.push_back(Q);

    RCLCPP_WARN_STREAM(logger_, "=> input_resolution: " << "[" << tmp_input_width << ", " << tmp_input_height << "]");
    RCLCPP_WARN_STREAM(logger_, "=> cam_resolution: " << "[" << cam_resolution[0] << ", " << cam_resolution[1] << "]");
    RCLCPP_WARN_STREAM(logger_, "=> width, height scale: " << "[" << width_scale << ", " << height_scale << "]");
    RCLCPP_WARN_STREAM(logger_,
                       "=> output_resolution: " << "[" << tmp_output_width << ", " << tmp_output_height << "]");
    RCLCPP_WARN_STREAM(logger_, "=> Kl: " << std::endl << Kl);
    RCLCPP_WARN_STREAM(logger_, "=> Dl: " << std::endl << Dl);
    RCLCPP_WARN_STREAM(logger_, "=> Kr: " << std::endl << Kr);
    RCLCPP_WARN_STREAM(logger_, "=> Dr: " << std::endl << Dr);
    RCLCPP_WARN_STREAM(logger_, "=> R_rl: " << std::endl << R_rl);
    RCLCPP_WARN_STREAM(logger_, "=> t_rl: " << std::endl << t_rl);
    RCLCPP_WARN_STREAM(logger_, "=> distortion_model: " << distortion_model);
    if (distortion_model == "equidistant") RCLCPP_WARN_STREAM(logger_, "=> fov_scale: " << fov_scale);
    if (distortion_model == "radtan" || distortion_model == "rational_polynomial")
      RCLCPP_WARN_STREAM(logger_, "=> alpha: " << alpha);
    double fx = Q.at<double>(2, 3);
    double fy = Q.at<double>(2, 3);
    double cx = -Q.at<double>(0, 3);
    double cy = -Q.at<double>(1, 3);
    double baseline = std::abs(1 / Q.at<double>(3, 2));
    RCLCPP_WARN_STREAM(logger_, "=> rectify fx: " << fx << ", fy: " << fy << ", cx: " << cx << ", cy: " << cy
                                                  << ", baseline: " << baseline);
    RCLCPP_WARN_STREAM(logger_, "=> ---------------------------------------------");
    if (i == Kls_.size() - 1) {
      float HFOV = 2 * atan(tmp_output_width / (2 * fx)) * 180 / M_PI;
      float VFOV = 2 * atan(tmp_output_height / (2 * fy)) * 180 / M_PI;
      RCLCPP_WARN_STREAM(logger_, "=> HFOV: " << HFOV << "°, VFOV: " << VFOV << "°");
      RCLCPP_WARN_STREAM(logger_, "=> ---------------------------------------------");
    }
  }

  undistmap_built_ = true;
  return 0;
}

void StereoRectify::rectify(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &rectified_left_image,
                            cv::Mat &rectified_right_image) {
  if (undistmap1ls_.size() == 1) {
    cv::remap(left_image, rectified_left_image, undistmap1ls_[0], undistmap2ls_[0], cv::INTER_LINEAR);
    cv::remap(right_image, rectified_right_image, undistmap1rs_[0], undistmap2rs_[0], cv::INTER_LINEAR);
  } else {
    cv::Mat tmp_left_image, tmp_right_image;
    for (int i = 0; i < undistmap1ls_.size(); i++) {
      if (i == 0) {
        cv::remap(left_image, tmp_left_image, undistmap1ls_[i], undistmap2ls_[i], cv::INTER_LINEAR);
        cv::remap(right_image, tmp_right_image, undistmap1rs_[i], undistmap2rs_[i], cv::INTER_LINEAR);
      } else if (i == undistmap1ls_.size() - 1) {
        cv::remap(tmp_left_image, rectified_left_image, undistmap1ls_[i], undistmap2ls_[i], cv::INTER_LINEAR);
        cv::remap(tmp_right_image, rectified_right_image, undistmap1rs_[i], undistmap2rs_[i], cv::INTER_LINEAR);
      } else {
        cv::remap(tmp_left_image, tmp_left_image, undistmap1ls_[i], undistmap2ls_[i], cv::INTER_LINEAR);
        cv::remap(tmp_right_image, tmp_right_image, undistmap1rs_[i], undistmap2rs_[i], cv::INTER_LINEAR);
      }
    }
  }
}

void StereoRectify::get_intrinsic(double &fx, double &fy, double &cx, double &cy, double &baseline) const {
  cv::Mat Q = Qs_[Qs_.size() - 1];
  fx = Q.at<double>(2, 3);
  fy = Q.at<double>(2, 3);
  cx = -Q.at<double>(0, 3);
  cy = -Q.at<double>(1, 3);
  baseline = std::abs(1 / Q.at<double>(3, 2));
}
