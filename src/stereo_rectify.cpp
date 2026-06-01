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
#include <cassert>

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
    bool fov_scale_provided = false;
    if (!stereo_node["cam1"]["fov_scale"].empty()) {
      stereo_node["cam1"]["fov_scale"] >> fov_scale;
      fov_scale_provided = true;
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
    Kl_ = Kl;
    Kr_ = Kr;
    Dl_ = Dl;
    Dr_ = Dr;
    R_rl_ = R_rl;
    t_rl_ = t_rl;
    cam_resolution_ = cam0_resolution;
    distortion_model_ = cam0_distortion_model;
    fov_scale_ = fov_scale;
    fov_scale_provided_ = fov_scale_provided;
    alpha_ = alpha;

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

  if (!fs["cameraMatrix1"].empty()) {
    // OpenCV format
    fs["cameraMatrix1"] >> Kl_;
    fs["distCoeffs1"] >> Dl_;
    fs["cameraMatrix2"] >> Kr_;
    fs["distCoeffs2"] >> Dr_;
    fs["R"] >> R_rl_;

    fs["T"] >> t_rl_;
    int width, height;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    cam_resolution_ = {width, height};
    distortion_model_ = Dl_.total() == 8 ? "rational_polynomial" : "radtan";
    if (!fs["distortion_model"].empty()) fs["distortion_model"] >> distortion_model_;
    alpha_ = 0.0f;
    if (!fs["alpha"].empty()) fs["alpha"] >> alpha_;
    // print
    RCLCPP_WARN_STREAM(logger_, "=> load stereo calib from: " << stereo_calib_file_path_);
    RCLCPP_WARN_STREAM(logger_, "=> Kl: " << std::endl << Kl_);
    RCLCPP_WARN_STREAM(logger_, "=> Dl: " << std::endl << Dl_);
    RCLCPP_WARN_STREAM(logger_, "=> Kr: " << std::endl << Kr_);
    RCLCPP_WARN_STREAM(logger_, "=> Dr: " << std::endl << Dr_);
    RCLCPP_WARN_STREAM(logger_, "=> R_rl: " << std::endl << R_rl_);
    RCLCPP_WARN_STREAM(logger_, "=> t_rl: " << std::endl << t_rl_);
    RCLCPP_WARN_STREAM(logger_,
                       "=> cam_resolution: " << "[" << cam_resolution_[0] << ", " << cam_resolution_[1] << "]");
    RCLCPP_WARN_STREAM(logger_, "=> cam_distortion_model: " << distortion_model_);
    if (distortion_model_ == "equidistant") RCLCPP_WARN_STREAM(logger_, "=> fov_scale: " << fov_scale_);
    if (distortion_model_ == "radtan" || distortion_model_ == "rational_polynomial")
      RCLCPP_WARN_STREAM(logger_, "=> alpha: " << alpha_);
    RCLCPP_WARN_STREAM(logger_, "=> ---------------------------------------------");
  } else if (!fs["stereo0"].empty()) {
    // stereo0 format
    /*
    stereo0:
      cam0:
        intrinsics: [...]
        distortion_coeffs: [...]
        resolution: [...]
        distortion_model: radtan|rational_polynomial|equidistant
      cam1:
        intrinsics: [...]
        distortion_coeffs: [...]
        T_cn_cnm1: [...]
    */
    extract_parameters(fs["stereo0"]);
  } else {
    // cam0/cam1 format
    /*
    cam0:
      intrinsics: [...]
      distortion_coeffs: [...]
      resolution: [...]
      distortion_model: radtan|rational_polynomial|equidistant
    cam1:
      intrinsics: [...]
      distortion_coeffs: [...]
      T_cn_cnm1: [...]
    */
    extract_parameters(fs);
  }

  fs.release();
}

static bool has_black_border(const cv::Mat &map1, const cv::Mat &map2, int input_w, int input_h) {
  for (int y = 0; y < map1.rows; ++y) {
    const float *mx = map1.ptr<float>(y);
    const float *my = map2.ptr<float>(y);

    for (int x = 0; x < map1.cols; ++x) {
      if (mx[x] < 0 || mx[x] >= input_w - 1 || my[x] < 0 || my[x] >= input_h - 1) {
        return true;
      }
    }
  }
  return false;
}

static float find_max_no_black_fovscale(const cv::Mat &Kl, const cv::Mat &Dl, const cv::Mat &Kr, const cv::Mat &Dr,
                                        const cv::Mat &R_rl, const cv::Mat &t_rl, int input_w, int input_h,
                                        int output_w, int output_h) {
  float best_scale = 0.1f;

  for (float scale = 0.1f; scale <= 2.0f; scale += 0.01f) {
    cv::Mat Rl, Rr, Pl, Pr, Q;
    cv::Mat map1_l, map2_l, map1_r, map2_r;

    cv::fisheye::stereoRectify(Kl, Dl, Kr, Dr, cv::Size(input_w, input_h), R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                               cv::fisheye::CALIB_ZERO_DISPARITY, cv::Size(output_w, output_h), 0.0, scale);

    cv::fisheye::initUndistortRectifyMap(Kl, Dl, Rl, Pl, cv::Size(output_w, output_h), CV_32FC1, map1_l, map2_l);

    cv::fisheye::initUndistortRectifyMap(Kr, Dr, Rr, Pr, cv::Size(output_w, output_h), CV_32FC1, map1_r, map2_r);

    bool left_black = has_black_border(map1_l, map2_l, input_w, input_h);
    bool right_black = has_black_border(map1_r, map2_r, input_w, input_h);

    if (!left_black && !right_black) {
      best_scale = scale;
    } else {
      break;
    }
  }

  return best_scale;
}

int StereoRectify::build_undistmap(const int &input_width, const int &input_height, const int &output_width,
                                   const int &output_height) {
  if (undistmap_built_) return 0;
  RCLCPP_WARN_STREAM(logger_, "=> -------- build_undistmap --------------------");
  cv::Mat Kl = Kl_.clone();
  cv::Mat Kr = Kr_.clone();
  cv::Mat Dl = Dl_.clone();
  cv::Mat Dr = Dr_.clone();
  cv::Mat R_rl = R_rl_.clone();
  cv::Mat t_rl = t_rl_.clone();

  double width_scale = static_cast<double>(input_width) / cam_resolution_[0];
  double height_scale = static_cast<double>(input_height) / cam_resolution_[1];
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
  if (distortion_model_ == "radtan" || distortion_model_ == "rational_polynomial") {
    if (alpha_ > 1.0f) alpha_ = 1.0f;
    cv::stereoRectify(Kl, Dl, Kr, Dr, cv::Size(input_width, input_height), R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                      cv::CALIB_ZERO_DISPARITY, alpha_, cv::Size(output_width, output_height));
    cv::initUndistortRectifyMap(Kl, Dl, Rl, Pl, cv::Size(output_width, output_height), CV_32FC1, undistmap1l,
                                undistmap2l);
    cv::initUndistortRectifyMap(Kr, Dr, Rr, Pr, cv::Size(output_width, output_height), CV_32FC1, undistmap1r,
                                undistmap2r);
  } else if (distortion_model_ == "equidistant") {
    if (fov_scale_provided_) {
      if (fov_scale_ <= 0) fov_scale_ = 0.8f;
    } else {
      fov_scale_ = find_max_no_black_fovscale(Kl, Dl, Kr, Dr, R_rl, t_rl, input_width, input_height, output_width,
                                              output_height);
      RCLCPP_WARN_STREAM(logger_, "=> auto selected max no-black fov_scale: " << fov_scale_);
    }
    cv::fisheye::stereoRectify(Kl, Dl, Kr, Dr, cv::Size(input_width, input_height), R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                               cv::fisheye::CALIB_ZERO_DISPARITY, cv::Size(output_width, output_height), 0.0,
                               fov_scale_);
    cv::fisheye::initUndistortRectifyMap(Kl, Dl, Rl, Pl, cv::Size(output_width, output_height), CV_32FC1, undistmap1l,
                                         undistmap2l);
    cv::fisheye::initUndistortRectifyMap(Kr, Dr, Rr, Pr, cv::Size(output_width, output_height), CV_32FC1, undistmap1r,
                                         undistmap2r);
  } else {
    RCLCPP_ERROR_STREAM(logger_, "=> unsupported distortion_model: " << distortion_model_);
    return -1;
  }

  Q_ = Q;
  undistmap1l_ = undistmap1l;
  undistmap2l_ = undistmap2l;
  undistmap1r_ = undistmap1r;
  undistmap2r_ = undistmap2r;

  RCLCPP_WARN_STREAM(logger_, "=> input_resolution: " << "[" << input_width << ", " << input_height << "]");
  RCLCPP_WARN_STREAM(logger_, "=> cam_resolution: " << "[" << cam_resolution_[0] << ", " << cam_resolution_[1] << "]");
  RCLCPP_WARN_STREAM(logger_, "=> width, height scale: " << "[" << width_scale << ", " << height_scale << "]");
  RCLCPP_WARN_STREAM(logger_, "=> output_resolution: " << "[" << output_width << ", " << output_height << "]");
  RCLCPP_WARN_STREAM(logger_, "=> Kl: " << std::endl << Kl);
  RCLCPP_WARN_STREAM(logger_, "=> Dl: " << std::endl << Dl);
  RCLCPP_WARN_STREAM(logger_, "=> Kr: " << std::endl << Kr);
  RCLCPP_WARN_STREAM(logger_, "=> Dr: " << std::endl << Dr);
  RCLCPP_WARN_STREAM(logger_, "=> R_rl: " << std::endl << R_rl);
  RCLCPP_WARN_STREAM(logger_, "=> t_rl: " << std::endl << t_rl);
  RCLCPP_WARN_STREAM(logger_, "=> distortion_model: " << distortion_model_);
  if (distortion_model_ == "equidistant") RCLCPP_WARN_STREAM(logger_, "=> fov_scale: " << fov_scale_);
  if (distortion_model_ == "radtan" || distortion_model_ == "rational_polynomial")
    RCLCPP_WARN_STREAM(logger_, "=> alpha: " << alpha_);
  double fx = Q.at<double>(2, 3);
  double fy = Q.at<double>(2, 3);
  double cx = -Q.at<double>(0, 3);
  double cy = -Q.at<double>(1, 3);
  double baseline = std::abs(1 / Q.at<double>(3, 2));
  RCLCPP_WARN_STREAM(logger_, "=> rectify fx: " << fx << ", fy: " << fy << ", cx: " << cx << ", cy: " << cy
                                                << ", baseline: " << baseline);
  RCLCPP_WARN_STREAM(logger_, "=> ---------------------------------------------");
  float HFOV = 2 * atan(output_width / (2 * fx)) * 180 / M_PI;
  float VFOV = 2 * atan(output_height / (2 * fy)) * 180 / M_PI;
  RCLCPP_WARN_STREAM(logger_, "=> HFOV: " << HFOV << "°, VFOV: " << VFOV << "°");
  RCLCPP_WARN_STREAM(logger_, "=> ---------------------------------------------");

  undistmap_built_ = true;
  return 0;
}

void StereoRectify::rectify(const cv::Mat &left_image, const cv::Mat &right_image, cv::Mat &rectified_left_image,
                            cv::Mat &rectified_right_image) {
  cv::remap(left_image, rectified_left_image, undistmap1l_, undistmap2l_, cv::INTER_LINEAR);
  cv::remap(right_image, rectified_right_image, undistmap1r_, undistmap2r_, cv::INTER_LINEAR);
}

void StereoRectify::rectify_nv12(const uint8_t *left_nv12, const uint8_t *right_nv12, int input_w, int input_h,
                                 uint8_t *rect_left_nv12, uint8_t *rect_right_nv12, int output_w, int output_h) {
  cv::Mat left_y(input_h, input_w, CV_8UC1, const_cast<uint8_t *>(left_nv12));
  cv::Mat right_y(input_h, input_w, CV_8UC1, const_cast<uint8_t *>(right_nv12));

  cv::Mat left_uv(input_h / 2, input_w, CV_8UC1, const_cast<uint8_t *>(left_nv12 + input_w * input_h));
  cv::Mat right_uv(input_h / 2, input_w, CV_8UC1, const_cast<uint8_t *>(right_nv12 + input_w * input_h));

  cv::Mat rect_left_y(output_h, output_w, CV_8UC1, rect_left_nv12);
  cv::Mat rect_right_y(output_h, output_w, CV_8UC1, rect_right_nv12);

  cv::Mat rect_left_uv(output_h / 2, output_w, CV_8UC1, rect_left_nv12 + output_w * output_h);
  cv::Mat rect_right_uv(output_h / 2, output_w, CV_8UC1, rect_right_nv12 + output_w * output_h);

  // Y plane: use original undistort maps
  cv::remap(left_y, rect_left_y, undistmap1l_, undistmap2l_, cv::INTER_LINEAR);
  cv::remap(right_y, rect_right_y, undistmap1r_, undistmap2r_, cv::INTER_LINEAR);

  // UV plane: map needs half y coordinate
  cv::Mat uv_map1_l = undistmap1l_.clone();
  cv::Mat uv_map2_l = undistmap2l_ * 0.5f;
  cv::Mat uv_map1_r = undistmap1r_.clone();
  cv::Mat uv_map2_r = undistmap2r_ * 0.5f;

  cv::resize(uv_map1_l, uv_map1_l, cv::Size(output_w, output_h / 2), 0, 0, cv::INTER_LINEAR);
  cv::resize(uv_map2_l, uv_map2_l, cv::Size(output_w, output_h / 2), 0, 0, cv::INTER_LINEAR);
  cv::resize(uv_map1_r, uv_map1_r, cv::Size(output_w, output_h / 2), 0, 0, cv::INTER_LINEAR);
  cv::resize(uv_map2_r, uv_map2_r, cv::Size(output_w, output_h / 2), 0, 0, cv::INTER_LINEAR);

  cv::remap(left_uv, rect_left_uv, uv_map1_l, uv_map2_l, cv::INTER_LINEAR);
  cv::remap(right_uv, rect_right_uv, uv_map1_r, uv_map2_r, cv::INTER_LINEAR);
}

void StereoRectify::get_intrinsic(double &fx, double &fy, double &cx, double &cy, double &baseline) const {
  cv::Mat Q = Q_;
  fx = Q.at<double>(2, 3);
  fy = Q.at<double>(2, 3);
  cx = -Q.at<double>(0, 3);
  cy = -Q.at<double>(1, 3);
  baseline = std::abs(1 / Q.at<double>(3, 2));
}
