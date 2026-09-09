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

#include "stereo_rectify.h"

StereoRectify::StereoRectify(const std::string &stereo_calib_file_path, const rclcpp::Logger &logger)
    : stereo_calib_file_path_(stereo_calib_file_path), logger_(logger) {
  RCLCPP_WARN_STREAM(logger_, "=> -------- init StereoRectify ------------------");

  cv::FileStorage fs(stereo_calib_file_path_, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to open stereo calibration file: " << stereo_calib_file_path_);
    return;
  }

  /*
   * Read a matrix from a normal YAML nested sequence.
   *
   * Example:
   *
   * T_cn_cnm1:
   * - [r00, r01, r02, tx]
   * - [r10, r11, r12, ty]
   * - [r20, r21, r22, tz]
   * - [0.0, 0.0, 0.0, 1.0]
   *
   * OpenCV matrix format is also supported.
   */
  auto readYamlMatrix = [&](const cv::FileNode &node, int expected_rows, int expected_cols, cv::Mat &output) -> bool {
    if (node.empty()) {
      return false;
    }

    /*
     * Read OpenCV matrix format.
     */
    if (node.isMap() && !node["data"].empty()) {
      cv::Mat matrix;
      node >> matrix;
      if (matrix.empty()) {
        RCLCPP_ERROR_STREAM(logger_, "Failed to read OpenCV matrix");
        return false;
      }
      if (matrix.rows != expected_rows || matrix.cols != expected_cols) {
        RCLCPP_ERROR_STREAM(logger_, "Invalid matrix size. Expected " << expected_rows << "x" << expected_cols
                                                                      << ", but got " << matrix.rows << "x"
                                                                      << matrix.cols);
        return false;
      }
      matrix.convertTo(output, CV_64F);
      return true;
    }

    /*
     * Read a standard nested YAML sequence.
     */
    if (!node.isSeq()) {
      RCLCPP_ERROR_STREAM(logger_, "The matrix node is not a YAML sequence");
      return false;
    }
    if (static_cast<int>(node.size()) != expected_rows) {
      RCLCPP_ERROR_STREAM(logger_,
                          "Invalid matrix row count. Expected " << expected_rows << ", but got " << node.size());
      return false;
    }

    output = cv::Mat::zeros(expected_rows, expected_cols, CV_64F);
    int row_index = 0;
    for (cv::FileNodeIterator row_it = node.begin(); row_it != node.end(); ++row_it, ++row_index) {
      const cv::FileNode row_node = *row_it;
      if (!row_node.isSeq()) {
        RCLCPP_ERROR_STREAM(logger_, "Matrix row " << row_index << " is not a YAML sequence");
        return false;
      }
      if (static_cast<int>(row_node.size()) != expected_cols) {
        RCLCPP_ERROR_STREAM(logger_, "Invalid matrix column count at row " << row_index << ". Expected "
                                                                           << expected_cols << ", but got "
                                                                           << row_node.size());
        return false;
      }
      int col_index = 0;
      for (cv::FileNodeIterator col_it = row_node.begin(); col_it != row_node.end(); ++col_it, ++col_index) {
        output.at<double>(row_index, col_index) = static_cast<double>(*col_it);
      }
    }
    return true;
  };

  /*
   * Print all loaded calibration parameters.
   */
  auto printParameters = [&]() {
    RCLCPP_WARN_STREAM(logger_, "=> load stereo calib from: " << stereo_calib_file_path_);
    RCLCPP_WARN_STREAM(logger_, "=> Kl:" << std::endl << Kl_);
    RCLCPP_WARN_STREAM(logger_, "=> Dl:" << std::endl << Dl_);
    RCLCPP_WARN_STREAM(logger_, "=> Kr:" << std::endl << Kr_);
    RCLCPP_WARN_STREAM(logger_, "=> Dr:" << std::endl << Dr_);
    RCLCPP_WARN_STREAM(logger_, "=> R_rl:" << std::endl << R_rl_);
    RCLCPP_WARN_STREAM(logger_, "=> t_rl:" << std::endl << t_rl_);
    if (!t_rl_.empty()) {
      RCLCPP_WARN_STREAM(logger_, "=> baseline: " << cv::norm(t_rl_) << " m");
    }
    if (cam_resolution_.size() >= 2) {
      RCLCPP_WARN_STREAM(logger_, "=> cam_resolution: [" << cam_resolution_[0] << ", " << cam_resolution_[1] << "]");
    }
    RCLCPP_WARN_STREAM(logger_, "=> distortion_model: " << distortion_model_);
    if (distortion_model_ == "mei") {
      RCLCPP_WARN_STREAM(logger_, "=> rectify_model: " << rectify_model_);
    }
    if (distortion_model_ == "equidistant") {
      RCLCPP_WARN_STREAM(logger_, "=> target_hfov(deg): " << target_hfov_
                                                          << (target_hfov_ > 0 ? "" : " (auto max-no-black)"));
    }
    if (distortion_model_ == "radtan" || distortion_model_ == "rational_polynomial") {
      RCLCPP_WARN_STREAM(logger_, "=> alpha: " << alpha_);
    }
    if (distortion_model_ == "mei") {
      RCLCPP_WARN_STREAM(logger_, "=> target_hfov(deg): " << target_hfov_
                                                          << (target_hfov_ > 0 ? "" : " (auto max-no-black)"));
    }
    RCLCPP_WARN_STREAM(logger_, "=> ---------------------------------------------");
  };

  /*
   * Read the cam0/cam1 calibration format.
   *
   * Supported formats:
   *
   * stereo0:
   *   cam0:
   *     intrinsics: [...]
   *   cam1:
   *     intrinsics: [...]
   *     T_cn_cnm1: [...]
   *
   * Or:
   *
   * cam0:
   *   intrinsics: [...]
   * cam1:
   *   intrinsics: [...]
   *   T_cn_cnm1: [...]
   */
  auto extractParameters = [&](const cv::FileNode &stereo_node) -> bool {
    const cv::FileNode cam0_node = stereo_node["cam0"];
    const cv::FileNode cam1_node = stereo_node["cam1"];

    if (cam0_node.empty() || cam1_node.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "Invalid calibration file: cam0 or cam1 is missing");
      return false;
    }

    /*
     * Read camera intrinsics.
     * The expected order is [fx, fy, cx, cy].
     */
    std::vector<double> cam0_intrinsics;
    std::vector<double> cam1_intrinsics;

    cam0_node["intrinsics"] >> cam0_intrinsics;
    cam1_node["intrinsics"] >> cam1_intrinsics;

    if (cam0_intrinsics.size() != 4) {
      RCLCPP_ERROR_STREAM(logger_,
                          "Invalid cam0 intrinsics size: " << cam0_intrinsics.size() << ". Expected [fx, fy, cx, cy]");
      return false;
    }

    if (cam1_intrinsics.size() != 4) {
      RCLCPP_ERROR_STREAM(logger_,
                          "Invalid cam1 intrinsics size: " << cam1_intrinsics.size() << ". Expected [fx, fy, cx, cy]");
      return false;
    }

    cv::Mat Kl = cv::Mat::eye(3, 3, CV_64F);
    Kl.at<double>(0, 0) = cam0_intrinsics[0];
    Kl.at<double>(1, 1) = cam0_intrinsics[1];
    Kl.at<double>(0, 2) = cam0_intrinsics[2];
    Kl.at<double>(1, 2) = cam0_intrinsics[3];

    cv::Mat Kr = cv::Mat::eye(3, 3, CV_64F);
    Kr.at<double>(0, 0) = cam1_intrinsics[0];
    Kr.at<double>(1, 1) = cam1_intrinsics[1];
    Kr.at<double>(0, 2) = cam1_intrinsics[2];
    Kr.at<double>(1, 2) = cam1_intrinsics[3];

    /*
     * Read distortion coefficients.
     */
    std::vector<double> cam0_distortion_coeffs;
    std::vector<double> cam1_distortion_coeffs;
    cam0_node["distortion_coeffs"] >> cam0_distortion_coeffs;
    cam1_node["distortion_coeffs"] >> cam1_distortion_coeffs;
    if (cam0_distortion_coeffs.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "cam0 distortion_coeffs is empty");
      return false;
    }
    if (cam1_distortion_coeffs.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "cam1 distortion_coeffs is empty");
      return false;
    }

    cv::Mat Dl(1, static_cast<int>(cam0_distortion_coeffs.size()), CV_64F, cam0_distortion_coeffs.data());
    Dl = Dl.clone();
    cv::Mat Dr(1, static_cast<int>(cam1_distortion_coeffs.size()), CV_64F, cam1_distortion_coeffs.data());
    Dr = Dr.clone();

    /*
     * Read camera resolutions.
     */
    std::vector<int> cam0_resolution;
    std::vector<int> cam1_resolution;
    cam0_node["resolution"] >> cam0_resolution;
    cam1_node["resolution"] >> cam1_resolution;
    if (cam0_resolution.size() != 2) {
      RCLCPP_ERROR_STREAM(logger_, "Invalid cam0 resolution. Expected [width, height]");
      return false;
    }
    if (cam1_resolution.size() != 2) {
      RCLCPP_ERROR_STREAM(logger_, "Invalid cam1 resolution. Expected [width, height]");
      return false;
    }
    if (cam0_resolution != cam1_resolution) {
      RCLCPP_WARN_STREAM(logger_, "cam0 and cam1 resolutions are different. "
                                  "cam0=["
                                      << cam0_resolution[0] << ", " << cam0_resolution[1] << "], cam1=["
                                      << cam1_resolution[0] << ", " << cam1_resolution[1] << "]");
    }

    /*
     * Read distortion models.
     */
    std::string cam0_distortion_model;
    std::string cam1_distortion_model;
    cam0_node["distortion_model"] >> cam0_distortion_model;
    cam1_node["distortion_model"] >> cam1_distortion_model;
    if (cam0_distortion_model.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "cam0 distortion_model is missing");
      return false;
    }
    if (cam1_distortion_model.empty()) {
      RCLCPP_WARN_STREAM(logger_, "cam1 distortion_model is missing. "
                                  "Using the cam0 distortion model");
    } else if (cam0_distortion_model != cam1_distortion_model) {
      RCLCPP_WARN_STREAM(logger_, "cam0 and cam1 distortion models are different. "
                                  "cam0="
                                      << cam0_distortion_model << ", cam1=" << cam1_distortion_model);
    }

    /*
     * Read Mei unified camera model xi.
     *
     * Current YAML format:
     *
     * cam1:
     *   xi_left: ...
     *   xi_right: ...
     */
    double xi_l = 0.0;
    double xi_r = 0.0;
    if (cam0_distortion_model == "mei") {
      bool has_xi = false;
      // Preferred format: xi stored in each camera.
      if (!cam0_node["xi"].empty() && !cam1_node["xi"].empty()) {
        cam0_node["xi"] >> xi_l;
        cam1_node["xi"] >> xi_r;
        has_xi = true;
      }

      // Compatible with current Python output.
      if (!has_xi && !cam1_node["xi_left"].empty() && !cam1_node["xi_right"].empty()) {
        cam1_node["xi_left"] >> xi_l;
        cam1_node["xi_right"] >> xi_r;
        has_xi = true;
      }

      // Also allow them under stereo0.
      if (!has_xi && !stereo_node["xi_left"].empty() && !stereo_node["xi_right"].empty()) {
        stereo_node["xi_left"] >> xi_l;
        stereo_node["xi_right"] >> xi_r;
        has_xi = true;
      }

      if (!has_xi) {
        RCLCPP_ERROR_STREAM(logger_, "Mei model requires xi_left and xi_right");
        return false;
      }

      RCLCPP_WARN_STREAM(logger_, "=> xi_left: " << xi_l);
      RCLCPP_WARN_STREAM(logger_, "=> xi_right: " << xi_r);

      /*
       * Read the optional Mei rectification model.
       *
       * Supported values: "RECTIFY_PERSPECTIVE" (default), "RECTIFY_LONGLATI".
       * It can be stored under cam0, cam1, or the stereo0 node.
       */
      std::string rectify_model = "RECTIFY_PERSPECTIVE";
      if (!cam1_node["rectify_model"].empty()) {
        cam1_node["rectify_model"] >> rectify_model;
      } else if (!cam0_node["rectify_model"].empty()) {
        cam0_node["rectify_model"] >> rectify_model;
      } else if (!stereo_node["rectify_model"].empty()) {
        stereo_node["rectify_model"] >> rectify_model;
      }
      if (rectify_model != "RECTIFY_PERSPECTIVE" && rectify_model != "RECTIFY_LONGLATI") {
        RCLCPP_WARN_STREAM(logger_, "Unknown rectify_model: "
                                        << rectify_model << ", fallback to RECTIFY_PERSPECTIVE");
        rectify_model = "RECTIFY_PERSPECTIVE";
      }
      RCLCPP_WARN_STREAM(logger_, "=> rectify_model: " << rectify_model);
      rectify_model_ = rectify_model;
    }

    /*
     * Read the stereo extrinsic transformation.
     *
     * T_cn_cnm1 in cam1 represents T_cam1_cam0:
     *
     * p_cam1 = T_cam1_cam0 * p_cam0
     */
    cv::Mat T_cam1_cam0;
    bool has_stereo_extrinsic = readYamlMatrix(cam1_node["T_cn_cnm1"], 4, 4, T_cam1_cam0);

    /*
     * If T_cn_cnm1 does not exist, calculate the stereo
     * transformation from T_cam_imu:
     *
     * T_cam1_cam0 =
     *     T_cam1_imu * inverse(T_cam0_imu)
     */
    if (!has_stereo_extrinsic) {
      cv::Mat T_cam0_imu;
      cv::Mat T_cam1_imu;

      const bool has_cam0_imu = readYamlMatrix(cam0_node["T_cam_imu"], 4, 4, T_cam0_imu);
      const bool has_cam1_imu = readYamlMatrix(cam1_node["T_cam_imu"], 4, 4, T_cam1_imu);
      if (!has_cam0_imu || !has_cam1_imu) {
        RCLCPP_ERROR_STREAM(logger_, "Stereo extrinsic is missing. "
                                     "Neither cam1/T_cn_cnm1 nor both "
                                     "cam0/T_cam_imu and cam1/T_cam_imu are available");
        return false;
      }
      T_cam1_cam0 = T_cam1_imu * T_cam0_imu.inv();
      RCLCPP_WARN_STREAM(logger_, "cam1/T_cn_cnm1 is missing. "
                                  "T_cam1_cam0 was calculated from "
                                  "cam0/T_cam_imu and cam1/T_cam_imu");
    }
    if (T_cam1_cam0.empty() || T_cam1_cam0.rows != 4 || T_cam1_cam0.cols != 4) {
      RCLCPP_ERROR_STREAM(logger_, "Invalid stereo transformation matrix. "
                                   "Expected 4x4");
      return false;
    }
    T_cam1_cam0.convertTo(T_cam1_cam0, CV_64F);

    /*
     * Extract rotation and translation without modifying them.
     */
    cv::Mat R_rl = T_cam1_cam0(cv::Rect(0, 0, 3, 3)).clone();
    cv::Mat t_rl = T_cam1_cam0(cv::Rect(3, 0, 1, 3)).clone();
    if (R_rl.empty() || R_rl.rows != 3 || R_rl.cols != 3) {
      RCLCPP_ERROR_STREAM(logger_, "Invalid stereo rotation matrix. Expected 3x3");
      return false;
    }
    if (t_rl.empty() || t_rl.rows != 3 || t_rl.cols != 1) {
      RCLCPP_ERROR_STREAM(logger_, "Invalid stereo translation vector. Expected 3x1");
      return false;
    }

    /*
     * Read the optional pinhole alpha parameter.
     */
    float alpha = 0.0f;
    if (!cam1_node["alpha"].empty()) {
      cam1_node["alpha"] >> alpha;
    } else if (!cam0_node["alpha"].empty()) {
      cam0_node["alpha"] >> alpha;
    } else if (!stereo_node["alpha"].empty()) {
      stereo_node["alpha"] >> alpha;
    }

    /*
     * Read the optional Mei perspective target horizontal FOV (degrees).
     * If > 0, the perspective rectification focal is computed directly from
     * this FOV instead of searching the maximum no-black FOV. A narrower FOV
     * (larger focal) avoids black borders; a wider FOV may introduce them.
     */
    double target_hfov = 0.0;
    if (!cam1_node["target_hfov"].empty()) {
      cam1_node["target_hfov"] >> target_hfov;
    } else if (!cam0_node["target_hfov"].empty()) {
      cam0_node["target_hfov"] >> target_hfov;
    } else if (!stereo_node["target_hfov"].empty()) {
      stereo_node["target_hfov"] >> target_hfov;
    }

    /*
     * Save all parameters.
     */
    Kl_ = Kl;
    Kr_ = Kr;
    Dl_ = Dl;
    Dr_ = Dr;
    R_rl_ = R_rl;
    t_rl_ = t_rl;
    xi_l_ = xi_l;
    xi_r_ = xi_r;
    cam_resolution_ = cam0_resolution;
    distortion_model_ = cam0_distortion_model;
    alpha_ = alpha;
    target_hfov_ = target_hfov;
    printParameters();

    return true;
  };

  bool load_success = false;

  /*
   * Mode 1:
   * OpenCV stereo calibration format.
   *
   * cameraMatrix1
   * distCoeffs1
   * cameraMatrix2
   * distCoeffs2
   * R
   * T
   */
  if (!fs["cameraMatrix1"].empty()) {
    fs["cameraMatrix1"] >> Kl_;
    fs["distCoeffs1"] >> Dl_;
    fs["cameraMatrix2"] >> Kr_;
    fs["distCoeffs2"] >> Dr_;
    fs["R"] >> R_rl_;
    fs["T"] >> t_rl_;

    int width = 0;
    int height = 0;
    fs["image_width"] >> width;
    fs["image_height"] >> height;

    if (Kl_.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "cameraMatrix1 is empty");
    } else if (Kr_.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "cameraMatrix2 is empty");
    } else if (Dl_.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "distCoeffs1 is empty");
    } else if (Dr_.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "distCoeffs2 is empty");
    } else if (R_rl_.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "R is empty");
    } else if (t_rl_.empty()) {
      RCLCPP_ERROR_STREAM(logger_, "T is empty");
    } else if (width <= 0 || height <= 0) {
      RCLCPP_ERROR_STREAM(logger_, "Invalid image resolution: " << width << "x" << height);
    } else {
      Kl_.convertTo(Kl_, CV_64F);
      Kr_.convertTo(Kr_, CV_64F);
      Dl_.convertTo(Dl_, CV_64F);
      Dr_.convertTo(Dr_, CV_64F);
      R_rl_.convertTo(R_rl_, CV_64F);
      t_rl_.convertTo(t_rl_, CV_64F);

      /*
       * Convert a row translation vector to a column vector.
       */
      if (t_rl_.rows == 1 && t_rl_.cols == 3) {
        t_rl_ = t_rl_.t();
      }
      if (R_rl_.rows != 3 || R_rl_.cols != 3) {
        RCLCPP_ERROR_STREAM(logger_, "Invalid R size. Expected 3x3, but got " << R_rl_.rows << "x" << R_rl_.cols);
      } else if (t_rl_.rows != 3 || t_rl_.cols != 1) {
        RCLCPP_ERROR_STREAM(logger_, "Invalid T size. Expected 3x1, but got " << t_rl_.rows << "x" << t_rl_.cols);
      } else {
        cam_resolution_ = {width, height};
        /*
         * Infer the distortion model from the coefficient count.
         */
        distortion_model_ = Dl_.total() == 8 ? "rational_polynomial" : "radtan";
        if (!fs["distortion_model"].empty()) {
          fs["distortion_model"] >> distortion_model_;
        }


        alpha_ = 0.0f;
        if (!fs["alpha"].empty()) {
          fs["alpha"] >> alpha_;
        }

        load_success = true;
        printParameters();
      }
    }

    /*
     * Mode 2:
     * stereo0/cam0/cam1 format.
     */
  } else if (!fs["stereo0"].empty()) {
    load_success = extractParameters(fs["stereo0"]);
    /*
     * Mode 3:
     * Root-level cam0/cam1 format.
     *
     * This mode supports the new Kalibr/OpenVINS format.
     */
  } else if (!fs["cam0"].empty() && !fs["cam1"].empty()) {
    load_success = extractParameters(fs.root());
  } else {
    RCLCPP_ERROR_STREAM(logger_, "Unsupported stereo calibration format: " << stereo_calib_file_path_);
  }

  fs.release();

  if (!load_success) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to load stereo calibration parameters from: " << stereo_calib_file_path_);
    return;
  }

  RCLCPP_WARN_STREAM(logger_, "=> StereoRectify initialization completed");
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

  const cv::Size input_size(input_w, input_h);
  const cv::Size output_size(output_w, output_h);

  constexpr float kMinScale = 0.1f;
  constexpr float kInitialMaxScale = 2.0f;
  constexpr float kTolerance = 1e-4f;
  constexpr int kMaxIterations = 30;

  auto is_valid = [&](float scale) -> bool {
    cv::Mat Rl, Rr, Pl, Pr, Q;
    cv::Mat map1_l, map2_l;
    cv::Mat map1_r, map2_r;

    cv::fisheye::stereoRectify(Kl, Dl, Kr, Dr, input_size, R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                               cv::fisheye::CALIB_ZERO_DISPARITY, output_size, 0.0, scale);

    cv::fisheye::initUndistortRectifyMap(Kl, Dl, Rl, Pl, output_size, CV_32FC1, map1_l, map2_l);

    cv::fisheye::initUndistortRectifyMap(Kr, Dr, Rr, Pr, output_size, CV_32FC1, map1_r, map2_r);

    return !has_black_border(map1_l, map2_l, input_w, input_h) && !has_black_border(map1_r, map2_r, input_w, input_h);
  };

  /*
   * fov_scale:
   *
   * smaller -> larger focal -> smaller FOV -> easier to be valid
   * larger  -> smaller focal -> larger FOV -> easier to have black border
   *
   * We search the maximum valid scale.
   */

  float low = kMinScale;

  if (!is_valid(low)) {
    throw std::runtime_error("Even minimum fisheye fov_scale has black border");
  }

  /*
   * Find an invalid upper bound.
   */
  float high = kInitialMaxScale;

  while (is_valid(high)) {
    low = high;
    high *= 1.5f;

    if (high > 20.0f) {
      // Everything tested is valid.
      return low;
    }
  }

  /*
   * Now:
   *
   * low  -> valid
   * high -> invalid
   *
   * Search maximum valid scale.
   */
  for (int i = 0; i < kMaxIterations; ++i) {

    const float mid = 0.5f * (low + high);

    if (is_valid(mid)) {
      low = mid;
    } else {
      high = mid;
    }

    if ((high - low) < kTolerance) {
      break;
    }
  }

  return low;
}

/*
 * Search the fisheye fov_scale that yields a target horizontal FOV (degrees).
 *
 * The rectified focal after cv::fisheye::stereoRectify is encoded in Q(2,3); the
 * resulting HFOV is 2*atan(output_w / (2*fx)). Larger fov_scale -> smaller focal
 * -> larger HFOV (monotonic), so we binary-search the scale whose HFOV matches
 * the target. If the target exceeds the achievable maximum, the largest scale is
 * returned.
 */
static float find_fovscale_for_target_hfov(const cv::Mat &Kl, const cv::Mat &Dl, const cv::Mat &Kr, const cv::Mat &Dr,
                                           const cv::Mat &R_rl, const cv::Mat &t_rl, int input_w, int input_h,
                                           int output_w, int output_h, double target_hfov_deg) {
  const cv::Size input_size(input_w, input_h);
  const cv::Size output_size(output_w, output_h);

  auto hfov_deg = [&](float scale) -> double {
    cv::Mat Rl, Rr, Pl, Pr, Q;
    cv::fisheye::stereoRectify(Kl, Dl, Kr, Dr, input_size, R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                               cv::fisheye::CALIB_ZERO_DISPARITY, output_size, 0.0, scale);
    const double fx = Q.at<double>(2, 3);
    if (fx <= 0.0) return 0.0;
    return 2.0 * std::atan(static_cast<double>(output_w) / (2.0 * fx)) * 180.0 / M_PI;
  };

  constexpr float kMinScale = 0.1f;
  constexpr float kTolerance = 1e-3f;
  constexpr int kMaxIterations = 50;

  float low = kMinScale;
  float high = 2.0f;
  // expand upper bound until HFOV reaches/exceeds the target (capped)
  while (hfov_deg(high) < target_hfov_deg && high < 50.0f) {
    low = high;
    high *= 1.5f;
  }
  if (hfov_deg(high) < target_hfov_deg) {
    // target larger than achievable -> return the largest tried scale
    return high;
  }
  // low -> HFOV < target, high -> HFOV >= target
  for (int i = 0; i < kMaxIterations; ++i) {
    const float mid = 0.5f * (low + high);
    if (hfov_deg(mid) < target_hfov_deg) {
      low = mid;
    } else {
      high = mid;
    }
    if ((high - low) < kTolerance) break;
  }
  return 0.5f * (low + high);
}

static bool check_mei_no_black(const cv::Mat &Kl, const cv::Mat &Dl, double xi_l, const cv::Mat &Kr, const cv::Mat &Dr,
                               double xi_r, const cv::Mat &Rl, const cv::Mat &Rr, const cv::Size &input_size,
                               const cv::Size &output_size, double focal, double border = 1.0) {

  const double cx = (output_size.width - 1) * 0.5;

  const double cy = (output_size.height - 1) * 0.5;

  cv::Mat Knew = (cv::Mat_<double>(3, 3) << focal, 0.0, cx, 0.0, focal, cy, 0.0, 0.0, 1.0);

  cv::Mat map1_l, map2_l;
  cv::Mat map1_r, map2_r;

  cv::omnidir::initUndistortRectifyMap(Kl, Dl, xi_l, Rl, Knew, output_size, CV_32FC1, map1_l, map2_l,
                                       cv::omnidir::RECTIFY_PERSPECTIVE);

  cv::omnidir::initUndistortRectifyMap(Kr, Dr, xi_r, Rr, Knew, output_size, CV_32FC1, map1_r, map2_r,
                                       cv::omnidir::RECTIFY_PERSPECTIVE);

  auto map_valid = [&](const cv::Mat &map_x, const cv::Mat &map_y) -> bool {
    for (int y = 0; y < map_x.rows; ++y) {

      const float *px = map_x.ptr<float>(y);

      const float *py = map_y.ptr<float>(y);

      for (int x = 0; x < map_x.cols; ++x) {

        const float mx = px[x];
        const float my = py[x];

        if (!std::isfinite(mx) || !std::isfinite(my) || mx < border || mx > input_size.width - 1 - border ||
            my < border || my > input_size.height - 1 - border) {
          return false;
        }
      }
    }

    return true;
  };

  return map_valid(map1_l, map2_l) && map_valid(map1_r, map2_r);
}

static double find_min_valid_mei_focal_scale(const cv::Mat &Kl, const cv::Mat &Dl, double xi_l, const cv::Mat &Kr,
                                             const cv::Mat &Dr, double xi_r, const cv::Mat &Rl, const cv::Mat &Rr,
                                             const cv::Size &input_size, const cv::Size &output_size,
                                             double &base_focal) {

  /*
   * Use one common focal length for both x/y directions.
   * This guarantees:
   *
   * fx == fy
   *
   * after rectification.
   */

  const double focal_l = std::min(Kl.at<double>(0, 0), Kl.at<double>(1, 1));

  const double focal_r = std::min(Kr.at<double>(0, 0), Kr.at<double>(1, 1));

  /*
   * Use the smaller focal length of the two cameras
   * to preserve a larger common stereo FOV.
   */
  base_focal = std::min(focal_l, focal_r);

  /*
   * Scale according to output resolution.
   *
   * Use the smaller resolution scale to avoid
   * artificially increasing the focal length when
   * aspect ratio changes.
   */
  const double scale_x = static_cast<double>(output_size.width) / input_size.width;

  const double scale_y = static_cast<double>(output_size.height) / input_size.height;

  const double resolution_scale = std::min(scale_x, scale_y);

  base_focal *= resolution_scale;

  double low = 0.1;
  double high = 3.0;

  constexpr double kTolerance = 1e-4;
  constexpr int kMaxIterations = 30;
  constexpr double kBorder = 1.0;

  auto valid = [&](double focal_scale) {
    const double focal = base_focal * focal_scale;
    return check_mei_no_black(Kl, Dl, xi_l, Kr, Dr, xi_r, Rl, Rr, input_size, output_size, focal, kBorder);
  };

  /*
   * Make sure the upper bound is valid.
   */
  while (!valid(high)) {
    high *= 1.5;

    if (high > 20.0) {
      throw std::runtime_error("Cannot find valid Mei focal scale");
    }
  }

  /*
   * If low is already valid, continue searching
   * for an even smaller focal length.
   */
  while (valid(low) && low > 0.01) {
    high = low;
    low *= 0.5;
  }

  /*
   * Binary search:
   *
   * low  -> invalid, larger FOV
   * high -> valid, smaller FOV
   *
   * Find the minimum valid focal scale.
   */
  for (int i = 0; i < kMaxIterations; ++i) {

    const double mid = 0.5 * (low + high);

    if (valid(mid)) {
      high = mid;
    } else {
      low = mid;
    }

    if ((high - low) < kTolerance) {
      break;
    }
  }

  return high;
}

/*
 * Build a custom longitude-latitude (equirectangular / spherical) remap for the
 * Mei unified camera model.
 *
 * Each output pixel (col, row) is mapped to a unit-sphere ray in the rectified
 * frame using the convention:
 *
 *   tt = (col / (W-1) - 0.5) * PI        // longitude, [-PI/2, PI/2]
 *   pp = (row / (H-1) - 0.5) * PI        // latitude,  [-PI/2, PI/2]
 *   ray = (sin(tt), cos(tt)*sin(pp), cos(tt)*cos(pp))   // +Z = forward
 *
 * This convention matches the Python longlati disparity -> pointcloud
 * reconstruction (so the rectified image and the depth/pointcloud computation
 * are self-consistent). The ray is rotated by iR = R^-1 into the original
 * camera frame and projected through the Mei model (xi, K, D) to obtain the
 * source pixel. It is NOT cv::omnidir::RECTIFY_LONGLATI (which uses a
 * different (-cos(theta), ...) ray convention).
 *
 * Knew implicitly is diag((W-1)/PI, (H-1)/PI, 1) with cx=(W-1)/2, cy=(H-1)/2.
 */
static void build_mei_longlat_map(const cv::Mat &K, const cv::Mat &D, double xi, const cv::Mat &R,
                                  const cv::Size &output_size, cv::Mat &map1, cv::Mat &map2) {
  const int W = output_size.width;
  const int H = output_size.height;
  map1.create(H, W, CV_32FC1);
  map2.create(H, W, CV_32FC1);

  const double fx = K.at<double>(0, 0);
  const double fy = K.at<double>(1, 1);
  const double cx = K.at<double>(0, 2);
  const double cy = K.at<double>(1, 2);
  const double s = K.at<double>(0, 1); // skew, usually 0
  const double k1 = D.at<double>(0, 0);
  const double k2 = D.at<double>(0, 1);
  const double p1 = D.at<double>(0, 2);
  const double p2 = D.at<double>(0, 3);

  const cv::Mat iR = R.inv();

  for (int row = 0; row < H; ++row) {
    float *m1 = map1.ptr<float>(row);
    float *m2 = map2.ptr<float>(row);
    const double pp = (H > 1) ? (static_cast<double>(row) / (H - 1) - 0.5) * M_PI : 0.0;
    const double sin_pp = std::sin(pp);
    const double cos_pp = std::cos(pp);
    for (int col = 0; col < W; ++col) {
      const double tt = (W > 1) ? (static_cast<double>(col) / (W - 1) - 0.5) * M_PI : 0.0;
      const double sin_tt = std::sin(tt);
      const double cos_tt = std::cos(tt);

      // rectified-frame sphere ray (+Z forward)
      const double xt = sin_tt;
      const double yt = cos_tt * sin_pp;
      const double wt = cos_tt * cos_pp;

      // rotate into the original camera frame
      const double _x = iR.at<double>(0, 0) * xt + iR.at<double>(0, 1) * yt + iR.at<double>(0, 2) * wt;
      const double _y = iR.at<double>(1, 0) * xt + iR.at<double>(1, 1) * yt + iR.at<double>(1, 2) * wt;
      const double _w = iR.at<double>(2, 0) * xt + iR.at<double>(2, 1) * yt + iR.at<double>(2, 2) * wt;

      const double r = std::sqrt(_x * _x + _y * _y + _w * _w);
      const double Xs = _x / r;
      const double Ys = _y / r;
      const double Zs = _w / r;

      // Mei unified model projection
      const double denom = Zs + xi;
      if (!std::isfinite(denom) || std::abs(denom) < 1e-12) {
        // ray outside the camera FOV -> mark as invalid (black after remap)
        m1[col] = -1.0f;
        m2[col] = -1.0f;
        continue;
      }
      const double xu = Xs / denom;
      const double yu = Ys / denom;

      // polynomial distortion (k1, k2, p1, p2)
      const double r2 = xu * xu + yu * yu;
      const double r4 = r2 * r2;
      const double xd = (1 + k1 * r2 + k2 * r4) * xu + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu);
      const double yd = (1 + k1 * r2 + k2 * r4) * yu + p1 * (r2 + 2 * yu * yu) + 2 * p2 * xu * yu;

      const double u = fx * xd + s * yd + cx;
      const double v = fy * yd + cy;

      m1[col] = static_cast<float>(u);
      m2[col] = static_cast<float>(v);
    }
  }
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
    /*
     * If a target horizontal FOV (target_hfov_, degrees) is set, search the
     * fov_scale that yields it; otherwise search the maximum no-black fov_scale.
     * (The legacy yaml "fov_scale" field is not read, to avoid conflicts.)
     */
    if (target_hfov_ > 0.0 && target_hfov_ < 180.0) {
      fov_scale_ = find_fovscale_for_target_hfov(Kl, Dl, Kr, Dr, R_rl, t_rl, input_width, input_height, output_width,
                                                 output_height, target_hfov_);
      RCLCPP_WARN_STREAM(logger_, "=> fisheye target HFOV: " << target_hfov_ << " deg -> fov_scale " << fov_scale_);
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
  } else if (distortion_model_ == "mei") {
    const cv::Size input_size(input_width, input_height);
    const cv::Size output_size(output_width, output_height);
    /*
     * Compute stereo rectification rotations (shared by all rectify models).
     */
    cv::omnidir::stereoRectify(R_rl, t_rl, Rl, Rr);

    rectify_cx_ = (output_width - 1) * 0.5;
    rectify_cy_ = (output_height - 1) * 0.5;

    if (rectify_model_ == "RECTIFY_LONGLATI") {
      /*
       * Custom longitude-latitude (equirectangular / spherical) projection
       * matching the Python longlati convention (see build_mei_longlat_map):
       *   tt = (col/(W-1) - 0.5)*PI   in [-PI/2, PI/2]  (longitude, front hemisphere)
       *   pp = (row/(H-1) - 0.5)*PI   in [-PI/2, PI/2]  (latitude)
       *   ray = (sin(tt), cos(tt)*sin(pp), cos(tt)*cos(pp))   (+Z forward)
       *
       * Knew(0,0) = (W-1)/PI is the longitude pixels-per-radian and is the
       * focal used by the spherical disparity->pointcloud reconstruction:
       *   diff = disp / Knew(0,0)        (angular disparity, rad)
       *   R    = baseline * cos(tt-diff) / sin(diff)   (radial distance, m)
       * Regions outside the camera FOV stay black (invalid disparity/depth).
       */
      rectify_fx_ = (output_width - 1) / M_PI;  // longitude px per rad
      rectify_fy_ = (output_height - 1) / M_PI; // latitude  px per rad

      build_mei_longlat_map(Kl, Dl, xi_l_, Rl, output_size, undistmap1l, undistmap2l);
      build_mei_longlat_map(Kr, Dr, xi_r_, Rr, output_size, undistmap1r, undistmap2r);

      RCLCPP_WARN_STREAM(logger_, "=> LONGLATI custom map: f_lon(px/rad)=" << rectify_fx_
                                                                           << ", f_lat(px/rad)=" << rectify_fy_);
    } else {
      /*
       * Perspective projection.
       *
       * If a target horizontal FOV (target_hfov_, degrees) is provided, derive
       * the rectified focal directly from it:
       *   fx = output_width / (2 * tan(hfov/2))
       * which narrows (larger focal) or widens (smaller focal) the view to the
       * requested FOV. Otherwise search the maximum no-black FOV.
       */
      double rectify_focal = 0.0;
      if (target_hfov_ > 0.0 && target_hfov_ < 180.0) {
        const double hfov_rad = target_hfov_ * M_PI / 180.0;
        rectify_focal = static_cast<double>(output_width) / (2.0 * std::tan(hfov_rad / 2.0));
        // not from the focal search; record a neutral scale for logging
        mei_focal_scale_ = 1.0;
        RCLCPP_WARN_STREAM(logger_, "=> Mei perspective target HFOV: " << target_hfov_
                                                                        << " deg -> focal " << rectify_focal);
      } else {
        double base_focal = 0.0;
        mei_focal_scale_ =
            find_min_valid_mei_focal_scale(Kl, Dl, xi_l_, Kr, Dr, xi_r_, Rl, Rr, input_size, output_size, base_focal);
        rectify_focal = base_focal * mei_focal_scale_;
      }

      rectify_fx_ = rectify_focal;
      rectify_fy_ = rectify_focal;
      cv::Mat Knew =
          (cv::Mat_<double>(3, 3) << rectify_fx_, 0.0, rectify_cx_, 0.0, rectify_fy_, rectify_cy_, 0.0, 0.0, 1.0);

      cv::omnidir::initUndistortRectifyMap(Kl, Dl, xi_l_, Rl, Knew, output_size, CV_32FC1, undistmap1l, undistmap2l,
                                           cv::omnidir::RECTIFY_PERSPECTIVE);
      cv::omnidir::initUndistortRectifyMap(Kr, Dr, xi_r_, Rr, Knew, output_size, CV_32FC1, undistmap1r, undistmap2r,
                                           cv::omnidir::RECTIFY_PERSPECTIVE);
    }

    /*
     * Stereo baseline.
     */
    rectify_baseline_ = cv::norm(t_rl);

    /*
     * Construct Q manually. Q here only carries fx (= Knew(0,0)) and the
     * baseline for get_intrinsic(); the actual depth/pointcloud is computed
     * in stereonet_process / publish_pointcloud2.
     *
     * PERSPECTIVE: Z = fx * B / disparity            (forward depth, pinhole)
     * LONGLATI:    R = B * cos(tt - diff) / sin(diff) (radial distance, m)
     *              where tt = (col-cx)/fx, diff = disparity/fx
     */
    Q = cv::Mat::zeros(4, 4, CV_64F);
    Q.at<double>(0, 0) = 1.0;
    Q.at<double>(0, 3) = -rectify_cx_;
    Q.at<double>(1, 1) = 1.0;
    Q.at<double>(1, 3) = -rectify_cy_;
    Q.at<double>(2, 3) = rectify_fx_;
    Q.at<double>(3, 2) = 1.0 / rectify_baseline_;
    Q.at<double>(3, 3) = 0.0;
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
  if (distortion_model_ == "mei") {
    RCLCPP_WARN_STREAM(logger_, "=> rectify_model: " << rectify_model_);
    RCLCPP_WARN_STREAM(logger_, "=> xi: left=" << xi_l_ << ", right=" << xi_r_);
    if (rectify_model_ == "RECTIFY_PERSPECTIVE") {
      RCLCPP_WARN_STREAM(logger_, "=> Mei max no-black focal_scale: " << mei_focal_scale_);
    }
  }
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
