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

#include "epipolar_align.h"

static std::vector<cv::Point3f> create_chessboard_points(int cols, int rows, double square_size) {
  std::vector<cv::Point3f> obj_pts;
  obj_pts.reserve(cols * rows);

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      obj_pts.emplace_back(x * square_size, y * square_size, 0.0);
    }
  }
  return obj_pts;
}

void EpipolarAlign::check_epipolar_alignment(const cv::Mat &left_img, const cv::Mat &right_img,
                                             const cv::Size &pattern_size, double square_size,
                                             const std::shared_ptr<stereonet::CameraIntrinsic> &cam,
                                             cv::Mat &visualize) {
  if (left_img.empty() || right_img.empty()) return;
  cv::hconcat(left_img, right_img, visualize);

  // === 1. convert to gray image ===
  cv::Mat grayL, grayR;
  if (left_img.channels() == 3)
    cv::cvtColor(left_img, grayL, cv::COLOR_BGR2GRAY);
  else
    grayL = left_img.clone();

  if (right_img.channels() == 3)
    cv::cvtColor(right_img, grayR, cv::COLOR_BGR2GRAY);
  else
    grayR = right_img.clone();

  // === 2. detect corners ===
  std::vector<cv::Point2f> cornersL, cornersR;

  bool retL = cv::findChessboardCorners(grayL, pattern_size, cornersL);
  bool retR = cv::findChessboardCorners(grayR, pattern_size, cornersR);

  if (!retL || !retR) return;

  // === 3. sub pixel ===
  int win_size = static_cast<int>(std::min(grayL.rows, grayL.cols) / 100);
  win_size = std::max(5, win_size | 1);
  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 50, 1e-6);
  cv::cornerSubPix(grayL, cornersL, cv::Size(win_size, win_size), cv::Size(-1, -1), criteria);
  cv::cornerSubPix(grayR, cornersR, cv::Size(win_size, win_size), cv::Size(-1, -1), criteria);

  // === 4. calculate epipolar alignment error ===
  std::vector<double> dy, abs_dy;
  dy.reserve(cornersL.size());
  abs_dy.reserve(cornersL.size());

  for (size_t i = 0; i < cornersL.size(); ++i) {
    double diff = cornersL[i].y - cornersR[i].y;
    dy.push_back(diff);
    abs_dy.push_back(std::abs(diff));
  }

  // mean |Δy|
  double mean_abs = std::accumulate(abs_dy.begin(), abs_dy.end(), 0.0) / abs_dy.size();

  // RMSE
  double sq_sum = 0.0;
  for (double v : dy) sq_sum += v * v;
  double rmse = std::sqrt(sq_sum / dy.size());

  // std
  double mean = std::accumulate(dy.begin(), dy.end(), 0.0) / dy.size();
  double var = 0.0;
  for (double v : dy) var += (v - mean) * (v - mean);
  double stddev = std::sqrt(var / dy.size());

  double max_abs = *std::max_element(abs_dy.begin(), abs_dy.end());

  auto ratio = [&](double th) {
    int cnt = 0;
    for (double v : abs_dy)
      if (v <= th) cnt++;
    return 100.0 * cnt / abs_dy.size();
  };

  // === 5. reproject error ===
  std::vector<cv::Point3f> obj_pts = create_chessboard_points(pattern_size.width, pattern_size.height, square_size);
  // K, D
  cv::Mat K = (cv::Mat_<double>(3, 3) << cam->fx, 0, cam->cx, 0, cam->fy, cam->cy, 0, 0, 1);
  cv::Mat D = cv::Mat::zeros(5, 1, CV_64F); // assume already distorted
  cv::Mat rvec, tvec;
  bool pnp_ok = cv::solvePnP(obj_pts, cornersL, K, D, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
  if (!pnp_ok) return;

  // project to left image
  std::vector<cv::Point2f> proj_pts;
  cv::projectPoints(obj_pts, rvec, tvec, K, D, proj_pts);

  std::vector<double> reproj_err;
  for (size_t i = 0; i < proj_pts.size(); ++i) {
    reproj_err.push_back(cv::norm(cornersL[i] - proj_pts[i]));
  }

  double mean_reproj = std::accumulate(reproj_err.begin(), reproj_err.end(), 0.0) / reproj_err.size();
  double max_reproj = *std::max_element(reproj_err.begin(), reproj_err.end());

  // project to right image
  std::vector<cv::Point2f> proj_pts_right;
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  for (size_t i = 0; i < obj_pts.size(); ++i) {
    // world coordinate
    cv::Mat Xw = (cv::Mat_<double>(3, 1) << obj_pts[i].x, obj_pts[i].y, obj_pts[i].z);

    // left camera coordinate
    cv::Mat Xc_L = R * Xw + tvec;

    // right camera coordinate (baseline along +X)
    cv::Mat Xc_R = Xc_L.clone();
    Xc_R.at<double>(0) -= cam->baseline;

    // project to right image
    double x = Xc_R.at<double>(0);
    double y = Xc_R.at<double>(1);
    double z = Xc_R.at<double>(2);

    double u = cam->fx * x / z + cam->cx;
    double v = cam->fy * y / z + cam->cy;

    proj_pts_right.emplace_back(u, v);
  }
  std::vector<double> reproj_err_right;
  for (size_t i = 0; i < proj_pts_right.size(); ++i) {
    reproj_err_right.push_back(cv::norm(cornersR[i] - proj_pts_right[i]));
  }
  double mean_reproj_right =
      std::accumulate(reproj_err_right.begin(), reproj_err_right.end(), 0.0) / reproj_err_right.size();
  double max_reproj_right = *std::max_element(reproj_err_right.begin(), reproj_err_right.end());

  // ===  visualize ===
  int offset = left_img.cols;
  for (const auto &pt : cornersL) {
    cv::circle(visualize, pt, 4, cv::Scalar(0, 255, 0), -1);
  }
  for (size_t i = 0; i < cornersR.size(); ++i) {
    cv::Point2f p1 = cornersL[i];
    cv::Point2f p2 = cornersR[i];
    p2.x += offset;

    cv::circle(visualize, p2, 4, cv::Scalar(255, 0, 0), -1);

    if (i % (pattern_size.width + 1) == 0) {
      cv::line(visualize, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
  }
  for (size_t i = 0; i < proj_pts.size(); ++i) {
    cv::circle(visualize, proj_pts[i], 5, cv::Scalar(0, 0, 255), 1);
  }
  for (size_t i = 0; i < proj_pts_right.size(); ++i) {
    cv::Point2f p = proj_pts_right[i];
    p.x += offset;
    cv::circle(visualize, p, 5, cv::Scalar(0, 0, 255), 1);
  }

  std::vector<std::string> info_lines;
  info_lines.emplace_back("=================");
  info_lines.emplace_back("Epipolar Alignment Error:");
  info_lines.emplace_back(cv::format("mean |dy| = %.4f px", mean_abs));
  info_lines.emplace_back(cv::format("RMSE       = %.4f px", rmse));
  info_lines.emplace_back(cv::format("std        = %.4f px", stddev));
  info_lines.emplace_back(cv::format("max |dy|   = %.4f px", max_abs));
  info_lines.emplace_back(cv::format("<= 0.5 px  = %.1f %%", ratio(0.5)));
  info_lines.emplace_back(cv::format("<= 1.0 px  = %.1f %%", ratio(1.0)));
  info_lines.emplace_back(cv::format("<= 2.0 px  = %.1f %%", ratio(2.0)));
  info_lines.emplace_back("=================");
  info_lines.emplace_back("Reprojection Error:");
  info_lines.emplace_back(cv::format("left mean  = %.4f px", mean_reproj));
  info_lines.emplace_back(cv::format("left max   = %.4f px", max_reproj));
  info_lines.emplace_back(cv::format("right mean = %.4f px", mean_reproj_right));
  info_lines.emplace_back(cv::format("right max  = %.4f px", max_reproj_right));
  info_lines.emplace_back("=================");

  int x0 = 10;
  int y0 = 25;
  double font_scale = std::min(visualize.cols, visualize.rows) / 720.0 * 0.6;
  font_scale = std::clamp(font_scale, 0.4, 1.5);
  int line_height = static_cast<int>(font_scale * 40);
  int thickness = std::max(1, static_cast<int>(std::round(font_scale * 1.8)));
  thickness = std::min(thickness, 3);
  for (size_t i = 0; i < info_lines.size(); ++i) {
    // cv::putText(visualize, info_lines[i], cv::Point(x0, y0 + static_cast<int>(i) * line_height),
    // cv::FONT_HERSHEY_SIMPLEX, font_scale, CV_RGB(255, 0, 0), thickness);
    // outline
    cv::putText(visualize, info_lines[i], cv::Point(x0, y0 + static_cast<int>(i) * line_height),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), thickness + 2);
    // text
    cv::putText(visualize, info_lines[i], cv::Point(x0, y0 + static_cast<int>(i) * line_height),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
  }
}