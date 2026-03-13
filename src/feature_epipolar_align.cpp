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

#include "feature_epipolar_align.h"

void FeatureEpipolarAlign::check_epipolar_alignment(const cv::Mat &left_img, const cv::Mat &right_img,
                                                    const std::shared_ptr<stereonet::CameraIntrinsic> &cam,
                                                    cv::Mat &visualize) {
  if (left_img.empty() || right_img.empty()) return;

  // === 1. detect orb ===
  auto orb = cv::ORB::create(2000);
  std::vector<cv::KeyPoint> kp1, kp2;
  cv::Mat desc1, desc2;
  orb->detectAndCompute(left_img, cv::noArray(), kp1, desc1);
  orb->detectAndCompute(right_img, cv::noArray(), kp2, desc2);
  if (desc1.empty() || desc2.empty()) return;

  // === 2. match ===
  std::vector<cv::DMatch> matches;
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  matcher.match(desc1, desc2, matches);
  if (matches.empty()) return;

  // === 3. sort by distance ===
  std::sort(matches.begin(), matches.end(),
            [](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });
  int use_match = std::min(100, (int)matches.size());
  std::vector<cv::DMatch> good_matches(matches.begin(), matches.begin() + use_match);

  // === 4. calculate epipolar alignment error & reprojection error ===
  cv::Mat right_with_proj = right_img.clone();
  double total_reproj_error = 0.0;

  for (const auto &m : good_matches) {
    const auto &kpL = kp1[m.queryIdx];
    const auto &kpR = kp2[m.trainIdx];

    float xL = kpL.pt.x;
    float yL = kpL.pt.y;
    float xR = kpR.pt.x;
    float yR = kpR.pt.y;

    float disparity = xL - xR;
    if (disparity <= 0.1f) continue; // avoid divide by zero

    // depth
    float Z = cam->fx * cam->baseline / disparity;

    // backproject to 3D
    float X = (xL - cam->cx) * Z / cam->fx;
    float Y = (yL - cam->cy) * Z / cam->fy;

    // transform to right camera
    float Xr = X - cam->baseline;

    // project
    float xR_proj = cam->fx * Xr / Z + cam->cx;
    float yR_proj = cam->fy * Y / Z + cam->cy;

    // calculate reprojection error
    float err = std::sqrt((xR_proj - xR) * (xR_proj - xR) + (yR_proj - yR) * (yR_proj - yR));
    total_reproj_error += err;

    // draw reprojection point
    cv::circle(right_with_proj, cv::Point2f(xR_proj, yR_proj), 5, cv::Scalar(255, 0, 0), -1);
  }
  double mean_reproj_error = total_reproj_error / good_matches.size();

  // === 4. show ===
  cv::drawMatches(left_img, kp1, right_with_proj, kp2, good_matches, visualize, cv::Scalar::all(-1), cv::Scalar::all(-1),
                  std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  if (good_matches.empty()) return;

  std::vector<double> dy_abs;
  dy_abs.reserve(good_matches.size());
  double sum = 0.0, sum_sq = 0.0;
  double max_abs = 0.0;
  double min_abs = 1e9;
  for (const auto &m : good_matches) {
    double dy = kp1[m.queryIdx].pt.y - kp2[m.trainIdx].pt.y;
    double abs_dy = std::abs(dy);
    dy_abs.push_back(abs_dy);
    sum += abs_dy;
    max_abs = std::max(max_abs, abs_dy);
    min_abs = std::min(min_abs, abs_dy);
  }
  double mean_abs = sum / dy_abs.size();

  auto ratio = [&](double th) -> double {
    int count = std::count_if(dy_abs.begin(), dy_abs.end(), [&](double v) { return v <= th; });
    return 100.0 * count / dy_abs.size();
  };

  std::vector<std::string> info_lines;
  info_lines.emplace_back("=================");
  info_lines.emplace_back("Epipolar Alignment Error:");
  info_lines.emplace_back(cv::format("match cnt = %zu", good_matches.size()));
  info_lines.emplace_back(cv::format("mean |dy|  = %.4f px", mean_abs));
  info_lines.emplace_back(cv::format("min |dy|   = %.4f px", min_abs));
  info_lines.emplace_back(cv::format("max |dy|   = %.4f px", max_abs));
  info_lines.emplace_back(cv::format("<= 1.0 px  = %.1f %%", ratio(1.0)));
  info_lines.emplace_back(cv::format("<= 2.0 px  = %.1f %%", ratio(2.0)));
  info_lines.emplace_back(cv::format("<= 3.0 px  = %.1f %%", ratio(3.0)));
  info_lines.emplace_back("=================");
  info_lines.emplace_back(cv::format("mean reproj error = %.4f px", mean_reproj_error));
  info_lines.emplace_back("=================");

  int x0 = 10;
  int y0 = 25;
  double font_scale = std::min(visualize.cols, visualize.rows) / 720.0 * 0.6;
  font_scale = std::clamp(font_scale, 0.4, 1.5);
  int line_height = static_cast<int>(font_scale * 40);
  int thickness = std::max(1, static_cast<int>(std::round(font_scale * 1.8)));
  thickness = std::min(thickness, 3);

  for (size_t i = 0; i < info_lines.size(); ++i) {
    cv::putText(visualize, info_lines[i], cv::Point(x0, y0 + static_cast<int>(i) * line_height),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), thickness + 2);
    cv::putText(visualize, info_lines[i], cv::Point(x0, y0 + static_cast<int>(i) * line_height),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
  }
}