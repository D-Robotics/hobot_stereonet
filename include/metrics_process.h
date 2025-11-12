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

#ifndef HOBOT_STEREONET_INCLUDE_METRICS_PROCESS_H_
#define HOBOT_STEREONET_INCLUDE_METRICS_PROCESS_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

struct MetricsProcess{
//----------------------------------------
// EPE (Endpoint Error)
//----------------------------------------
  static cv::Mat dispToDepth(const cv::Mat& disp, double fx, double baseline) {
    cv::Mat depth;
    disp.convertTo(depth, CV_64F);
    depth = fx * baseline / (depth + 1e-7); // avoid divide by zero
    return depth;
  }

  static double calculateEPE(const cv::Mat& gt_disp, const cv::Mat& pred_disp) {
    CV_Assert(gt_disp.size() == pred_disp.size());
    cv::Mat gt, pred;
    gt_disp.convertTo(gt, CV_64F);
    pred_disp.convertTo(pred, CV_64F);

    cv::Mat valid_mask = (gt > 0) & (pred > 0) & (gt <= 192) & (pred <= 192);

    int valid_count = cv::countNonZero(valid_mask);
    if (valid_count == 0)
      return std::numeric_limits<double>::quiet_NaN();

    cv::Mat abs_error;
    cv::absdiff(gt, pred, abs_error);
    double mean_error = cv::mean(abs_error, valid_mask)[0];
    return mean_error;
  }

//----------------------------------------
// Bad-2 / Bad-4 pixel ratio
//----------------------------------------
  static std::pair<double, double> calculateBadPixels(const cv::Mat& gt_disp, const cv::Mat& pred_disp,
                                               double threshold_2 = 2.0, double threshold_4 = 4.0) {
    CV_Assert(gt_disp.size() == pred_disp.size());
    cv::Mat gt, pred;
    gt_disp.convertTo(gt, CV_64F);
    pred_disp.convertTo(pred, CV_64F);

    cv::Mat valid_mask = (gt > 0) & (pred > 0) & (gt <= 192) & (pred <= 192);
    int valid_count = cv::countNonZero(valid_mask);
    if (valid_count == 0)
      return { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN() };

    cv::Mat abs_error;
    cv::absdiff(gt, pred, abs_error);
    double bad2 = cv::countNonZero((abs_error > threshold_2) & valid_mask) / static_cast<double>(valid_count);
    double bad4 = cv::countNonZero((abs_error > threshold_4) & valid_mask) / static_cast<double>(valid_count);
    return { bad2, bad4 };
  }

//----------------------------------------
// FNPR (False Negative Prediction Rate)
//----------------------------------------
  static double calculateFNPR(const cv::Mat& gt_disp, const cv::Mat& pred_disp,
                       double fx, double baseline,
                       double depth_threshold = 2.0, double error_threshold = 6.0) {
    CV_Assert(gt_disp.size() == pred_disp.size());
    cv::Mat gt, pred;
    gt_disp.convertTo(gt, CV_64F);
    pred_disp.convertTo(pred, CV_64F);

    cv::Mat valid_mask = (gt > 0) & (pred > 0) & (gt <= 192) & (pred <= 192);
    cv::Mat gt_depth = dispToDepth(gt, fx, baseline);

    cv::Mat near_mask = (gt_depth < depth_threshold) & valid_mask;
    int near_count = cv::countNonZero(near_mask);
    if (near_count == 0)
      return std::numeric_limits<double>::quiet_NaN();

    cv::Mat error = gt - pred;
    double fnpr = cv::countNonZero((error > error_threshold) & near_mask) / static_cast<double>(near_count);
    return fnpr;
  }

//----------------------------------------
// FFPR (False Far Prediction Rate)
//----------------------------------------
  static double calculateFFPR(const cv::Mat& gt_disp, const cv::Mat& pred_disp,
                       double fx, double baseline,
                       double depth_threshold = 2.0, double error_threshold = 6.0) {
    CV_Assert(gt_disp.size() == pred_disp.size());
    cv::Mat gt, pred;
    gt_disp.convertTo(gt, CV_64F);
    pred_disp.convertTo(pred, CV_64F);

    cv::Mat valid_mask = (gt > 0) & (pred > 0) & (gt <= 192) & (pred <= 192);
    cv::Mat gt_depth = dispToDepth(gt, fx, baseline);

    cv::Mat near_mask = (gt_depth < depth_threshold) & valid_mask;
    int near_count = cv::countNonZero(near_mask);
    if (near_count == 0)
      return std::numeric_limits<double>::quiet_NaN();

    cv::Mat error = pred - gt;
    double ffpr = cv::countNonZero((error > error_threshold) & near_mask) / static_cast<double>(near_count);
    return ffpr;
  }

//----------------------------------------
// A99 Depth Relative Error
//----------------------------------------
  static std::vector<double> calculateA99DepthRelativeError(const cv::Mat& gt_disp, const cv::Mat& pred_disp,
                                                     const std::vector<std::pair<double, double>>& ranges,
                                                     double fx, double baseline) {
    CV_Assert(gt_disp.size() == pred_disp.size());
    cv::Mat gt, pred;
    gt_disp.convertTo(gt, CV_64F);
    pred_disp.convertTo(pred, CV_64F);

    cv::Mat valid_mask = (gt > 0) & (pred > 0) & (gt <= 192) & (pred <= 192);
    cv::Mat gt_depth = dispToDepth(gt, fx, baseline);
    std::vector<double> results;

    for (auto r : ranges) {
      double lower = r.first, upper = r.second;
      cv::Mat range_mask = (gt_depth >= lower) & (gt_depth < upper) & valid_mask;

      int count = cv::countNonZero(range_mask);
      if (count == 0) {
        results.push_back(std::numeric_limits<double>::quiet_NaN());
        continue;
      }

      cv::Mat abs_error;
      cv::absdiff(gt, pred, abs_error);
      cv::Mat relative_error = abs_error / gt;

      std::vector<double> rel_vals;
      rel_vals.reserve(count);

      for (int i = 0; i < gt.rows; ++i) {
        const double* mask_ptr = range_mask.ptr<double>(i);
        const double* rel_ptr = relative_error.ptr<double>(i);
        for (int j = 0; j < gt.cols; ++j) {
          if (mask_ptr[j])
            rel_vals.push_back(rel_ptr[j]);
        }
      }

      std::sort(rel_vals.begin(), rel_vals.end());
      int idx = static_cast<int>(0.99 * rel_vals.size());
      results.push_back(rel_vals[idx]);
    }

    return results;
  }

//----------------------------------------
// Infinity Metric
//----------------------------------------
  static double calculateInfinityMetric(const cv::Mat& gt_disp, const cv::Mat& pred_disp,
                                 double fx, double baseline,
                                 double depth_threshold = 10.0, double pred_threshold = 7.5) {
    CV_Assert(gt_disp.size() == pred_disp.size());
    cv::Mat gt, pred;
    gt_disp.convertTo(gt, CV_64F);
    pred_disp.convertTo(pred, CV_64F);

    cv::Mat gt_depth = dispToDepth(gt, fx, baseline);
    cv::Mat pred_depth = dispToDepth(pred, fx, baseline);

    cv::Mat valid_mask = (gt > 0) & (pred > 0) & (gt <= 192) & (pred <= 192);
    cv::Mat far_mask = (gt_depth > depth_threshold) & valid_mask;

    int far_count = cv::countNonZero(far_mask);
    if (far_count == 0)
      return std::numeric_limits<double>::quiet_NaN();

    cv::Mat infinity_mask = (pred_depth <= pred_threshold) & far_mask;
    double ratio = cv::countNonZero(infinity_mask) / static_cast<double>(far_count);
    return ratio;
  }

};


#endif //HOBOT_STEREONET_INCLUDE_METRICS_PROCESS_H_
