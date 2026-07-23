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
#include <fstream>
#include <sstream>
#include <string>
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

struct MetricsJsonWriter {
 public:
  MetricsJsonWriter() {
    doc_.SetObject();
  }

  // Add metrics for one image
  void add_image_metrics(
      const std::string& image_name,
      double epe,
      double bad2,
      double bad4,
      double fnpr,
      double ffpr,
      const std::vector<double>& a99_depth_relative_error,
      double infinity_metric
  ) {
    auto& allocator = doc_.GetAllocator();
    rapidjson::Value entry(rapidjson::kObjectType);

    entry.AddMember("EPE", double_to_json(epe, allocator), allocator);
    entry.AddMember("Bad-2", double_to_json(bad2, allocator), allocator);
    entry.AddMember("Bad-4", double_to_json(bad4, allocator), allocator);
    entry.AddMember("FNPR", double_to_json(fnpr, allocator), allocator);
    entry.AddMember("FFPR", double_to_json(ffpr, allocator), allocator);

    // A99 Depth Relative Error array
    rapidjson::Value a99_array(rapidjson::kArrayType);
    for (double v : a99_depth_relative_error) {
      a99_array.PushBack(double_to_json(v, allocator), allocator);
    }
    entry.AddMember("A99 Depth Relative Error", a99_array, allocator);

    entry.AddMember("Infinity metric", double_to_json(infinity_metric, allocator), allocator);

    rapidjson::Value key(image_name.c_str(), allocator);
    doc_.AddMember(key, entry, allocator);
  }

  // Save JSON to file
  bool save_to_file(const std::string& filename, bool pretty = true) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return false;
    ofs << to_string(pretty);
    ofs.close();
    return true;
  }

  // Load JSON from file
  bool load_from_file(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) return false;

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    ifs.close();

    doc_.Parse(buffer.str().c_str());
    return !doc_.HasParseError();
  }

  // Convert JSON to string
  std::string to_string(bool pretty = true) const {
    rapidjson::StringBuffer buffer;
    if (pretty) {
      rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
      doc_.Accept(writer);
    } else {
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      doc_.Accept(writer);
    }
    return buffer.GetString();
  }

  // Compute average of a specific scalar metric (e.g., "EPE", "Bad-2")
  double average_metric(const std::string& metric_name) const {
    if (!doc_.IsObject()) return std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    size_t count = 0;

    for (auto it = doc_.MemberBegin(); it != doc_.MemberEnd(); ++it) {
      const auto& entry = it->value;
      if (entry.HasMember(metric_name.c_str()) && entry[metric_name.c_str()].IsNumber()) {
        sum += entry[metric_name.c_str()].GetDouble();
        ++count;
      }
    }
    std::cout << "metric_name: " << metric_name << ", count: " << count << std::endl;
    return (count > 0) ? (sum / count) : std::numeric_limits<double>::quiet_NaN();
  }

  // Compute average of A99 Depth Relative Error per position across all images, ignoring nulls
  std::vector<double> average_a99_depth_relative_error() const {
    std::vector<double> sums(3, 0.0);
    std::vector<size_t> counts(3, 0);

    if (!doc_.IsObject()) return std::vector<double>{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};

    for (auto it = doc_.MemberBegin(); it != doc_.MemberEnd(); ++it) {
      const auto& entry = it->value;
      if (!entry.HasMember("A99 Depth Relative Error")) continue;

      const auto& arr = entry["A99 Depth Relative Error"];
      if (!arr.IsArray()) continue;

      for (rapidjson::SizeType i = 0; i < arr.Size() && i < 3; ++i) {
        if (arr[i].IsNumber()) {
          sums[i] += arr[i].GetDouble();
          counts[i]++;
        }
      }
    }

    std::vector<double> averages(3, std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < 3; ++i) {
      if (counts[i] > 0) averages[i] = sums[i] / counts[i];
    }
    return averages;
  }

  std::vector<double> get_metrics(const std::string& image_name, const std::string& metric_name) const {
    std::vector<double> nans;
    nans.push_back(std::numeric_limits<double>::quiet_NaN());
    if (!doc_.IsObject()) return nans;
    if (!doc_.HasMember(image_name.c_str())) return nans;

    const rapidjson::Value& entry = doc_[image_name.c_str()];
    if (!entry.HasMember(metric_name.c_str())) return nans;

    const rapidjson::Value& value = entry[metric_name.c_str()];
    if (value.IsNull()) return nans;
    if (value.IsNumber()) return std::vector<double>{value.GetDouble()};

    if (value.IsArray()) {
      std::vector<double> v;
      for (rapidjson::SizeType i = 0; i < value.GetArray().Size() && i < 3; ++i) {
        if (value.GetArray()[i].IsNumber()) {
          v.push_back(value.GetArray()[i].GetDouble());
        } else {
          v.push_back(std::numeric_limits<double>::quiet_NaN());
        }
      }
      return v;
    }
    std::cout << "image_name: " << image_name << ", metric_name: " << metric_name << std::endl;
    return nans;
  }

  // Get number of images
  size_t image_count() const {
    if (!doc_.IsObject()) return 0;
    return doc_.MemberCount();
  }

  rapidjson::Value double_to_json(double value, rapidjson::Document::AllocatorType& allocator) const {
    if (std::isnan(value)) {
      rapidjson::Value v;
      v.SetNull();
      return v; // Rely on move constructor
    } else {
      return rapidjson::Value(value);
    }
  }

 private:
  rapidjson::Document doc_;
};

struct MetricsProcess{
 public:
  // Min disparity threshold. Derived from the user's max display depth:
  // min_disparity = fx * baseline / max_depth. Pixels with disparity <= this
  // are treated as invalid (too far / infinite).
  static void set_min_disparity(double v) { min_disparity_ = v; }
  static double min_disparity() { return min_disparity_; }

  // Max disparity ceiling (default 192, the model's max_disp). Pixels with
  // disparity > this are treated as invalid.
  static void set_max_disparity(double v) { max_disparity_ = v; }
  static double max_disparity() { return max_disparity_; }

//----------------------------------------
// EPE (Endpoint Error)
//----------------------------------------
  static cv::Mat dispToDepth(const cv::Mat& disp, double fx,
      double baseline, double min_depth = 1e-2, double max_depth = 50) {
    cv::Mat depth, depth2;
    disp.convertTo(depth, CV_64F);
    depth = fx * baseline / (depth); // avoid divide by zero
    depth = cv::max(depth, min_depth);
    depth = cv::min(depth, max_depth);
    return depth;
  }

  static double calculateEPE(const cv::Mat& gt_disp, const cv::Mat& pred_disp) {
    CV_Assert(gt_disp.size() == pred_disp.size());
    cv::Mat gt, pred;
    gt_disp.convertTo(gt, CV_64F);
    pred_disp.convertTo(pred, CV_64F);

    cv::Mat valid_mask = (gt > min_disparity()) & (pred > min_disparity()) &
                         (gt <= max_disparity()) & (pred <= max_disparity());

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

    cv::Mat valid_mask = (gt > min_disparity()) & (pred > min_disparity()) &
                         (gt <= max_disparity()) & (pred <= max_disparity());
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

    cv::Mat valid_mask = (gt > min_disparity()) & (pred > min_disparity()) &
                         (gt <= max_disparity()) & (pred <= max_disparity());
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

    cv::Mat valid_mask = (gt > min_disparity()) & (pred > min_disparity()) &
                         (gt <= max_disparity()) & (pred <= max_disparity());
    cv::Mat gt_depth = dispToDepth(gt, fx, baseline);

    cv::Mat near_mask = (gt_depth < depth_threshold) & valid_mask;
    int near_count = cv::countNonZero(near_mask);
    if (near_count == 0)
      return std::numeric_limits<double>::quiet_NaN();

    cv::Mat error = pred - gt;
    double ffpr = cv::countNonZero((error > error_threshold) & near_mask) / static_cast<double>(near_count);
    return ffpr;
  }

  static int findFirstLessThan(const std::vector<double>& vec, double target) {
    auto it = std::lower_bound(vec.begin(), vec.end(), target);
    if (it == vec.begin()) {
      return -1;
    }
    return std::distance(vec.begin(), it) - 1;
  }

//----------------------------------------
// A99 Depth Relative Error
//----------------------------------------
  static std::vector<double> calculateA99DepthRelativeError(const std::string &file_name,
      const cv::Mat& gt_disp, const cv::Mat& pred_disp,
      const std::vector<std::pair<double, double>>& ranges,
      double fx, double baseline, bool dump_error_image = false) {
    CV_Assert(gt_disp.size() == pred_disp.size());
    cv::Mat gt, pred;
    gt_disp.convertTo(gt, CV_64F);
    pred_disp.convertTo(pred, CV_64F);
    cv::Mat valid_mask = (gt > min_disparity()) & (pred > min_disparity()) &
                         (gt <= max_disparity()) & (pred <= max_disparity());
    cv::Mat gt_depth = dispToDepth(gt, fx, baseline);
    cv::Mat pred_depth = dispToDepth(pred, fx, baseline);
    std::vector<double> results;
    int index = 0;

    std::vector<double> request_precision {0.03, 0.06, 0.22};

    for (auto r : ranges) {
      index++;
      double max_val;
      cv::Point max_pt;
      double lower = r.first, upper = r.second;
      cv::Mat range_mask = (gt_depth >= lower) & (gt_depth < upper) & valid_mask;
      int count = cv::countNonZero(range_mask);
      if (count == 0) {
        results.push_back(std::numeric_limits<double>::quiet_NaN());
        continue;
      }
      cv::Mat abs_error;
      cv::absdiff(gt_depth, pred_depth, abs_error);
      cv::Mat relative_error = abs_error / gt_depth;
      std::vector<double> rel_vals;
      rel_vals.reserve(count);
      for (int i = 0; i < gt_depth.rows; ++i) {
        const uint8_t* mask_ptr = range_mask.ptr<uint8_t>(i);
        const double* rel_ptr = relative_error.ptr<double>(i);
        for (int j = 0; j < gt_depth.cols; ++j) {
          if (mask_ptr[j] != 0 && !std::isnan(rel_ptr[j]) && !std::isinf(rel_ptr[j]))
            rel_vals.push_back(rel_ptr[j]);
        }
      }

      if (dump_error_image) {
        cv::Mat norm_u8, log_img;
        cv::Mat relative_error_visual;
        cv::Mat filtered;
        cv::log(relative_error + 1, log_img);
        cv::normalize(log_img, norm_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
        norm_u8.copyTo(filtered, range_mask);
        //  cv::convertScaleAbs(feat_visual, feat_visual, 2);
        cv::applyColorMap(norm_u8, norm_u8, cv::COLORMAP_JET);
        cv::imwrite("./result/a99_" + std::to_string(index - 1)
                    + "_relative_error_" + file_name + ".jpeg", norm_u8);
      }

      cv::minMaxLoc(relative_error, nullptr, &max_val, nullptr, &max_pt, range_mask);
      std::cout << "in " << file_name << " a99_" << index << ", max_error: " << max_val << ", pt: " << max_pt
                << ", gt: " << gt_depth.at<double>(max_pt) << "m, pred: "
                << pred_depth.at<double>(max_pt) << "m" << std::endl;
      //  ascending by default
      std::sort(rel_vals.begin(), rel_vals.end());
//      std::ofstream relerror(file_name + "_a99_" + std::to_string(index) + ".txt", std::ios::out);
//      for (const auto &v: rel_vals) {
//        relerror << v << std::endl;
//      }
//      relerror.close();
      int request_idx = findFirstLessThan(rel_vals, request_precision[index - 1]);
      std::cout << std::fixed << std::setprecision(3)
                << "(" << request_idx + 1 << "/" << rel_vals.size()
                << "): " << (double )(request_idx + 1) / rel_vals.size() * 100 << "%, "
                << "can meet the request precision: "
                << request_precision[index - 1]
                << ", in the range of [" << lower << ", " << upper << "]" << std::endl;
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

    cv::Mat valid_mask = (gt > min_disparity()) & (pred > min_disparity()) &
                         (gt <= max_disparity()) & (pred <= max_disparity());
    cv::Mat far_mask = (gt_depth > depth_threshold) & valid_mask;

    int far_count = cv::countNonZero(far_mask);
    if (far_count == 0)
      return std::numeric_limits<double>::quiet_NaN();

    cv::Mat infinity_mask = (pred_depth <= pred_threshold) & far_mask;
    double ratio = cv::countNonZero(infinity_mask) / static_cast<double>(far_count);
    return ratio;
  }

 private:
  static inline double min_disparity_ = 1e-3;
  static inline double max_disparity_ = 192.0;
};


#endif //HOBOT_STEREONET_INCLUDE_METRICS_PROCESS_H_
