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
#include <arm_neon.h>
#include <dirent.h>
#include <rclcpp_components/register_node_macro.hpp>
#include "stereonet_component.h"
#include "pcl_filter.h"

namespace stereonet {

void StereoNetNode::save_mat_to_bin(const cv::Mat &mat, const std::string &filename) {
  if (mat.empty()) {
    RCLCPP_ERROR(this->get_logger(), "=> The input matrix is empty!");
    return;
  }

  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "=> Failed to open file for writing!");
    return;
  }

  const uchar *dataPtr = mat.data;
  size_t dataSize = mat.total() * mat.elemSize();
  // RCLCPP_INFO(this->get_logger(), "=> dataSize: %ld, mat.total(): %ld, mat.elemSize(): %ld", dataSize, mat.total(), mat.elemSize());

  ofs.write(reinterpret_cast<const char *>(dataPtr), dataSize);

  ofs.close();
}

int StereoNetNode::inference(inference_data_t &inference_data,
                             std::vector<float> &points) {
  bool is_nv12;
  // cv::Mat resized_left_img, resized_right_img;
  cv::Mat &left_img = inference_data.left_sub_img.image;
  cv::Mat &right_img = inference_data.right_sub_img.image;
  is_nv12 = inference_data.left_sub_img.image_type == sub_image_type::NV12;
  if (is_nv12) {
    if (left_img.rows * 2 / 3 != model_input_h_ || left_img.cols != model_input_w_
        || right_img.rows * 2 / 3 != model_input_h_ || right_img.cols != model_input_w_) {
      RCLCPP_FATAL(this->get_logger(), "when encoding of image is nv12, "
                                       "the size(%d, %d) of image MUST equal to size(%d, %d) of model",
                   left_img.cols, left_img.rows,
                   model_input_w_, model_input_h_);
      return -1;
    }
    //  resized_left_img = left_img;
    //  resized_right_img = right_img;
  } else {
    if (left_img.rows != model_input_h_ || left_img.cols != model_input_w_) {
      // RCLCPP_INFO(this->get_logger(), "\033[31m=> resize img [%d, %d] to [%d, %d]\033[0m", left_img.cols, left_img.rows, model_input_w_, model_input_h_);
      cv::resize(left_img, left_img, cv::Size(model_input_w_, model_input_h_));
      cv::resize(right_img, right_img, cv::Size(model_input_w_, model_input_h_));
    } else {
      //  resized_left_img = left_img;
      //  resized_right_img = right_img;
    }
  }

  /*
  inference_data.left_sub_img.origin_height = left_img.rows;
  inference_data.left_sub_img.origin_width = left_img.cols;
  inference_data.right_sub_img.origin_height = right_img.rows;
  inference_data.right_sub_img.origin_width = right_img.cols;
 */
  return stereonet_process_->stereonet_inference(left_img, right_img,
                                                 is_nv12, points);
}

int StereoNetNode::pub_depth_image(pub_data_t &pub_raw_data) {
  cv_bridge::CvImage img_bridge;
  sensor_msgs::msg::Image depth_img_msg;
  const cv::Mat &depth_img = pub_raw_data.model_depth_img;

  if (depth_image_pub_->get_subscription_count() > 0) {
    img_bridge = cv_bridge::CvImage(pub_raw_data.left_sub_img.header,
                                    "mono16", depth_img);
    img_bridge.toImageMsg(depth_img_msg);
    depth_img_msg.header.frame_id = "camera_depth_frame";
    depth_image_pub_->publish(depth_img_msg);
  }

  if (depthcompressed_image_pub_->get_subscription_count() > 0) {
    sensor_msgs::msg::CompressedImage depth_compressed_img_msg;
    depth_compressed_img_msg.format = "compressedDepth";
    depth_compressed_img_msg.header = pub_raw_data.left_sub_img.header;
    std::vector<uchar> compressed_png;
    cv::imencode(".png", depth_img, compressed_png);
    depth_compressed_img_msg.data.reserve(compressed_png.size());
    depth_compressed_img_msg.data.insert(depth_compressed_img_msg.data.end(),
                                         compressed_png.begin(), compressed_png.end());
    depth_compressed_img_msg.header.frame_id = "camera_depth_frame";
    depthcompressed_image_pub_->publish(depth_compressed_img_msg);
  }

  return 0;
}

float compute_percentile(const std::vector<float> &sorted_arr, const int &positive_idx, const float &percentile) {
  int idx = positive_idx + static_cast<int>(percentile * (sorted_arr.size() - positive_idx));
  return sorted_arr[idx];
}

int compute_percentile(const std::vector<int> &sorted_arr, const int &positive_idx, const float &percentile) {
  int idx = positive_idx + static_cast<int>(percentile * (sorted_arr.size() - positive_idx));
  return sorted_arr[idx];
}

int custom_normalize(const cv::Mat &input, cv::Mat &output, const float &min_val, const float &max_val,
                     const float &percentile1, const float &percentile2, const float &percentile3) {
  input.forEach<float>([&output, &min_val, &max_val, &percentile1, &percentile2, &percentile3](float &pixel,
                                                                                               const int *position) -> void {
    uint8_t normalized_val = 0;
    if (pixel <= 0) {
      normalized_val = 0;
    } else if (pixel <= percentile1) {
      normalized_val = ((pixel - min_val) / (percentile1 - min_val) * 0.25) * 255;
    } else if (pixel <= percentile2) {
      normalized_val = (0.25 + (pixel - percentile1) / (percentile2 - percentile1) * 0.25) * 255;
    } else if (pixel <= percentile3) {
      normalized_val = (0.5 + (pixel - percentile2) / (percentile3 - percentile2) * 0.25) * 255;
    } else {
      normalized_val = (0.75 + (pixel - percentile3) / (max_val - percentile3) * 0.25) * 255;
    }

    output.at<uint8_t>(position[0], position[1]) = normalized_val;
  });

  return 0;
}

int custom_normalize(const cv::Mat &input, cv::Mat &output, const int &min_val, const int &max_val,
                     const int &percentile1, const int &percentile2, const int &percentile3) {
  input.forEach<uint16_t>([&output, &min_val, &max_val, &percentile1, &percentile2, &percentile3](uint16_t &pixel,
                                                                                                  const int *position) -> void {
    float pixel_float = static_cast<float>(pixel);
    uint8_t normalized_val = 0;
    if (pixel_float <= 0) {
      normalized_val = 0;
    } else if (pixel_float <= percentile1) {
      normalized_val = ((pixel_float - min_val) / (percentile1 - min_val) * 0.25) * 255;
    } else if (pixel_float <= percentile2) {
      normalized_val = (0.25 + (pixel_float - percentile1) / (percentile2 - percentile1) * 0.25) * 255;
    } else if (pixel_float <= percentile3) {
      normalized_val = (0.5 + (pixel_float - percentile2) / (percentile3 - percentile2) * 0.25) * 255;
    } else {
      normalized_val = (0.75 + (pixel_float - percentile3) / (max_val - percentile3) * 0.25) * 255;
    }

    output.at<uint8_t>(position[0], position[1]) = normalized_val;
  });

  return 0;
}

/*
void mark_zero_positions_disp(const cv::Mat& mat1, cv::Mat& mat2) {
  for (int i = 0; i < mat1.rows; ++i) {
    for (int j = 0; j < mat1.cols; ++j) {
      if (mat1.at<float>(i, j) <= 2) {
          mat2.at<cv::Vec3b>(i + mat2.rows / 2, j) = cv::Vec3b(0, 0, 0);
        }
      }
  }
}
*/

/*
void mark_zero_positions_depth(const cv::Mat& mat1, cv::Mat& mat2, const uint16_t &render_max_depth) {
  for (int i = 0; i < mat1.rows; ++i) {
    for (int j = 0; j < mat1.cols; ++j) {
      if (mat1.at<uint16_t>(i, j) <= 0 || mat1.at<uint16_t>(i, j) > render_max_depth) {
          mat2.at<cv::Vec3b>(i + mat2.rows / 2, j) = cv::Vec3b(0, 0, 0);
        }
      }
  }
}
*/

void mark_zero_positions(const cv::Mat &disp, const cv::Mat &depth, cv::Mat &visual, const uint16_t &render_max_depth) {
  for (int i = 0; i < disp.rows; ++i) {
    for (int j = 0; j < disp.cols; ++j) {
      if (disp.at<float>(i, j) <= 0 || depth.at<uint16_t>(i, j) > render_max_depth) {
        visual.at<cv::Vec3b>(i + visual.rows / 2, j) = cv::Vec3b(0, 0, 0);
      }
    }
  }
}

int StereoNetNode::pub_visual_image(pub_data_t &pub_raw_data) {
  cv_bridge::CvImage img_bridge;
  sensor_msgs::msg::Image visual_img_msg;
  const cv::Mat &image = pub_raw_data.left_sub_img.image;
  const std::vector<float> &points = pub_raw_data.points;
  cv::Mat &depth_img = pub_raw_data.model_depth_img;
  cv::Mat bgr_image;
  if (visual_image_pub_->get_subscription_count() < 1 && !save_image_all_) return 0;
  if (pub_raw_data.left_sub_img.bgr.empty()) {
    image_conversion::nv12_to_bgr(pub_raw_data.left_sub_img.image,
                                  pub_raw_data.left_sub_img.bgr);
  }
  bgr_image = pub_raw_data.left_sub_img.bgr;
  double font_scale = 0.5 * bgr_image.cols / 640;
  int step_num = 6;
  int x_step = bgr_image.cols / step_num;
  int y_step = bgr_image.rows / step_num;
  int start = 1;

  cv::Mat visual_img(bgr_image.rows * 2, bgr_image.cols, CV_8UC3);
  bgr_image.copyTo(visual_img(cv::Rect(0, 0, bgr_image.cols, bgr_image.rows)));
  if (render_perf_) {
    std::stringstream perf_text;
    perf_text << "FPS: " << pub_raw_data.fps;
    cv::putText(visual_img, perf_text.str(), cv::Point(10, 18),
                cv::FONT_HERSHEY_SIMPLEX, font_scale * 1.3,
                CV_RGB(0, 0, 255), 2);
    perf_text.str("");
    perf_text.clear();
    perf_text << "Latency: " << pub_raw_data.latency << "ms";
    cv::putText(visual_img, perf_text.str(), cv::Point(10, 40),
                cv::FONT_HERSHEY_SIMPLEX, font_scale * 1.3,
                CV_RGB(0, 0, 255), 2);

    perf_text.str("");
    perf_text.clear();
    perf_text << "CPU: " << pub_raw_data.cpu_usage << "%";
    cv::putText(visual_img, perf_text.str(), cv::Point(10 + x_step * 2, 18),
                cv::FONT_HERSHEY_SIMPLEX, font_scale * 1.3,
                CV_RGB(0, 0, 255), 2);
    perf_text.str("");
    perf_text.clear();
    perf_text << "BPU: " << pub_raw_data.bpu_usage << "%";
    cv::putText(visual_img, perf_text.str(), cv::Point(10 + x_step * 2, 40),
                cv::FONT_HERSHEY_SIMPLEX, font_scale * 1.3,
                CV_RGB(0, 0, 255), 2);
  }
  cv::Mat disp_mat(bgr_image.rows, bgr_image.cols, CV_32FC1, const_cast<float *>(points.data()));
  cv::Mat feat_visual;
  if (render_type_ == 0) {
    disp_mat.convertTo(feat_visual, CV_8UC1, visual_alpha_, visual_beta_);
  } else if (render_type_ == 1) {
    // calc percentile
    std::vector<float> disp_vals(points.size());
    std::copy(points.begin(), points.end(), disp_vals.begin());
    std::sort(disp_vals.begin(), disp_vals.end());
    int positive_idx = 0;
    for (int i = 0; i < disp_vals.size(); i++) {
      if (disp_vals[i] > 0) {
        positive_idx = i;
        break;
      }
    }
    float percentile1 = compute_percentile(disp_vals, positive_idx, 0.1);
    float percentile2 = compute_percentile(disp_vals, positive_idx, 0.5);
    float percentile3 = compute_percentile(disp_vals, positive_idx, 0.9);

    feat_visual = cv::Mat::zeros(disp_mat.size(), CV_8UC1);
    // norm
    custom_normalize(disp_mat,
                     feat_visual,
                     disp_vals[positive_idx],
                     disp_vals[disp_vals.size() - 1],
                     percentile1,
                     percentile2,
                     percentile3);
  } else {
    // calc percentile
    std::vector<int> depth_vals;
    depth_vals.reserve(depth_img.rows * depth_img.cols);
    depth_img.forEach<uint16_t>([&depth_vals](uint16_t &pixel, const int *position) {
      depth_vals.push_back(pixel);
    });
    std::sort(depth_vals.begin(), depth_vals.end());
    int positive_idx = 0;
    for (int i = 0; i < depth_vals.size(); i++) {
      if (depth_vals[i] > 0) {
        positive_idx = i;
        break;
      }
    }
    int percentile1 = compute_percentile(depth_vals, positive_idx, 0.1);
    int percentile2 = compute_percentile(depth_vals, positive_idx, 0.5);
    int percentile3 = compute_percentile(depth_vals, positive_idx, 0.9);

    feat_visual = cv::Mat::zeros(depth_img.size(), CV_8UC1);
    // norm
    custom_normalize(depth_img,
                     feat_visual,
                     depth_vals[positive_idx],
                     depth_vals[depth_vals.size() - 1],
                     percentile1,
                     percentile2,
                     percentile3);
  }

  //  cv::convertScaleAbs(feat_visual, feat_visual, 2);
  cv::applyColorMap(feat_visual,
                    visual_img(cv::Rect(0, bgr_image.rows, bgr_image.cols, bgr_image.rows)),
                    cv::COLORMAP_JET);

  if (render_need_filter_ && render_type_ > 0) {
    // speckle filter
    cv::Mat disp_norm;
    cv::normalize(disp_mat, disp_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::filterSpeckles(disp_norm, 0, 10, 3);
    cv::Mat mask_disp;
    cv::threshold(disp_norm, mask_disp, 0, 1, cv::THRESH_BINARY);
    mask_disp.convertTo(mask_disp, CV_32FC1);
    cv::Mat disp_mat_filter = disp_mat.mul(mask_disp);
    mark_zero_positions(disp_mat_filter, depth_img, visual_img, render_max_depth_);

    if (depth_need_filter_) {
      // cv::Mat depth_norm;
      // cv::normalize(depth_img, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
      // cv::filterSpeckles(depth_norm, 0, 3000, 2);
      // cv::Mat mask_dpth;
      // cv::threshold(depth_norm, mask_dpth, 0, 1, cv::THRESH_BINARY);
      // mask_dpth.convertTo(mask_dpth, CV_16UC1);
      mask_disp.convertTo(mask_disp, CV_16UC1);
      depth_img = depth_img.mul(mask_disp);
      // depth_img = depth_img.mul(mask_dpth);
    }
  }

  /*
  if (render_type_ == 1) {
    // speckle filter
    cv::Mat disp_norm;
    cv::normalize(disp_mat, disp_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::filterSpeckles(disp_norm, 0, 10, 3);
    cv::Mat mask;
    cv::threshold(disp_norm, mask, 0, 1, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_32FC1);
    cv::Mat disp_mat_filter = disp_mat.mul(mask);
    mark_zero_positions_disp(disp_mat_filter, visual_img);
  }
  if (render_type_ > 1)
  {
    // speckle filter
    cv::Mat depth_norm;
    cv::normalize(depth_img, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::filterSpeckles(depth_norm, 0, 10, 3);
    cv::Mat mask;
    cv::threshold(depth_norm, mask, 0, 1, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_16UC1);
    cv::Mat depth_img_filter = depth_img.mul(mask);
    mark_zero_positions_depth(depth_img_filter, visual_img);
  }
  */


  // RCLCPP_WARN_ONCE(this->get_logger(), "=> x_step: %d, y_step: %d", x_step, y_step);
  if (!depth_type_point_) {
    start = 0;
  }
  for (int i = start; i < step_num; i++) {
    for (int j = start; j < step_num; j++) {
      // 横线
      cv::line(visual_img, cv::Point2i(0, bgr_image.rows + i * y_step),
               cv::Point2i(bgr_image.cols, bgr_image.rows + i * y_step),
               cv::Scalar(255, 255, 255), 1);
      // 竖线
      cv::line(visual_img, cv::Point2i(j * x_step, bgr_image.rows),
               cv::Point2i(j * x_step, bgr_image.rows * 2),
               cv::Scalar(255, 255, 255), 1);

      // 横线
      cv::line(visual_img, cv::Point2i(0, i * y_step),
               cv::Point2i(bgr_image.cols, i * y_step),
               cv::Scalar(255, 255, 255), 1);
      // 竖线
      cv::line(visual_img, cv::Point2i(j * x_step, 0),
               cv::Point2i(j * x_step, bgr_image.rows),
               cv::Scalar(255, 255, 255), 1);
      // 取出Z值
      double distance;
      cv::Point2i bgr_location, depth_location;
      if (!depth_type_point_) {
        auto z_region = cv::mean(depth_img(
            cv::Rect(j * x_step, i * y_step, x_step, y_step)));
        distance = static_cast<double>(z_region[0]) / 1000.0;
        depth_location = cv::Point2i(j * x_step + 3 + x_step / 2,
                                     bgr_image.rows + i * y_step - 3 + y_step / 2);
        bgr_location = cv::Point2i(j * x_step + 3 + x_step / 2,
                                   i * y_step - 3 + y_step / 2);
      } else {
        uint16_t Z = depth_img.at<uint16_t>(i * y_step, j * x_step);
        distance = static_cast<double>(Z) / 1000.0;
        depth_location = cv::Point2i(j * x_step + 3,
                                     bgr_image.rows + i * y_step - 3);
        bgr_location = cv::Point2i(j * x_step + 3, i * y_step - 3);
      }

      std::ostringstream ss;
      ss << std::fixed << std::setprecision(3) << distance << "m";
      cv::putText(visual_img, ss.str(), bgr_location,
                  cv::FONT_HERSHEY_SIMPLEX, font_scale,
                  cv::Scalar(255, 255, 255), 2);

      cv::putText(visual_img, ss.str(), depth_location,
                  cv::FONT_HERSHEY_SIMPLEX, font_scale,
                  cv::Scalar(255, 255, 255), 2);
    }
  }

  if (depth_compare) {
    cv::putText(visual_img, "camera", cv::Point2i(3, 38),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 2);

    cv::putText(visual_img, "AI-depth", cv::Point2i(3, bgr_image.rows + 38),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 2);
    {
      std::lock_guard<std::mutex> lck(compare_visual_mtx_);

      cv::hconcat(visual_img, compare_visual_, visual_img);
    }
  }

  img_bridge = cv_bridge::CvImage(pub_raw_data.left_sub_img.header,
                                  "bgr8", visual_img);
  img_bridge.toImageMsg(visual_img_msg);
  visual_image_pub_->publish(visual_img_msg);
//  static uint32_t i = 0;
//  std::ofstream result(std::to_string(i) + ".txt", std::ios::out);
//  cv::imwrite(std::to_string(i) + ".jpg", visual_img);
//  i++;
//  result << pub_raw_data.depth_img;

  if (save_image_all_) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (save_cnt_ == 0) {
      // create save dir
      if (!fs::exists(save_dir_)) {
        if (!fs::create_directory(save_dir_)) {
          RCLCPP_ERROR_STREAM(this->get_logger(), "\033[31m=> failed to create save dir: " << save_dir_ << "\033[0m");
          save_image_all_ = false;
        } else {
          RCLCPP_INFO_STREAM(this->get_logger(), "\033[31m=> create save dir: " << save_dir_ << "\033[0m");
        }
      }
      // else
      // {
      //   bool use_local_image;
      //   this->get_parameter("use_local_image", use_local_image);
      //   if (!use_local_image) {
      //     RCLCPP_ERROR_STREAM(this->get_logger(), "\033[31m=> save dir: " << save_dir_ << " already exists, the image will not be saved\033[0m");
      //     save_image_all_ = false;
      //     return 0;
      //   }
      // }
      // save calib param
      std::stringstream ss;
      ss << "[fx, fy, cx, cy, baseline] = [" << camera_fx << ", " << camera_fy << ", " << camera_cx << ", " << camera_cy
         << ", " << base_line * 1000 << "]" << std::endl;
      std::string result = ss.str();
      std::ofstream outFile(save_dir_ + "/calib_param.txt");
      if (outFile.is_open()) {
        outFile << result;
        outFile.close();
        RCLCPP_WARN_STREAM(this->get_logger(),
                           "\033[31m=> calib param save to: " << save_dir_ << "/calib_param.txt\033[0m");
      }
    }

    // save images
    if (save_cnt_ % save_freq_ == 0) {
      const cv::Mat &left_image = pub_raw_data.left_sub_img.image;
      const cv::Mat &right_image = pub_raw_data.right_sub_img.image;
      cv::Mat left_img_bgr;
      cv::Mat right_img_bgr;
      if (pub_raw_data.left_sub_img.image_type == sub_image_type::NV12) {
        cv::cvtColor(left_image, left_img_bgr, cv::COLOR_YUV2BGR_NV12);
        cv::cvtColor(right_image, right_img_bgr, cv::COLOR_YUV2BGR_NV12);
      } else {
        left_img_bgr = left_image;
        right_img_bgr = right_image;
      }
      save_images_with_nv12(left_img_bgr, right_img_bgr, image_format_);

      // cv::Mat disp_img = cv::Mat(left_img_bgr.rows, left_img_bgr.cols, CV_32FC1);
      // std::memcpy(disp_img.data, points.data(), points.size() * sizeof(float));
      std::stringstream ss;
      ss << std::setw(6) << std::setfill('0') << save_cnt_;
      std::string disp_path = save_dir_ + "/disp" + ss.str() + ".pfm";
      std::string depth_path = save_dir_ + "/depth" + ss.str() + ".png";
      std::string visual_path = save_dir_ + "/visual" + ss.str() + ".png";
      cv::imwrite(disp_path, disp_mat);
      cv::imwrite(depth_path, depth_img);
      cv::imwrite(visual_path, visual_img);
      RCLCPP_WARN_STREAM(this->get_logger(), "\033[31m=> save infer result to: " << disp_path << "\033[0m");

      if (save_total_ != -1 && (save_cnt_ / save_freq_ + 1) >= save_total_) {
        save_image_all_ = false;
        RCLCPP_WARN_STREAM(this->get_logger(),
                           "\033[31m=> save total: " << save_total_ << " images, stop saving\033[0m");
      }
    }

    // save cnt ++
    save_cnt_++;
  }

  return 0;
}

int StereoNetNode::pub_rectified_image(pub_data_t &pub_raw_data) {
  if (rectified_image_pub_->get_subscription_count() > 0) {
    sensor_msgs::msg::Image pub_img_msg;
    const cv::Mat &image = pub_raw_data.left_sub_img.image;
    const uint8_t *nv12_data_ptr = nullptr;
    cv::Mat nv12_image;

    pub_img_msg.header = pub_raw_data.left_sub_img.header;

    if (pub_rectified_bgr_) {
      if (pub_raw_data.left_sub_img.bgr.empty()) {
        image_conversion::nv12_to_bgr(pub_raw_data.left_sub_img.image,
                                      pub_raw_data.left_sub_img.bgr);
      }
      const cv::Mat &bgr_image = pub_raw_data.left_sub_img.bgr;
      pub_img_msg.encoding = "bgr8";
      pub_img_msg.height = bgr_image.rows;
      pub_img_msg.width = bgr_image.cols;
      pub_img_msg.step = pub_img_msg.width * 3;
      size_t data_len = pub_img_msg.width * pub_img_msg.height * 3;
      pub_img_msg.data.resize(data_len);
      memcpy(pub_img_msg.data.data(), bgr_image.data, data_len);
    } else {
      pub_img_msg.encoding = "nv12";
      pub_img_msg.step = pub_img_msg.width;
      if (pub_raw_data.left_sub_img.image_type == sub_image_type::NV12) {
        pub_img_msg.height = image.rows / 3 * 2;
        pub_img_msg.width = image.cols;
        nv12_data_ptr = image.ptr<uint8_t>();
      } else {
        pub_img_msg.height = image.rows;
        pub_img_msg.width = image.cols;
        nv12_image = cv::Mat(pub_img_msg.height * 3 / 2, pub_img_msg.width, CV_8UC1);
        image_conversion::bgr24_to_nv12_neon(image.data, nv12_image.data, pub_img_msg.width, pub_img_msg.height);
        nv12_data_ptr = nv12_image.ptr<uint8_t>();
      }
      size_t data_len = pub_img_msg.width * pub_img_msg.height * 3 / 2;
      pub_img_msg.data.resize(data_len);
      memcpy(pub_img_msg.data.data(), nv12_data_ptr, data_len);
    }
    rectified_image_pub_->publish(pub_img_msg);
  }

  if (rectified_right_image_pub_->get_subscription_count() > 0) {
    sensor_msgs::msg::Image pub_img_msg;
    const cv::Mat &image = pub_raw_data.right_sub_img.image;
    const uint8_t *nv12_data_ptr = nullptr;
    cv::Mat nv12_image;

    pub_img_msg.header = pub_raw_data.right_sub_img.header;

    if (pub_rectified_bgr_) {
      if (pub_raw_data.right_sub_img.bgr.empty()) {
        image_conversion::nv12_to_bgr(pub_raw_data.right_sub_img.image,
                                      pub_raw_data.right_sub_img.bgr);
      }
      const cv::Mat &bgr_image = pub_raw_data.right_sub_img.bgr;
      pub_img_msg.encoding = "bgr8";
      pub_img_msg.height = bgr_image.rows;
      pub_img_msg.width = bgr_image.cols;
      pub_img_msg.step = pub_img_msg.width * 3;
      size_t data_len = pub_img_msg.width * pub_img_msg.height * 3;
      pub_img_msg.data.resize(data_len);
      memcpy(pub_img_msg.data.data(), bgr_image.data, data_len);
    } else {
      pub_img_msg.encoding = "nv12";
      pub_img_msg.step = pub_img_msg.width;
      if (pub_raw_data.right_sub_img.image_type == sub_image_type::NV12) {
        pub_img_msg.height = image.rows / 3 * 2;
        pub_img_msg.width = image.cols;
        nv12_data_ptr = image.ptr<uint8_t>();
      } else {
        pub_img_msg.height = image.rows;
        pub_img_msg.width = image.cols;
        nv12_image = cv::Mat(pub_img_msg.height * 3 / 2, pub_img_msg.width, CV_8UC1);
        image_conversion::bgr24_to_nv12_neon(image.data, nv12_image.data, pub_img_msg.width, pub_img_msg.height);
        nv12_data_ptr = nv12_image.ptr<uint8_t>();
      }
      size_t data_len = pub_img_msg.width * pub_img_msg.height * 3 / 2;
      pub_img_msg.data.resize(data_len);
      memcpy(pub_img_msg.data.data(), nv12_data_ptr, data_len);
    }
    rectified_right_image_pub_->publish(pub_img_msg);
  }
  RCLCPP_WARN_ONCE(this->get_logger(),
                   "pub rectified image with topic name [%s]",
                   rectified_image_topic_.data());
  return 0;
}

int StereoNetNode::pub_pointcloud2(pub_data_t &pub_raw_data) {
  uint32_t point_size = 0;
  const cv::Mat &depth_img = pub_raw_data.model_depth_img;
  uint16_t *depth_ptr = reinterpret_cast<uint16_t *>(depth_img.data);

  if (pointcloud2_pub_->get_subscription_count() < 1) return 0;
  if (pub_raw_data.left_sub_img.bgr.empty()) {
    image_conversion::nv12_to_bgr(pub_raw_data.left_sub_img.image,
                                  pub_raw_data.left_sub_img.bgr);
  }
  const cv::Mat &image = pub_raw_data.left_sub_img.bgr;
  sensor_msgs::msg::PointCloud2 point_cloud_msg;
  sensor_msgs::PointCloud2Modifier modifier(point_cloud_msg);

  point_cloud_msg.header = pub_raw_data.left_sub_img.header;
  point_cloud_msg.header.frame_id = "camera_link";
  point_cloud_msg.is_dense = false;
  point_cloud_msg.fields.resize(4);
  point_cloud_msg.fields[0].name = "x";
  point_cloud_msg.fields[0].offset = 0;
  point_cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  point_cloud_msg.fields[0].count = 1;

  point_cloud_msg.fields[1].name = "y";
  point_cloud_msg.fields[1].offset = 4;
  point_cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  point_cloud_msg.fields[1].count = 1;

  point_cloud_msg.fields[2].name = "z";
  point_cloud_msg.fields[2].offset = 8;
  point_cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  point_cloud_msg.fields[2].count = 1;

  point_cloud_msg.fields[3].name = "rgb";
  point_cloud_msg.fields[3].offset = 12;
  point_cloud_msg.fields[3].datatype = sensor_msgs::msg::PointField::UINT32;
  point_cloud_msg.fields[3].count = 1;

  point_cloud_msg.height = 1;
  point_cloud_msg.point_step = 16;

  //  point_cloud_msg.width = (img_origin_width / 2) * (img_origin_height / 2);
  //  point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width;
  point_cloud_msg.data.resize(
      (depth_w_ / 2) * (depth_h_ / 2) * point_cloud_msg.point_step *
          point_cloud_msg.height);

  float *pcd_data_ptr = reinterpret_cast<float *>(point_cloud_msg.data.data());
  float fy;
  for (int y = 0; y < depth_h_; y += 2) {
    fy = (camera_cy - y) / camera_fy;
    for (int x = 0; x < depth_w_; x += 2) {
      float depth = depth_ptr[y * depth_w_ + x] / 1000.0f;
      if (depth > pc_max_depth_) continue;
      //if (depth < height_min_ || depth > height_max_) continue;
      float X = (camera_cx - x) / camera_fx * depth;
      float Y = fy * depth;
      if (Y < height_min_ || Y > height_max_) {
        continue;
      }
      *pcd_data_ptr++ = depth;
      *pcd_data_ptr++ = X;
      *pcd_data_ptr++ = Y;
      cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
      *(uint32_t *) pcd_data_ptr++ = (pixel[2] << 16) | (pixel[1] << 8) | (pixel[0] << 0);
      point_size++;
    }
  }
  point_cloud_msg.width = point_size;
  point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width;
  point_cloud_msg.data.resize(point_size * point_cloud_msg.point_step *
      point_cloud_msg.height);

  if (need_pcl_filter_) {
    ScopeProcessTime t("pcl_filter");
    pcl_filter::applyfilter(point_cloud_msg,
                            leaf_size_, KMean_, stdv_);
  }

//  float32x4_t fx_vec = vdupq_n_f32(1 / camera_fx);
//  float32x4_t fy_vec = vdupq_n_f32(1 / camera_fy);
//  float32x4_t cx_vec = vdupq_n_f32(camera_cx);
//  float32x4_t cy_vec = vdupq_n_f32(camera_cy);
//  float32x4_t v1000 = vdupq_n_f32(0.001);
//  for (uint32_t y = 0; y < img_origin_height; y += 2) {
//    float32x4_t y_f32 = vdupq_n_f32(static_cast<float>(y));
//    for (uint32_t x = 0; x < img_origin_width; x += 8) {
//      uint32_t idx = y * img_origin_width + x;
//      uint32_t xx[4] = {x, x + 2, x + 4, x + 6};
//      uint16x4x2_t d = vld2_u16(&depth_ptr[idx]);
//      float32x4_t depth_f32 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(d.val[0])), v1000);
//      float32x4_t x_f32 = vcvtq_f32_u32(vld1q_u32(xx));
//      x_f32 = vmulq_f32(vsubq_f32(x_f32, cx_vec), fx_vec);
//      y_f32 = vmulq_f32(vsubq_f32(y_f32, cy_vec), fy_vec);
//      float32x4x3_t pts = {vmulq_f32(x_f32, depth_f32),
//                           vmulq_f32(y_f32, depth_f32),
//                           depth_f32};
//      vst3q_f32(pcd_data_ptr, pts);
//      pcd_data_ptr += 12;
//    }
//  }
  {
    ScopeProcessTime t("pcd publisher");
    pointcloud2_pub_->publish(point_cloud_msg);
  }
  return 0;
}

int StereoNetNode::pub_depth_camera_info(pub_data_t &pub_raw_data) {
  sensor_msgs::msg::CameraInfo depth_camera_info;
  if (depth_camera_info_pub_->get_subscription_count() < 1) return 0;
  depth_camera_info.height = pub_raw_data.model_depth_img.rows;
  depth_camera_info.width = pub_raw_data.model_depth_img.cols;
  depth_camera_info.header = pub_raw_data.left_sub_img.header;

  depth_camera_info.distortion_model = "plumb_bob";
  depth_camera_info.d.resize(5, 0.f);

  depth_camera_info.k = {
      camera_fx, 0.f, camera_cx,
      0.f, camera_fy, camera_cy,
      0.f, 0.f, 1.f
  };

  depth_camera_info.r = {
      1.f, 0.f, 0.f,
      0.f, 1.f, 0.f,
      0.f, 0.f, 1.f
  };

  depth_camera_info.p = {
      camera_fx, 0.f, camera_cx, 0.f,
      0.f, camera_fy, camera_cy, 0.f,
      0.f, 0.f, 1.f, 0.f
  };
  depth_camera_info_pub_->publish(depth_camera_info);
  return 0;
}

void dump_rectified_image(cv::Mat &left_img, cv::Mat &right_img,
                          cv::Mat &rectified_left_img, cv::Mat &rectified_right_img) {
  std::stringstream iss;
  static std::atomic_int iii{0};
  cv::Mat img_src, img_rtf;
  cv::hconcat(left_img, right_img, img_src);
  cv::hconcat(rectified_left_img, rectified_right_img, img_rtf);
  for (int i = 0; i < 10; ++i) {
    cv::Point a, b;
    a.x = 0;
    a.y = img_rtf.rows / 10 * i;
    b.x = img_rtf.cols;
    b.y = img_rtf.rows / 10 * i;
    cv::line(img_rtf, a, b, cv::Scalar(0, 255, 0), 2);
  }
  iss << std::setw(6) << std::setfill('0') << iii++;
  auto image_seq = iss.str();
  cv::imwrite("./230ai_data/left" + image_seq + "_rectify.png", rectified_left_img);
  cv::imwrite("./230ai_data/right" + image_seq + "_rectify.png", rectified_right_img);
  cv::imwrite("./before.jpg", img_src);
  cv::imwrite("./after.jpg", img_rtf);
}

void StereoNetNode::save_images_with_nv12(cv::Mat &left_img, cv::Mat &right_img, const std::string &image_format) {
  std::stringstream iss;
  iss << std::setw(6) << std::setfill('0') << save_cnt_;
  auto image_seq = iss.str();
  cv::imwrite(save_dir_ + "/left" + image_seq + "." + image_format, left_img);
  cv::imwrite(save_dir_ + "/right" + image_seq + "." + image_format, right_img);

  if (save_image_to_nv12_) {
    cv::Mat left_img_nv12, right_img_nv12;
    image_conversion::bgr_to_nv12(left_img, left_img_nv12);
    image_conversion::bgr_to_nv12(right_img, right_img_nv12);
    save_mat_to_bin(left_img_nv12, save_dir_ + "/left" + image_seq + ".nv12");
    save_mat_to_bin(right_img_nv12, save_dir_ + "/right" + image_seq + ".nv12");
  }
}

void save_images(cv::Mat &left_img, cv::Mat &right_img, uint64_t ts,
                 const std::string &image_format) {
  static std::atomic_bool directory_created{false};
  static std::atomic_int i{0};
  std::stringstream iss;
  cv::Mat image_combine;
  if (!directory_created) {
    directory_created = true;
    system("mkdir -p"
           " ./images/cam0/data/"
           " ./images/cam1/data/"
           " ./images/cam_combine/data/");
  }
  iss << std::setw(3) << std::setfill('0') << i++;
  auto image_seq = iss.str();
  cv::imwrite("./images/cam0/data/" + std::to_string(ts) + "." + image_format, left_img);
  cv::imwrite("./images/cam1/data/" + std::to_string(ts) + "." + image_format, right_img);
  //cv::vconcat(left_img, right_img, image_combine);
  //cv::imwrite("./images/cam_combine/data/combine_" + image_seq + "." + image_format, image_combine);
}

void StereoNetNode::stereo_image_cb(const sensor_msgs::msg::Image::SharedPtr img) {
  cv::Mat stereo_img, left_img, right_img;
  sub_image left_sub_img, right_sub_img;
  const std::string &encoding = img->encoding;
  int stereo_img_width, stereo_img_height;
  builtin_interfaces::msg::Time now = this->get_clock()->now();
  RCLCPP_DEBUG(this->get_logger(),
               "we have received stereo msg at: %d.%09d,\n"
               "timestamp of stereo is: %d.%09d, latency is %f sec,\n"
               "encoding: %s, width: %d, height: %d",
               now.sec, now.nanosec,
               img->header.stamp.sec, img->header.stamp.nanosec,
               (rclcpp::Time(now) - rclcpp::Time(img->header.stamp)).seconds(),
               encoding.c_str(), img->width, img->height);
  {
    auto pub_data = std::make_shared<pub_data_t>();
    pub_data->ts = img->header.stamp.sec * 1e9 + img->header.stamp.nanosec;
    pub_data->is_dummy = true;
    pub_que_.put_silence(pub_data->ts, pub_data);
  }

  if (stereo_combine_mode_ == 0) {
    stereo_img_width = img->width / 2;
    stereo_img_height = img->height;
  } else if (stereo_combine_mode_ == 1) {
    stereo_img_width = img->width;
    stereo_img_height = img->height / 2;
  }

  if (encoding == "nv12" || encoding == "NV12") {
    if (need_rectify_ || save_image_ || save_image_all_) {
      ScopeProcessTime t("nv12->bgr");
      stereo_img = cv::Mat(img->height, img->width, CV_8UC3);
      image_conversion::nv12_to_bgr24_neon(img->data.data(), stereo_img.data, img->width, img->height);
      left_sub_img.image_type = sub_image_type::BGR;
      right_sub_img.image_type = sub_image_type::BGR;
    } else {
      left_sub_img.image_type = sub_image_type::NV12;
      right_sub_img.image_type = sub_image_type::NV12;
    }
  } else if (encoding == "bgr8" || encoding == "BGR8") {
    ScopeProcessTime t("cv_bridge::toCvShare");
    stereo_img = cv_bridge::toCvShare(img)->image;
    left_sub_img.image_type = sub_image_type::BGR;
    right_sub_img.image_type = sub_image_type::BGR;
  }

  if (left_sub_img.image_type == sub_image_type::BGR) {
    if (stereo_combine_mode_ == 0) {
      left_img = stereo_img(
          cv::Rect(0, 0, stereo_img_width, stereo_img_height));
      right_img = stereo_img(
          cv::Rect(stereo_img_width, 0, stereo_img_width, stereo_img_height));
    } else if (stereo_combine_mode_ == 1) {
      left_img = stereo_img(
          cv::Rect(0, 0, stereo_img_width, stereo_img_height));
      right_img = stereo_img(
          cv::Rect(0, stereo_img_height, stereo_img_width, stereo_img_height));
    }
    left_sub_img.image = left_img.clone();
    right_sub_img.image = right_img.clone();
    left_sub_img.bgr = left_sub_img.image;
    right_sub_img.bgr = right_sub_img.image;
  } else if (left_sub_img.image_type == sub_image_type::NV12) {
    if (stereo_combine_mode_ == 0) {
      RCLCPP_FATAL(this->get_logger(), "Horizontal stitching is unsupported for NV12-encoded images."
                                       " Only Vertical stitching is supported.");
      return;
    } else if (stereo_combine_mode_ == 1) {
      uint y_size = stereo_img_height * stereo_img_width;
      uint uv_size = y_size >> 1;

      left_sub_img.image = cv::Mat(stereo_img_height * 3 / 2, stereo_img_width, CV_8UC1);
      right_sub_img.image = cv::Mat(stereo_img_height * 3 / 2, stereo_img_width, CV_8UC1);

      std::memcpy(left_sub_img.image.data, img->data.data(), y_size);
      std::memcpy(left_sub_img.image.data + y_size,
                  img->data.data() + y_size * 2, uv_size);

      std::memcpy(right_sub_img.image.data, img->data.data() + y_size, y_size);
      std::memcpy(right_sub_img.image.data + y_size,
                  img->data.data() + y_size * 2 + uv_size, uv_size);
    }
  }

  left_sub_img.header = img->header;
  right_sub_img.header = img->header;

  left_sub_img.origin_height = stereo_img_height;
  left_sub_img.origin_width = stereo_img_width;
  right_sub_img.origin_height = stereo_img_height;
  right_sub_img.origin_width = stereo_img_width;

  inference_data_t inference_data{left_sub_img, right_sub_img, false, now};
  int que_size = inference_que_.put(inference_data);
  if (que_size > 2) {
    RCLCPP_WARN_THROTTLE(this->get_logger(),
                         *this->get_clock(), 5000, "inference que is full!");
    inference_que_.pop_front();
  }
}

void StereoNetNode::render_func() {
  while (is_running_ && rclcpp::ok()) {
    std::shared_ptr<pub_data_t> pub_data;
    if (pub_que_.get(pub_data)) {
      if (pub_data->is_dummy) {
        pub_que_.put_silence(pub_data->ts, pub_data);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      } else {
        pub_func(*pub_data);
      }
    }
  }
}

void StereoNetNode::inference_func() {
  int ret = 0;
  cv::Mat rectified_left_image, rectified_right_image;
  while (is_running_ && rclcpp::ok()) {
    inference_data_t inference_data;
    std::vector<float> points;
    if (inference_que_.get(inference_data)) {
      cv::Mat &left_image = inference_data.left_sub_img.image;
      cv::Mat &right_image = inference_data.right_sub_img.image;
      if (need_rectify_) {
        ScopeProcessTime t("stereo_rectify");
        for (auto &s : stereo_rectify_list_) {
          s->Rectify(left_image, right_image, rectified_left_image, rectified_right_image);
          left_image = rectified_left_image;
          right_image = rectified_right_image;
        }
      }

      if (save_image_) {
        save_images(inference_data.left_sub_img.image,
                    inference_data.right_sub_img.image,
                    inference_data.left_sub_img.header.stamp.sec * 1e9
                        + inference_data.left_sub_img.header.stamp.nanosec,
                    image_format_);
      }

      ret = inference(inference_data, points);
      if (ret != 0) {
        RCLCPP_ERROR(this->get_logger(), "inference failed.");
      } else {
        sub_image left_sub_img = inference_data.left_sub_img;
        sub_image right_sub_img = inference_data.right_sub_img;
        left_sub_img.image = inference_data.left_sub_img.image.clone();
        right_sub_img.image = inference_data.right_sub_img.image.clone();
        left_sub_img.bgr = left_sub_img.image;
        right_sub_img.bgr = right_sub_img.image;
        std::vector<float> points_pub = std::move(points);
        cv::Mat depth;
        auto pub_data = std::make_shared<pub_data_t>();
        pub_data->left_sub_img = left_sub_img;
        pub_data->right_sub_img = right_sub_img;
        pub_data->points = points_pub;
        pub_data->depth_img = depth;
        pub_data->ts = left_sub_img.header.stamp.sec * 1e9
            + left_sub_img.header.stamp.nanosec;
        pub_data->is_dummy = false;
        {
          ScopeProcessTime t("convert to depth");
          convert_depth(*pub_data);
        }
        if (render_perf_) {
          int current_latency = (this->now() - inference_data.received_time).seconds() * 1000;
          latency_list_.emplace_back(current_latency);
          if (latency_list_.size() >= 4) {
            current_latency = 0.1 * latency_list_[3] + 0.2 * latency_list_[2] +
                0.3 * latency_list_[1] + 0.4 * latency_list_[0];
            //latency_list_.back() = current_latency;
            latency_list_.pop_front();
          }
          pub_data->latency = current_latency;
          performance_writer::Get()->record_performance(pub_data->latency);
          pub_data->fps = performance_writer::Get()->get_fps();
          pub_data->cpu_usage = performance_writer::Get()->get_cpu_usage();
          pub_data->bpu_usage = performance_writer::Get()->get_bpu_usage();
        }
        if (inference_data.is_local_image) {
          pub_func(*pub_data);
        } else {
          int que_size = pub_que_.size();
          if (que_size > 2) {
            RCLCPP_WARN_THROTTLE(this->get_logger(),
                                 *this->get_clock(), 5000, "pub_que is full!");
            pub_que_.clear();
          }
          pub_que_.put(pub_data->ts, pub_data);
        }
//        dump_one_point_disparity(pub_data,
//            inference_data.right_sub_img.image, 659, 301);
      }
    }
  }
  inference_que_.clear();
}

void StereoNetNode::dump_one_point_disparity(
    pub_data_t &pub_raw_data, const cv::Mat &right_image,
    int x, int y) {
  cv::Mat left_image = pub_raw_data.left_sub_img.image.clone();
  cv::Mat right_image2 = right_image.clone();
  std::vector<float> &points = pub_raw_data.points;
  auto disparity = points[y * left_image.cols + x];
  cv::circle(left_image, cv::Point(x, y), 10,
             cv::Scalar(255, 0, 0), 3);
  cv::circle(right_image2, cv::Point(x - disparity, y), 10,
             cv::Scalar(255, 0, 0), 3);
  RCLCPP_INFO(this->get_logger(), "[x: %d, y: %d, disp: %f]\n", x, y, disparity);
  cv::imwrite("one_point_disparity_left.jpeg", left_image);
  cv::imwrite("one_point_disparity_right.jpeg", right_image2);
}

void StereoNetNode::convert_depth(pub_data_t &pub_raw_data) {
  int img_origin_width, img_origin_height;
  const cv::Mat &image = pub_raw_data.left_sub_img.image;
  std::vector<float> &points = pub_raw_data.points;
  std::vector<float> &image_size_points = pub_raw_data.image_size_points;
  cv::Mat &depth_img = pub_raw_data.depth_img;
  cv::Mat model_depth_img;
  std::vector<float> resized_points;
  img_origin_width = pub_raw_data.left_sub_img.origin_width;
  img_origin_height = pub_raw_data.left_sub_img.origin_height;

  model_depth_img = cv::Mat(depth_h_, depth_w_, CV_16UC1);
  uint16_t *depth_data = (uint16_t *) model_depth_img.data;
  float factor = 1000 * (camera_fx * base_line);
  uint32_t num_pixels = points.size();
//  for (uint32_t i = 0; i < num_pixels; ++i) {
//    if (points[i] > 0) {
//      depth_data[i] = factor / points[i];
//    } else {
//      depth_data[i] = 0;
//    }
//  }

  float32x4_t zero_vec = vdupq_n_f32(0.f);
  //uint16x4_t zero_vec_u16 = vget_low_u16(vdupq_n_u16(0));
  float32x4_t factor_vector = vdupq_n_f32(factor);
  for (uint32_t i = 0; i < num_pixels; i += 4) {
    float32x4_t points_vec = vld1q_f32(&points[i]);
    uint32x4_t mask = vcgtq_f32(points_vec, zero_vec);
    float32x4_t depth_vec = vdivq_f32(factor_vector, points_vec);
    uint16x4_t depth_int16_vec = vmovn_u32(vcvtq_u32_f32(vbslq_f32(mask, depth_vec, zero_vec)));
    vst1_u16(&depth_data[i], depth_int16_vec);
  }

  pub_raw_data.model_depth_img = model_depth_img;

  if (img_origin_width != depth_w_ || img_origin_height != depth_h_) {
    cv::resize(model_depth_img, depth_img,
               cv::Size(img_origin_width, img_origin_height));
    image_size_points.resize(img_origin_width * img_origin_height);
    cv::Mat image_size_points_mat(img_origin_height, img_origin_width,
                                  CV_32FC1, image_size_points.data());
    cv::Mat points_mat(depth_h_, depth_w_, CV_32FC1, points.data());
    cv::resize(points_mat, image_size_points_mat,
               cv::Size(img_origin_width, img_origin_height));
  } else {
    depth_img = model_depth_img;
    image_size_points = points;
  }
}

void StereoNetNode::pub_func(pub_data_t &pub_raw_data) {
  int ret = 0;
  static uint64_t last_ts;
  if (last_ts > pub_raw_data.ts) {
    RCLCPP_ERROR(this->get_logger(),
                 "disorder find! current_ts: %f, last_ts: %f",
                pub_raw_data.ts * 1e-9, last_ts * 1e-9);
    return;
  }
  if (render_perf_) {
    RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 4000,
        "fps: %d, latency: %dms, cpu_usage: %d%, bpu_usage: %d%",
        pub_raw_data.fps, pub_raw_data.latency, pub_raw_data.cpu_usage, pub_raw_data.bpu_usage);
  }

  {
    if (pointcloud2_pub_->get_subscription_count() > 0 ||
        visual_image_pub_->get_subscription_count() > 0) {
      if (pub_raw_data.left_sub_img.image_type == sub_image_type::NV12) {
        image_conversion::nv12_to_bgr(pub_raw_data.left_sub_img.image,
                                      pub_raw_data.left_sub_img.bgr);
        //image_conversion::nv12_to_bgr(pub_raw_data.right_sub_img.image,
        //    pub_raw_data.right_sub_img.bgr);
      }
    }
  }

  {
    ScopeProcessTime t("pub_visual");
    ret = pub_visual_image(pub_raw_data);
  }
  {
    ScopeProcessTime t("pub_depth_image");
    ret = pub_depth_image(pub_raw_data);
    pub_depth_camera_info(pub_raw_data);
  }
  {
    ScopeProcessTime t("pub_pointcloud2");
    ret = pub_pointcloud2(pub_raw_data);
  }
  {
    ScopeProcessTime t("pub_rectified");
    ret = pub_rectified_image(pub_raw_data);
  }

  if (ret != 0) {
    RCLCPP_ERROR(this->get_logger(), "pub failed, ret: %d", ret);
  }
}

int StereoNetNode::start() {
  int ret = 0;
  if (is_running_) return 0;
  stereonet_process_ = std::make_shared<StereonetProcess>();
  ret = stereonet_process_->stereonet_init(stereonet_model_file_path_, max_disp_, postprocess_, uncertainty_th_);
  if (ret != 0) {
    RCLCPP_FATAL(this->get_logger(), "stereonet model init failed");
    stereonet_process_ = nullptr;
    return ret;
  } else {
    RCLCPP_INFO(this->get_logger(), "stereonet model init successed");
  }
  stereonet_process_->get_input_width_height(model_input_w_, model_input_h_);
  stereonet_process_->get_depth_width_height(depth_w_, depth_h_);
  camera_config_parse(stereo_calib_file_path_,
                      model_input_w_, model_input_h_);
  RCLCPP_WARN(this->get_logger(),
              "\033[31m=> rectified fx: %f, fy: %f, cx: %f, cy: %f, base_line: :%f\033[0m",
              camera_fx,
              camera_fy,
              camera_cx,
              camera_cy,
              base_line);
  compare_visual_ = cv::Mat::zeros(cv::Size(depth_w_, depth_h_ * 2), CV_8UC3);

  if (render_perf_) {
    performance_writer::Get();
  }

  is_running_ = true;
  work_thread_.emplace_back(std::make_shared<std::thread>(
      [this] { inference_func(); }));
  work_thread_.emplace_back(std::make_shared<std::thread>(
      [this] { inference_func(); }));

  render_thread_.emplace_back(std::make_shared<std::thread>(
      [this] { render_func(); }));

  return 0;
}

int StereoNetNode::stop() {
  if (!is_running_) return 0;
  is_running_ = false;
  for (auto &t : work_thread_) {
    t->join();
  }
  work_thread_.clear();

  for (auto &t : render_thread_) {
    t->join();
  }
  render_thread_.clear();

  if (stereonet_process_) {
    stereonet_process_->stereonet_deinit();
    stereonet_process_ = nullptr;
  }
  return 0;
}

void StereoNetNode::camera_config_parse(const std::string &file_path,
                                        int model_input_w, int model_input_h) {
  int i = 0;
  cv::FileStorage fs(file_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    RCLCPP_WARN_STREAM(this->get_logger(), "Failed to open " << file_path);
    return;
  }

  do {
    std::string stereo_no = "stereo" + std::to_string(i);
    if (!fs[stereo_no].empty()) {
      RCLCPP_WARN_STREAM(this->get_logger(), "Add StereoRectify Instance: " << stereo_no);
      stereo_rectify_list_.emplace_back(std::make_shared<StereoRectify>(
          fs[stereo_no], model_input_w, model_input_h, resize_before_rectify_));
    } else {
      if (i == 0) {
        RCLCPP_WARN_STREAM(this->get_logger(), "Add StereoRectify fake Instance: " << stereo_no);
        stereo_rectify_list_.emplace_back(std::make_shared<StereoRectify>(
            fs, model_input_w, model_input_h, resize_before_rectify_));
      }
      break;
    }
    i++;
  } while (true);

  if (need_rectify_ || load_rectify_param_) {
    stereo_rectify_list_.back()->GetIntrinsic(camera_cx, camera_cy, camera_fx, camera_fy, base_line);
    RCLCPP_WARN(this->get_logger(), "rectified fx: %f, fy: %f, cx: %f, cy: %f, base_line: :%f",
                camera_fx, camera_fy, camera_cx, camera_cy, base_line);
  }

  fs.release();
}

void StereoNetNode::parameter_configuration() {
  this->declare_parameter("camera_cx", 479.5f);
  this->get_parameter("camera_cx", camera_cx);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_cx: " << camera_cx);

  this->declare_parameter("camera_fx", 450.0f);
  this->get_parameter("camera_fx", camera_fx);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_fx: " << camera_fx);

  this->declare_parameter("camera_cy", 269.5f);
  this->get_parameter("camera_cy", camera_cy);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_cy: " << camera_cy);

  this->declare_parameter("camera_fy", 450.0f);
  this->get_parameter("camera_fy", camera_fy);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_fy: " << camera_fy);

  this->declare_parameter("need_rectify", true);
  this->get_parameter("need_rectify", need_rectify_);
  RCLCPP_INFO_STREAM(this->get_logger(), "need_rectify: " << need_rectify_);

  this->declare_parameter("need_pcl_filter", false);
  this->get_parameter("need_pcl_filter", need_pcl_filter_);
  RCLCPP_INFO_STREAM(this->get_logger(), "need_pcl_filter: " << need_pcl_filter_);

  this->declare_parameter("save_image", false);
  this->get_parameter("save_image", save_image_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_image: " << save_image_);

  this->declare_parameter("save_dir", "./stereonet_images");
  this->get_parameter("save_dir", save_dir_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_dir: " << save_dir_);

  this->declare_parameter("save_image_all", false);
  this->get_parameter("save_image_all", save_image_all_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_image_all: " << save_image_all_);

  this->declare_parameter("save_freq", 1);
  this->get_parameter("save_freq", save_freq_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_freq: " << save_freq_);

  this->declare_parameter("save_total", -1);
  this->get_parameter("save_total", save_total_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_total: " << save_total_);

  this->declare_parameter("save_image_to_nv12", false);
  this->get_parameter("save_image_to_nv12", save_image_to_nv12_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_image_to_nv12: " << save_image_to_nv12_);

  this->declare_parameter("postprocess", "v1");
  this->get_parameter("postprocess", postprocess_);
  RCLCPP_INFO_STREAM(this->get_logger(), "postprocess: " << postprocess_);

  this->declare_parameter("base_line", 0.1f);
  this->get_parameter("base_line", base_line);
  RCLCPP_INFO_STREAM(this->get_logger(), "base_line: " << base_line);

  this->declare_parameter("stereonet_model_file_path", "./config/model.hbm");
  this->get_parameter("stereonet_model_file_path", stereonet_model_file_path_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereonet_model_file_path: " << stereonet_model_file_path_);

  this->declare_parameter("stereo_calib_file_path", "./config/stereo.yaml");
  this->get_parameter("stereo_calib_file_path", stereo_calib_file_path_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereo_calib_file_path: " << stereo_calib_file_path_);

  this->declare_parameter("stereo_image_topic", "/stereo_image");
  this->get_parameter("stereo_image_topic", stereo_image_topic_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereo_image_topic: " << stereo_image_topic_);

  this->declare_parameter("camera_info_topic", "/camera_right_info");
  this->get_parameter("camera_info_topic", camera_info_topic_);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_info_topic: " << camera_info_topic_);

  this->declare_parameter("local_image_path", "./config/");
  this->get_parameter("local_image_path", local_image_path_);
  RCLCPP_INFO_STREAM(this->get_logger(), "local_image_path_: " << local_image_path_);

  this->declare_parameter("height_min", -0.2);
  this->get_parameter("height_min", height_min_);
  RCLCPP_INFO_STREAM(this->get_logger(), "height_min_: " << height_min_);

  this->declare_parameter("height_max", 1.f);
  this->get_parameter("height_max", height_max_);
  RCLCPP_INFO_STREAM(this->get_logger(), "height_max: " << height_max_);

  this->declare_parameter("stereo_combine_mode", stereo_combine_mode_);
  this->get_parameter("stereo_combine_mode", stereo_combine_mode_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereo_combine_mode: " << stereo_combine_mode_);

  this->declare_parameter("leaf_size", leaf_size_);
  this->get_parameter("leaf_size", leaf_size_);
  RCLCPP_INFO_STREAM(this->get_logger(), "leaf_size: " << leaf_size_);

  this->declare_parameter("KMean", KMean_);
  this->get_parameter("KMean", KMean_);
  RCLCPP_INFO_STREAM(this->get_logger(), "KMean: " << KMean_);

  this->declare_parameter("stdv", stdv_);
  this->get_parameter("stdv", stdv_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stdv: " << stdv_);

  this->declare_parameter("visual_beta", visual_beta_);
  this->get_parameter("visual_beta", visual_beta_);
  RCLCPP_INFO_STREAM(this->get_logger(), "visual_beta: " << visual_beta_);

  this->declare_parameter("visual_alpha", visual_alpha_);
  this->get_parameter("visual_alpha", visual_alpha_);
  RCLCPP_INFO_STREAM(this->get_logger(), "visual_alpha: " << visual_alpha_);

  this->declare_parameter("max_disp", max_disp_);
  this->get_parameter("max_disp", max_disp_);
  RCLCPP_INFO_STREAM(this->get_logger(), "max_disp: " << max_disp_);

  this->declare_parameter("rectify_bgr", pub_rectified_bgr_);
  this->get_parameter("rectify_bgr", pub_rectified_bgr_);
  RCLCPP_INFO_STREAM(this->get_logger(), "pub_rectified_bgr: " << pub_rectified_bgr_);

  this->declare_parameter("image_format", image_format_);
  this->get_parameter("image_format", image_format_);
  RCLCPP_INFO_STREAM(this->get_logger(), "image_format: " << image_format_);

  this->declare_parameter("image_sleep", image_inference_sleep_ms_);
  this->get_parameter("image_sleep", image_inference_sleep_ms_);
  RCLCPP_INFO_STREAM(this->get_logger(), "image_inference_sleep_ms: " << image_inference_sleep_ms_);

  std::string depth_type = "point";
  this->declare_parameter("depth_type", depth_type);
  this->get_parameter("depth_type", depth_type);
  RCLCPP_INFO_STREAM(this->get_logger(), "depth_type: " << depth_type);
  if (depth_type == "region") {
    depth_type_point_ = false;
  }

  render_type_ = this->declare_parameter("render_type", render_type_);
  if (render_type_ < 0 || render_type_ >= 3) {
    render_type_ = 0;
  }
  RCLCPP_INFO_STREAM(this->get_logger(), "render_type: " << render_type_);

  render_need_filter_ = this->declare_parameter("render_need_filter", render_need_filter_);
  RCLCPP_INFO_STREAM(this->get_logger(), "render_need_filter: " << render_need_filter_);

  render_max_depth_ = this->declare_parameter("render_max_depth", render_max_depth_);
  RCLCPP_INFO_STREAM(this->get_logger(), "render_max_depth: " << render_max_depth_);

  depth_need_filter_ = this->declare_parameter("depth_need_filter", depth_need_filter_);
  RCLCPP_INFO_STREAM(this->get_logger(), "depth_need_filter: " << depth_need_filter_);

  pc_max_depth_ = this->declare_parameter("pc_max_depth", pc_max_depth_);
  RCLCPP_INFO_STREAM(this->get_logger(), "pc_max_depth: " << pc_max_depth_);

  uncertainty_th_ = this->declare_parameter("uncertainty_th", uncertainty_th_);
  RCLCPP_INFO_STREAM(this->get_logger(), "uncertainty_th: " << uncertainty_th_);

  resize_before_rectify_ = this->declare_parameter("resize_before_rectify", resize_before_rectify_);
  RCLCPP_INFO_STREAM(this->get_logger(), "resize_before_rectify: " << resize_before_rectify_);

  load_rectify_param_ = this->declare_parameter("load_rectify_param", false);
  RCLCPP_INFO_STREAM(this->get_logger(), "load_rectify_param: " << load_rectify_param_);

  render_perf_ = this->declare_parameter("render_perf", false);
  RCLCPP_INFO_STREAM(this->get_logger(), "render_perf: " << render_perf_);
}

void StereoNetNode::inference_by_usb_camera() {
/*
  cv::VideoCapture *capture = nullptr;
  cv::Mat stereo_img, left_img, right_img;
  cv::Rect right_rect(0, 0, 640, 360),
           left_rect(0, 360, 640, 360);
  uint64_t current_ts;

  capture = new cv::VideoCapture("/dev/video0");
  assert(capture != nullptr);
  int fps = capture->get(cv::CAP_PROP_FPS);
  int width = capture->get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture->get(cv::CAP_PROP_FRAME_HEIGHT);

  RCLCPP_INFO(this->get_logger(),
      "usb camera fps: %d, width: %d, height: %d",
      fps, width, height);

  while (rclcpp::ok()) {
    if (capture->grab()) {
      current_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
      capture->retrieve(stereo_img);
      left_img = stereo_img(left_rect);
      right_img = stereo_img(right_rect);
      inference_data_t inference_data = std::make_tuple(left_img, right_img);
      inference(inference_data);
    }
  }
  delete capture;
*/
}

int get_image(const std::string &image_path,
              cv::Mat &left_img, cv::Mat &right_img, int64_t &ts,
              const std::string &image_format) {
  static uint32_t i_num = 0;
  std::stringstream iss;
  std::string image_seq;
  iss << std::setw(6) << std::setfill('0') << i_num++;
  image_seq = iss.str();
  std::string left_img_path = image_path + "/left" + image_seq + "." + image_format;
  std::string right_img_path = image_path + "/right" + image_seq + "." + image_format;
  left_img = cv::imread(left_img_path);
  right_img = cv::imread(right_img_path);
  ts = 0;
  if (left_img.empty() || right_img.empty()) {
    // RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "=> left_img_path: " << left_img_path << ", right_img_path: " << right_img_path << " not existed!");
    // i_num = 0;
    return -1;
  }
  RCLCPP_INFO_STREAM(rclcpp::get_logger(""),
                     "=> left_img_path: " << left_img_path << ", right_img_path: " << right_img_path);
  return 0;
}

void get_image_file_list(const std::string &image_path,
                         std::vector<std::string> &file_names) {
  DIR *pDir;
  struct dirent *ptr;
  if (!(pDir = opendir(image_path.c_str()))) {
    RCLCPP_ERROR(rclcpp::get_logger(""),
                 "image path is not existed: %s ", image_path.c_str());
    return;
  }
  while ((ptr = readdir(pDir)) != 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
      std::string file_name = ptr->d_name;
      size_t lastDot = file_name.find_last_of('.');
      if (lastDot != std::string::npos) {
        auto extension = file_name.substr(lastDot + 1);
        std::transform(extension.begin(),
                       extension.end(), extension.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (extension == "png" ||
            extension == "jpg" || extension == "jpeg") {
          file_names.push_back(file_name);
        }
      }
    }
  }
  sort(file_names.begin(), file_names.end());
  closedir(pDir);
}

int get_image2(const std::string &image_path, cv::Mat &left_img,
               cv::Mat &right_img, int64_t &ts, const std::string &image_format) {
  static std::vector<std::string> left_file_names, right_file_names;
  static uint32_t i_num = 0;
  if (i_num == 0) {
    get_image_file_list(image_path + "/cam0/data/", left_file_names);
    get_image_file_list(image_path + "/cam1/data/", right_file_names);
  }
  if (i_num < left_file_names.size()) {
    std::string file_name;
    size_t lastDot = left_file_names[i_num].find_last_of('.');
    // 分离文件名和后缀
    if (lastDot == std::string::npos) {
      file_name = left_file_names[i_num];
    } else {
      file_name = left_file_names[i_num].substr(0, lastDot);
    }
    ts = std::atoll(file_name.c_str());
    left_img = cv::imread(image_path + "/cam0/data/" + left_file_names[i_num]);
    right_img = cv::imread(image_path + "/cam1/data/" + right_file_names[i_num]);
    i_num++;
    return 0;
  }
  i_num = 0;
  return -1;
}

void StereoNetNode::inference_by_image() {
  std_msgs::msg::Header image_header;
  sub_image left_sub_img, right_sub_img;
  int64_t ts;
  while (rclcpp::ok()) {
    if (inference_que_.size() > 5) {
      RCLCPP_WARN_THROTTLE(this->get_logger(),
                           *this->get_clock(), 2000, "inference que is full!");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    if (-1 == get_image(local_image_path_, left_sub_img.image,
                        right_sub_img.image, ts, image_format_)) {
//    if (-1 == get_image2(local_image_path_, left_sub_img.image,
//                         right_sub_img.image, ts, image_format_)) {
      continue;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(image_inference_sleep_ms_));
    image_header.frame_id = "default_cam";
    auto now = this->now();
    if (ts != 0) {
      image_header.stamp = rclcpp::Time(ts);
    } else {
      image_header.stamp = now;
    }
    left_sub_img.image_type = sub_image_type::BGR;
    right_sub_img.image_type = sub_image_type::BGR;
    left_sub_img.header = image_header;
    right_sub_img.header = image_header;
    left_sub_img.origin_height = left_sub_img.image.rows;
    left_sub_img.origin_width = left_sub_img.image.cols;
    right_sub_img.origin_height = right_sub_img.image.rows;
    right_sub_img.origin_width = right_sub_img.image.cols;
    inference_que_.put({left_sub_img, right_sub_img, true, now});
  }
}

void StereoNetNode::pub_sub_configuration() {
  stereo_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_image_topic_, 10,
      std::bind(&StereoNetNode::stereo_image_cb, this, std::placeholders::_1));

  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, 10,
      std::bind(&StereoNetNode::camera_info_cb, this, std::placeholders::_1)
  );

  pointcloud2_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "~/stereonet_pointcloud2", 10);

  depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "~/stereonet_depth", 10);

  depthcompressed_image_pub_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
      "~/stereonet_compresseddepth", 10);

  visual_topic_ = this->declare_parameter("visual_topic", visual_topic_);
  RCLCPP_INFO_STREAM(this->get_logger(), "visual_topic: " << visual_topic_);
  visual_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      visual_topic_, 10);

  rectified_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      rectified_image_topic_, 10);

  rectified_right_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      rectified_right_image_topic_, 10);

  depth_camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
      "~/camera_info", 10);

  depth_compare = false;
  std::string compare_depth_topic = "~/stereonet_depth";
  std::string compare_image_topic = "~/rectified_image";

  depth_compare = this->declare_parameter("depth_compare", depth_compare);
  RCLCPP_INFO_STREAM(this->get_logger(), "depth_compare: " << depth_compare);

  compare_depth_topic = this->declare_parameter("compare_depth_topic", compare_depth_topic);
  RCLCPP_INFO_STREAM(this->get_logger(), "compare_depth_topic: " << compare_depth_topic);

  compare_image_topic = this->declare_parameter("compare_image_topic", compare_image_topic);
  RCLCPP_INFO_STREAM(this->get_logger(), "compare_image_topic: " << compare_image_topic);

  if (depth_compare) {
    depth_subscriber_.subscribe(this, compare_depth_topic);
    //color_subscriber_.subscribe(this, visual_topic);
    compare_left_subscriber_.subscribe(this, compare_image_topic);

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10),
        depth_subscriber_,
        //color_subscriber_,
        compare_left_subscriber_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
    sync_->registerCallback(std::bind(&StereoNetNode::sync_callback,
                                      this, std::placeholders::_1,
        //std::placeholders::_2,
                                      std::placeholders::_2
    ));

    compare_visual_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "~/compare_stereonet_visual", 10);

/*
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
    qos.durability(rclcpp::DurabilityPolicy::TransientLocal);

     depth_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
       compare_depth_topic, qos,
       std::bind(&StereoNetNode::d_callback, this, std::placeholders::_1));


     compare_left_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
       compare_image_topic, qos,
       std::bind(&StereoNetNode::c_callback, this, std::placeholders::_1));
      */
  }
}

void StereoNetNode::d_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
  std::cout << "d cb" << std::endl;

}

void StereoNetNode::c_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
  std::cout << "c cb" << std::endl;

}

void StereoNetNode::sync_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &depth_msg,
    //const sensor_msgs::msg::Image::ConstSharedPtr &color_msg,
                                  const sensor_msgs::msg::CompressedImage::ConstSharedPtr &rs_msg) {
  sensor_msgs::msg::Image visual_image_msg;
  cv_bridge::CvImage img_bridge;

  const float fx_bl = 37050;
  int width, height;
  std::vector<unsigned char> compressed_data(depth_msg->data.begin() + 12, depth_msg->data.end());
  cv::Mat depth_image = cv::imdecode(compressed_data, cv::IMREAD_UNCHANGED);
  //  cv::Mat depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::MONO16)->image;
  cv::Mat rs_image = cv_bridge::toCvCopy(rs_msg, sensor_msgs::image_encodings::BGR8)->image;
  width = depth_w_;
  height = depth_h_;

  if (depth_image.empty() || rs_image.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to decode compressed image");
    return;
  }

  if (depth_image.cols != width || depth_image.rows != height) {
    cv::resize(depth_image, depth_image, cv::Size(width, height));
  }
  if (rs_image.cols != width || rs_image.rows != height) {
    cv::resize(rs_image, rs_image, cv::Size(width, height));
  }

  cv::Mat dis_image = fx_bl / depth_image;
  dis_image.convertTo(dis_image, CV_8U);
  cv::applyColorMap(dis_image, dis_image, userColor_);

  int step_num = 6;
  int x_step = width / step_num;
  int y_step = height / step_num;

  for (int i = 1; i < step_num; i++) {
    for (int j = 1; j < step_num; j++) {
      cv::line(rs_image, cv::Point2i(0, i * y_step),
               cv::Point2i(width, i * y_step),
               cv::Scalar(255, 255, 255), 1);
      cv::line(rs_image, cv::Point2i(j * x_step, 0),
               cv::Point2i(j * x_step, height),
               cv::Scalar(255, 255, 255), 1);

      cv::line(dis_image, cv::Point2i(0, i * y_step),
               cv::Point2i(width, i * y_step),
               cv::Scalar(255, 255, 255), 1);
      cv::line(dis_image, cv::Point2i(j * x_step, 0),
               cv::Point2i(j * x_step, height),
               cv::Scalar(255, 255, 255), 1);
      uint16_t Z = depth_image.at<uint16_t>(i * y_step, j * x_step);
      double distance = static_cast<double>(Z) / 1000.0;
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(3) << distance << "m";

      cv::putText(rs_image, ss.str(), cv::Point2i(j * x_step + 3,
                                                  i * y_step - 3),
                  cv::FONT_HERSHEY_SIMPLEX, 1,
                  cv::Scalar(255, 255, 255), 2);

      cv::putText(dis_image, ss.str(), cv::Point2i(j * x_step + 3,
                                                   i * y_step - 3),
                  cv::FONT_HERSHEY_SIMPLEX, 1,
                  cv::Scalar(255, 255, 255), 2);

    }
  }

  cv::putText(rs_image, "realsense", cv::Point2i(3, 38),
              cv::FONT_HERSHEY_SIMPLEX, 1.5,
              cv::Scalar(0, 255, 0), 3);

  cv::putText(dis_image, "relasense-depth", cv::Point2i(3, 38),
              cv::FONT_HERSHEY_SIMPLEX, 1.5,
              cv::Scalar(0, 255, 0), 3);

  {
    std::lock_guard<std::mutex> lck(compare_visual_mtx_);
    cv::vconcat(rs_image, dis_image, compare_visual_);
  }
}

void StereoNetNode::camera_info_cb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info_msg) {
  camera_fx = camera_info_msg->p[0];
  camera_fy = camera_info_msg->p[5];
  camera_cx = camera_info_msg->p[2];
  camera_cy = camera_info_msg->p[6];
  base_line = camera_info_msg->p[3] / camera_fx;

  RCLCPP_WARN_ONCE(this->get_logger(), "\033[31m=> sub rectified fx: %f, fy: %f, cx: %f, cy: %f, base_line: :%f\033[0m",
                   camera_fx, camera_fy, camera_cx, camera_cy, base_line);
}

void StereoNetNode::publish_static_tf()
{
  static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
  geometry_msgs::msg::TransformStamped t;
  t.header.stamp = now();
  t.header.frame_id = "camera_link";
  t.child_frame_id = "camera_depth_frame";

  t.transform.translation.x = 0.0;
  t.transform.translation.y = 0.0;
  t.transform.translation.z = 0.0;

  tf2::Quaternion q;
  q.setRPY(-M_PI / 2, 0, -M_PI / 2);
  q.normalize();

  t.transform.rotation.x = q.x();
  t.transform.rotation.y = q.y();
  t.transform.rotation.z = q.z();
  t.transform.rotation.w = q.w();

  static_broadcaster_->sendTransform(t);
}

}

RCLCPP_COMPONENTS_REGISTER_NODE(stereonet::StereoNetNode)
