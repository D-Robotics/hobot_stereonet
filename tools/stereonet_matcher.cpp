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

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <dirent.h>
#include <sensor_msgs/msg/image.hpp>
#include <functional>
#include <cv_bridge/cv_bridge.h>
//#include "../include/stereo_rectify.h"

class StereoNetMatcher : public rclcpp::Node {
 public:

  void stereo_image_cb(const sensor_msgs::msg::Image::SharedPtr img) {
    cv::Mat stereo_img, left_img, right_img;
    const std::string &encoding = img->encoding;
    if (encoding == "nv12" || encoding == "NV12") {
      cv::Mat nv12(img->height * 3 / 2, img->width, CV_8UC1, img->data.data());
      cv::cvtColor(nv12, stereo_img, cv::COLOR_YUV2BGR_NV12);
    } else if (encoding == "bgr8" || encoding == "BGR8") {
      stereo_img = cv_bridge::toCvShare(img)->image;
    }

    {
      std::lock_guard<std::mutex> lck(image_mutex);
      stereo_image = stereo_img;
      image_ts = 1e9 * img->header.stamp.sec + img->header.stamp.nanosec;
      image_update = true;
    }
  }

  StereoNetMatcher(const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions())
      : rclcpp::Node("StereoNetMatcher", node_options) {
    stereo_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        stereo_image_topic_, 10,
        std::bind(&StereoNetMatcher::stereo_image_cb, this, std::placeholders::_1));
  }

  ~StereoNetMatcher() {}

  int get_image_from_ros(cv::Mat& left_image, cv::Mat& right_image, int64_t &ts) {
    std::lock_guard<std::mutex> lck(image_mutex);
    if (image_update) {
      left_image = stereo_image(
          cv::Rect(0, 0, stereo_image.cols, stereo_image.rows / 2)).clone();
      right_image = stereo_image(
          cv::Rect(0, stereo_image.rows / 2, stereo_image.cols, stereo_image.rows / 2)).clone();
      image_update = false;
      ts = image_ts;
      return 0;
    }
    return -1;
  }

 private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr stereo_image_sub_;
  std::string stereo_image_topic_ = "/image_combine_raw";

  std::mutex image_mutex;
  cv::Mat stereo_image;
  bool image_update {false};
  int64_t image_ts;
};

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
               cv::Mat &right_img, int64_t &ts) {
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
    right_img = cv::imread(image_path + "/cam1/data/"+ right_file_names[i_num]);
    i_num++;
    return 0;
  }
  //i_num = 0;
  return -1;
}

float scale = 1;
int click_x, click_y;
cv::Mat match_image;
std::mutex mtx;

void draw_text(cv::Mat &image) {
  cv::Vec3b pixel = image.at<cv::Vec3b>(click_y, click_x);
  int b = pixel[0];
  int g = pixel[1];
  int r = pixel[2];
  std::string text = "x: " + std::to_string(click_x) + " y: " + std::to_string(click_y);
  std::string color_info = "rgb: " + std::to_string(r) + ", " + std::to_string(g) + ", " + std::to_string(b);
  cv::Point text_origin(10, 10);
  cv::Point rect_top_left(click_x - 5, click_y - 5);
  cv::Point rect_bottom_right(click_x + 5, click_y + 5);
  text_origin.x = std::max(0, text_origin.x);
  text_origin.y = std::max(20, text_origin.y);
  rect_top_left.x = std::max(0, rect_top_left.x);
  rect_top_left.y = std::max(0, rect_top_left.y);
  rect_bottom_right.x = std::min(image.cols - 1, rect_bottom_right.x);
  rect_bottom_right.y = std::min(image.rows - 1, rect_bottom_right.y);
  cv::rectangle(image, rect_top_left, rect_bottom_right, cv::Scalar(0, 255, 0), 1);

  int font = cv::FONT_HERSHEY_SIMPLEX;
  double font_scale = 0.5;
  int thickness = 1;

  cv::Size text_size = cv::getTextSize(text, font, font_scale, thickness, nullptr);
  cv::Point box_bottom_right(text_origin.x + std::max(text_size.width, 200), text_origin.y + 20); // 文本框大小

  cv::rectangle(image, text_origin - cv::Point(5, text_size.height + 5), box_bottom_right, cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(image, text, text_origin, font, font_scale, cv::Scalar(255, 255, 255), thickness);
  cv::putText(image, color_info, text_origin + cv::Point(0, text_size.height + 5), font, font_scale, cv::Scalar(255, 255, 255), thickness);
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
  if (event == cv::EVENT_MOUSEHWHEEL) {
    int delta = cv::getMouseWheelDelta(flags);
    if (delta > 0) {
      scale -= 0.1;
    } else if (delta < 0) {
      scale += 0.1;
    }
    if (scale > 10) scale = 10.f;
    if (scale < 0.1) scale = 0.1f;

  } else if (event == cv::EVENT_LBUTTONUP)  {
    click_x = x / scale;
    click_y = y / scale;
    std::lock_guard<std::mutex> lck(mtx);
    draw_text(match_image);
  } else if (event == cv::EVENT_MOUSEMOVE) {
  }
}

bool isWithinRadius(const cv::KeyPoint& kp,
    const std::vector<cv::KeyPoint>& selected_points, float radius) {
  for (const auto& selected_kp : selected_points) {
    float dist = std::sqrt(std::pow(kp.pt.x - selected_kp.pt.x, 2)
        + std::pow(kp.pt.y - selected_kp.pt.y, 2));
    if (dist < radius) {
      return true;
    }
  }
  return false;
}

int get_image_from_file(const std::string &local_image_path,
    cv::Mat& left_image, cv::Mat& right_image, int64_t &ts) {
  return get_image2(local_image_path,
                    left_image, right_image, ts);
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int64_t ts;
  std::string local_image_path = "./images/", image_source = "file";
  cv::Mat img_left, img_right;

  std::shared_ptr<StereoNetMatcher> stereo_net_matcher = std::make_shared<StereoNetMatcher>();

  stereo_net_matcher->declare_parameter("image_path", local_image_path);
  stereo_net_matcher->get_parameter("image_path", local_image_path);
  RCLCPP_INFO(stereo_net_matcher->get_logger(), "image_path: %s", local_image_path.c_str());

  stereo_net_matcher->declare_parameter("image_source", image_source);
  stereo_net_matcher->get_parameter("image_source", image_source);
  RCLCPP_INFO(stereo_net_matcher->get_logger(), "image_source: %s", image_source.c_str());

  match_image = cv::Mat::zeros(800, 600, CV_8UC3);

  auto render = [&](){
    cv::Mat render_image;
    cv::namedWindow("Example Image", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Example Image", onMouse);
    while (rclcpp::ok()) {
      {
        std::lock_guard<std::mutex> lck(mtx);
        render_image = match_image.clone();
      }
      cv::resize(render_image, render_image,
                 cv::Size(render_image.cols * scale, render_image.rows * scale));
      cv::imshow("Example Image", render_image);
      int key = cv::waitKey(100);
      if (key == 'q' || key == 27) return ;
    }
    cv::destroyAllWindows();
  };

  std::thread render_thread(render);
  std::thread ros_thread([&]() {
    while (rclcpp::ok()) {
      rclcpp::spin(stereo_net_matcher);
    }
  });

  while (rclcpp::ok()) {
    if (image_source == "file") {
      if (get_image_from_file(local_image_path, img_left, img_right, ts) == -1) {
        std::cout << "image complete" << std::endl;
      }
    } else {
      if (stereo_net_matcher->get_image_from_ros(img_left, img_right, ts) == -1) {
        std::cout << "image not update" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        continue;
      }
    }

    cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);

    // 检测关键点并计算描述符
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img_left, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img_right, cv::Mat(), keypoints2, descriptors2);

    // 检查是否成功提取描述符
    if (descriptors1.empty() || descriptors2.empty()) {
      std::cerr << "Error: Could not compute descriptors!" << std::endl;
      continue;
    }

    // 创建 BFMatcher 进行特征匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING, true); // NORM_HAMMING 用于二进制特征描述符
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> filtered_matches;
    matcher.match(descriptors1, descriptors2, matches);

    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
      points1.push_back(keypoints1[match.queryIdx].pt);
      points2.push_back(keypoints2[match.trainIdx].pt);
    }
    cv::Mat mask;
    cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 3.0, mask);

    std::vector<cv::DMatch> finalMatches;
    for (size_t i = 0; i < matches.size(); i++) {
      if (mask.at<uchar>(i)) {
        finalMatches.push_back(matches[i]);
      }
    }

    // 按匹配质量排序
    std::sort(finalMatches.begin(), finalMatches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
      return a.distance < b.distance;
    });
    const int max_matches = 10;
    const float RADIUS = 50.0f; // 半径限制
    std::vector<cv::KeyPoint> selected_keypoints1;
    for (const auto& match : finalMatches) {
      const cv::KeyPoint& kp1 = keypoints1[match.queryIdx]; // img1 中的点
      if (!isWithinRadius(kp1, selected_keypoints1, RADIUS)) {
        selected_keypoints1.push_back(kp1);
        filtered_matches.push_back(match);
      }
      if (filtered_matches.size() >= max_matches) {
        break; // 筛选出前 10 个点后停止
      }
    }

    {
      std::unique_lock<std::mutex> lck(mtx);
      cv::drawMatches(img_left, keypoints1, img_right, keypoints2,
                      filtered_matches,
                      match_image, cv::Scalar::all(-1), cv::Scalar::all(-1),
                      std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

      int i = 0;
      for (const auto &match : filtered_matches) {
        auto pt1 = keypoints1[match.queryIdx].pt;
        auto pt2 = keypoints2[match.trainIdx].pt;
        std::cout << "Match " << i << ": "
                  << "left point: (x: " << pt1.x << ", y:" << pt1.y << "), "
                  << "right point: (x: " << pt2.x << ", y:" << pt2.y << ")"
                  << ". y_diff: " << pt2.y - pt1.y << std::endl;
        i++;
        auto color = cv::Scalar(0, 255, 0);
        if (std::abs(pt1.y - pt2.y) > 1) {
          color = cv::Scalar(0, 0, 255);
        }
        std::string text = "y:" + std::to_string((int)pt1.y);
        cv::putText(match_image, text, pt1,
                    cv::FONT_HERSHEY_PLAIN, 2,
                    color, 2);

        std::string text2 = "y:" + std::to_string((int)pt2.y);
        pt2.x += img_left.cols;
        cv::putText(match_image, text2, pt2,
                    cv::FONT_HERSHEY_PLAIN, 2,
                    color, 2);
      }
      draw_text(match_image);
      cv::imwrite("./result.jpg", match_image);
      lck.unlock();
      int key = cv::waitKey(0);
      if (key == 'q' || key == 27) break;
    }
  }

  rclcpp::shutdown();

  render_thread.join();
  ros_thread.join();
  return 0;
}
