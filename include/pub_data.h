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

#ifndef HOBOT_STEREONET_INCLUDE_PUB_DATA_H_
#define HOBOT_STEREONET_INCLUDE_PUB_DATA_H_

#include <vector>
#include "sensor_msgs/msg/image.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace stereonet {

/**
 * @struct PubData
 * @brief Structure to hold data for publishing
 */
struct PubData {
  uint64_t timestamp;
  std_msgs::msg::Header header;
  sensor_msgs::msg::Image::SharedPtr origin_stereo_msg = nullptr;
  cv::Mat disp;
  cv::Mat depth;
  cv::Mat uncert;
  std::vector<uint8_t> rectify_left_img_data;  // nv12
  std::vector<uint8_t> rectify_right_img_data; // nv12
  cv::Mat left_bgr;
  cv::Mat right_bgr;
  cv::Mat origin_left;
  cv::Mat origin_right;
  double fps;
  int latency;
  int cpu_usage;
  int bpu_usage;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud = nullptr;
  cv::Mat visual_img;

  int count;
};

} // namespace stereonet

#endif // HOBOT_STEREONET_INCLUDE_PUB_DATA_H_