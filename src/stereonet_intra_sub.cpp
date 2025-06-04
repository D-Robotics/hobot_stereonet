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

#include "stereonet_intra_sub.h"

namespace stereonet {

void StereoNetSubNode::sub_configuration() {
  point_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/StereoNetNode/stereonet_pointcloud2", 10,
      std::bind(&StereoNetSubNode::point_cloud_cb, this, std::placeholders::_1));
}

void StereoNetSubNode::point_cloud_cb(sensor_msgs::msg::PointCloud2::SharedPtr point_msg) {
  double now = std::chrono::high_resolution_clock::now().time_since_epoch().count() * 1e-9;
  double ts = point_msg->header.stamp.sec + point_msg->header.stamp.nanosec * 1e-9;
  RCLCPP_INFO(this->get_logger(),
              "we received point_cloud msg at: %f, timestamp of point_cloud is: %f, latency is %f",
              now, ts, now - ts);
}

}

RCLCPP_COMPONENTS_REGISTER_NODE(stereonet::StereoNetSubNode)
