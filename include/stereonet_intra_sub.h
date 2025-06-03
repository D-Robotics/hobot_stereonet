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

#ifndef STEREONET_MODEL_INCLUDE_STEREONET_INTRA_SUB_H_
#define STEREONET_MODEL_INCLUDE_STEREONET_INTRA_SUB_H_
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace stereonet {

class StereoNetSubNode : public rclcpp::Node {
 public:
  explicit StereoNetSubNode(const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions())
      : rclcpp::Node("StereoNetSubNode", node_options) {
    sub_configuration();
  }
  void sub_configuration();
  void point_cloud_cb(sensor_msgs::msg::PointCloud2::SharedPtr point_msg);

 private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_sub_;
};

}

#endif //STEREONET_MODEL_INCLUDE_STEREONET_INTRA_SUB_H_
