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
#include "stereonet_component.h"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<stereonet::StereoNetNode>();
  RCLCPP_INFO(node->get_logger(), "This is example of D-Robotics stereonet");

  bool use_usb_camera;
  node->declare_parameter("use_usb_camera", false);
  node->get_parameter("use_usb_camera", use_usb_camera);

  bool use_local_image;
  node->declare_parameter("use_local_image", false);
  node->get_parameter("use_local_image", use_local_image);

  RCLCPP_INFO(node->get_logger(), "Node start successed!");

  if (use_usb_camera) {
    std::thread([&]() {
      while(rclcpp::ok()) {
        //  node->inference_by_usb_camera();
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
      }
    }).detach();
  }

  if (use_local_image) {
    std::thread([&]() {
      while (rclcpp::ok()) {
        node->inference_by_image();
        RCLCPP_INFO(node->get_logger(), "image inference completed!");
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      }
    }).detach();
  }

  while (rclcpp::ok()) {
    rclcpp::spin(node);
  }

  node = nullptr;
  return 0;
}
