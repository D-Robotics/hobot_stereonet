//
// Created by zhy on 7/16/24.
//
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
      while(rclcpp::ok()) {
        node->inference_by_image();
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
      }
    }).detach();
  }

  while (rclcpp::ok()) {
    rclcpp::spin(node);
  }

  node = nullptr;
  return 0;
}
