//
// Created by zhy on 7/17/24.
//
#include "stereonet_intra_sub.h"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<stereonet::StereoNetSubNode>();

  while (rclcpp::ok()) {
    rclcpp::spin(node);
  }

  return 0;
}