//
// Created by zhy on 7/17/24.
//

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
