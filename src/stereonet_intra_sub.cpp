//
// Created by zhy on 7/17/24.
//

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
