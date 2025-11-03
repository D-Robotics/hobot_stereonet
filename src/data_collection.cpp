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
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#ifdef CV_BRIDGE_CPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <opencv2/opencv.hpp>
#include <fstream>

class Ros2SubNode : public rclcpp::Node {
 public:
  Ros2SubNode() : Node("data_collector"),
                  image_queue_limit_(50),
                  imu_queue_limit_(800) {

    this->declare_parameter("image0_topic", "/StereoNetNode/rectified_image");
    this->get_parameter("image0_topic", sub_image0_topic_);

    this->declare_parameter("image1_topic", "/StereoNetNode/rectified_right_image");
    this->get_parameter("image1_topic", sub_image1_topic_);

    this->declare_parameter("imu_topic", "/imu_data");
    this->get_parameter("imu_topic", sub_imu_topic_);

    this->declare_parameter("imu_que_limit", imu_queue_limit_);
    this->get_parameter("imu_que_limit", imu_queue_limit_);

    std::cout << "image0_topic: " << sub_image0_topic_ << std::endl;
    std::cout << "img_que_limit: " << image_queue_limit_ << std::endl;
    std::cout << "imu_que_limit: " << imu_queue_limit_ << std::endl;

    system("mkdir -p storage/cam0/data/");
    system("mkdir -p storage/cam1/data/");
    system("mkdir -p storage/imu0/");

    imu_file_ = std::ofstream("storage/imu0/data.csv", std::ios::out);

    sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
        sub_imu_topic_, imu_queue_limit_,
        std::bind(&Ros2SubNode::CollectImu, this, std::placeholders::_1));

    std::function<void(const std::shared_ptr<sensor_msgs::msg::Image>)>
        img0_callback = std::bind(&Ros2SubNode::CollectImage, this, std::placeholders::_1, 0);
    sub_img0_ = this->create_subscription<sensor_msgs::msg::Image>(
        sub_image0_topic_, image_queue_limit_, img0_callback);

    std::function<void(const std::shared_ptr<sensor_msgs::msg::Image>)>
        img1_callback = std::bind(&Ros2SubNode::CollectImage, this, std::placeholders::_1, 1);
    sub_img1_ = this->create_subscription<sensor_msgs::msg::Image>(
        sub_image1_topic_, image_queue_limit_, img1_callback);
  }

  void CollectImu(const sensor_msgs::msg::Imu::SharedPtr imu) {
    uint64_t timestamp =
        imu->header.stamp.sec * 1e9 + imu->header.stamp.nanosec;
    static uint64_t last_ts = timestamp;
//    if (timestamp - last_ts > 3502912) {
//      std::cout << "find imu gap! imu gap: " << timestamp - last_ts << std::endl;
//    }
    last_ts = timestamp;
    float imu_data[9];
    imu_data[0] = imu->linear_acceleration.x;
    imu_data[1] = imu->linear_acceleration.y;
    imu_data[2] = imu->linear_acceleration.z;
    imu_data[3] = imu->angular_velocity.x;
    imu_data[4] = imu->angular_velocity.y;
    imu_data[5] = imu->angular_velocity.z;
    imu_file_ << timestamp << "," << imu_data[3] << "," << imu_data[4] << "," << imu_data[5]
              << "," << imu_data[0] << "," << imu_data[1] << "," << imu_data[2] << std::endl;
  }

  void CollectImage(const sensor_msgs::msg::Image::SharedPtr img, int type) {
    cv::Mat image;
    uint64_t timestamp =
        img->header.stamp.sec * 1e9 + img->header.stamp.nanosec;
    try {
      const std::string &encoding = img->encoding;
      if (encoding == "nv12" || encoding == "NV12" || encoding == "mono8") {
        //  only reserve the part Y
        image = cv::Mat(img->height, img->width,
                        CV_8UC1, img->data.data());
      } else {
        if (encoding == "bgr8") {
          image = cv_bridge::toCvShare(img)->image;
        }
      }

      const std::string &stamp_str = std::to_string(timestamp);
      if (type == 0) {
        cv::imwrite("storage/cam0/data/" + stamp_str + ".png", image);
      } else if (type == 1) {
        cv::imwrite("storage/cam1/data/" + stamp_str + ".png", image);
      }
    } catch (cv_bridge::Exception &e) {
      std::cout << "receive unknow image: " << e.what() << std::endl;
      return;
    }
  }

  int image_queue_limit_;
  int imu_queue_limit_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img0_, sub_img1_, sub_img_;
  std::string sub_image0_topic_, sub_image1_topic_, sub_imu_topic_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;

  std::ofstream imu_file_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto ros2Node = std::make_shared<Ros2SubNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(ros2Node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}