//
// Created by zhy on 7/16/24.
//

#ifndef STEREONET_MODEL_INCLUDE_STEREONET_COMPONENT_H_
#define STEREONET_MODEL_INCLUDE_STEREONET_COMPONENT_H_

#include "blockqueue.h"
#include "stereonet_process.h"
#include "stereo_rectify.h"

#include <fstream>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cv_bridge/cv_bridge.h>
#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/time.hpp>

namespace stereonet {

class StereoNetNode : public rclcpp::Node {
 public:

  enum sub_image_type {
    BGR,
    NV12
  };

  struct sub_image {
    cv::Mat image;
    sub_image_type image_type;
    std_msgs::msg::Header header;
  };

  struct inference_data_t {
     sub_image left_sub_img;
     sub_image right_sub_img;
  };
  struct pub_data_t {
    sub_image left_sub_img;
    std::vector<float> points;
    cv::Mat depth_img;
  };

  StereoNetNode(const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions())
  : rclcpp::Node("StereoNetNode", node_options) {
    parameter_configuration();
    pub_sub_configuration();
    if (start() != 0) {
      RCLCPP_FATAL(get_logger(), "Node start failed");
    }
  }

  ~StereoNetNode() {
    stop();
  }

  void parameter_configuration();
  void camera_config_parse(const std::string &file_path,
                           int model_input_w, int model_input_h);

  int inference(const inference_data_t &, std::vector<float> &points);
  void inference_func();
  void pub_func(pub_data_t &pub_raw_data);

  int start();
  int stop();

  void stereo_image_cb(const sensor_msgs::msg::Image::SharedPtr img);
  void inference_by_usb_camera();
  void inference_by_image();

  int pub_depth_image(const pub_data_t &);
  int pub_pointcloud2(const pub_data_t &);
  int pub_visual_image(const pub_data_t &);
  int pub_rectified_image(const pub_data_t &);

  void pub_sub_configuration();

  std::atomic_bool is_running_;

  std::vector<std::shared_ptr<std::thread>> work_thread_;

  blockqueue<inference_data_t> inference_que_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr stereo_image_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_, visual_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud2_pub_;

 private:
  std::shared_ptr<StereonetProcess> stereonet_process_;

 private:
  bool save_image_;

  int depth_w_, depth_h_;
  int model_input_w_, model_input_h_;
  float camera_cx, camera_cy, camera_fx, camera_fy, base_line;
  bool need_rectify_, need_pcl_filter_;

  int origin_image_width_, origin_image_height_;
  float height_min_, height_max_;
  std::string stereonet_model_file_path_,
      stereo_image_topic_,
      local_image_path_,
      stereo_calib_file_path_;
  int stereo_combine_mode_ = 1;
  float leaf_size_, stdv_;
  int KMean_;
  void convert_depth(pub_data_t &pub_raw_data);

  std::string rectified_image_topic_ = "~/rectified_image";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
    rectified_image_pub_ = nullptr;
  bool pub_rectified_bgr_ = false;

  int visual_alpha_ = 2, visual_beta_ = 0;

 private:
  std::vector<std::shared_ptr<StereoRectify>> stereo_rectify_list_;
};
}
#endif //STEREONET_MODEL_INCLUDE_STEREONET_COMPONENT_H_
