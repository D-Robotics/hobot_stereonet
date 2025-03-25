//
// Created by zhy on 7/16/24.
//

#ifndef STEREONET_MODEL_INCLUDE_STEREONET_COMPONENT_H_
#define STEREONET_MODEL_INCLUDE_STEREONET_COMPONENT_H_

#include "blockqueue.h"
#include "stereonet_process.h"
#include "stereo_rectify.h"

#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "image_conversion.h"

namespace fs = std::filesystem;

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
    int origin_width, origin_height;
  };

  struct inference_data_t {
    sub_image left_sub_img;
    sub_image right_sub_img;
  };
  struct pub_data_t {
    sub_image left_sub_img;
    sub_image right_sub_img;
    std::vector<float> points;
    std::vector<float> image_size_points;
    cv::Mat depth_img;
    cv::Mat model_depth_img;
  };

  StereoNetNode(const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions())
      : rclcpp::Node("StereoNetNode", node_options) {
    is_running_ = false;
    parameter_configuration();
    pub_sub_configuration();
    if (start() != 0) {
      RCLCPP_FATAL(get_logger(), "Node start failed");
    }
    userColor_.create(256, 1, CV_8UC3); // 256 × 1 的 CV_8UC3 矩阵
    userColor_.at<cv::Vec3b>(0) = 0;
    int s;
    cv::Vec3b color;
    for (s = 1; s < 32; s++) {
      color[0] = 128 + 4 * s;
      color[1] = 0;
      color[2] = 0;
      userColor_.at<cv::Vec3b>(s) = color;
    }
    color[0] = 255;
    color[1] = 0;
    color[2] = 0;
    userColor_.at<cv::Vec3b>(32) = color;
    for (s = 0; s < 63; s++) {
      color[0] = 255;
      color[1] = 4+4*s;
      color[2] = 0;
      userColor_.at<cv::Vec3b>(s + 33) = color;
    }
    color[0] = 254;
    color[1] = 255;
    color[2] = 2;
    userColor_.at<cv::Vec3b>(96) = color;
    for (s = 0; s < 62; s++) {
      color[0] = 250 - 4 * s;
      color[1] = 255;
      color[2] = 6+4*s;
      userColor_.at<cv::Vec3b>(s + 97) = color;
    }
    color[0] = 1;
    color[1] = 255;
    color[2] = 254;
    userColor_.at<cv::Vec3b>(159) = color;
    for (s = 0; s < 64; s++) {
      color[0] = 0;
      color[1] = 252 - (s * 4);
      color[2] = 255;
      userColor_.at<cv::Vec3b>(s+160) = color;

    }
    for (s = 0; s < 32; s++) {
      color[0] = 0;
      color[1] = 0;
      color[2] = 252-4*s;
      userColor_.at<cv::Vec3b>(s+224) = color;
    }
    rclcpp::on_shutdown([this]() {
      stop();
    });
  }

  ~StereoNetNode() {
    stop();
  }

  void parameter_configuration();
  void camera_config_parse(const std::string &file_path,
                           int model_input_w, int model_input_h);

  int inference(inference_data_t &, std::vector<float> &points);
  void inference_func();
  void pub_func(pub_data_t &pub_raw_data);

  int start();
  int stop();

  void stereo_image_cb(const sensor_msgs::msg::Image::SharedPtr img);
  void inference_by_usb_camera();
  void inference_by_image();

  int pub_depth_image(const pub_data_t &);
  int pub_pointcloud2(const pub_data_t &);
  int pub_visual_image(pub_data_t &);
  int pub_rectified_image(const pub_data_t &);
  int pub_depth_camera_info(const pub_data_t &);

  void pub_sub_configuration();

  void dump_one_point_disparity(pub_data_t &pub_raw_data,
                                const cv::Mat &right_image, int x, int y);

  std::atomic_bool is_running_ = false;

  std::vector<std::shared_ptr<std::thread>> work_thread_;

  blockqueue<inference_data_t> inference_que_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr stereo_image_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_, visual_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud2_pub_;

  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_camera_info_pub_;

 private:
  std::shared_ptr<StereonetProcess> stereonet_process_;
  void save_images_with_nv12(cv::Mat &left_img, cv::Mat &right_img, const std::string &image_format);
  void save_mat_to_bin(const cv::Mat &mat, const std::string &filename);

 private:
  bool save_image_;
  std::string save_dir_ = "./stereonet_images";

  std::mutex mtx_;
  bool save_image_all_;
  bool save_image_to_nv12_;
  int save_cnt_ = 0;
  int save_freq_ = 1;
  int save_total_ = -1;

  std::string postprocess_;

  int depth_w_, depth_h_;
  int model_input_w_, model_input_h_;
  float camera_cx, camera_cy, camera_fx, camera_fy, base_line;
  bool need_rectify_, need_pcl_filter_;

  int origin_image_width_, origin_image_height_;
  float height_min_, height_max_;
  std::string stereonet_model_file_path_,
      stereo_image_topic_,
      local_image_path_,
      stereo_calib_file_path_,
      camera_info_topic_;
  int stereo_combine_mode_ = 1;
  float leaf_size_, stdv_;
  int KMean_;
  void convert_depth(pub_data_t &pub_raw_data);

  std::string rectified_image_topic_ = "~/rectified_image";
  std::string rectified_right_image_topic_ = "~/rectified_right_image";
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
      rectified_image_pub_ = nullptr, rectified_right_image_pub_ = nullptr;
  bool pub_rectified_bgr_ = false;

  int visual_alpha_ = 1, visual_beta_ = 0;
  int max_disp_ = 192;
  int image_inference_sleep_ms_ = 1;

  bool depth_type_point_ = true;
  std::string image_format_ = "png";

  int render_type_ = 0;
  bool render_need_filter_ = true;
  uint16_t render_max_depth_ = 10000;

  bool depth_need_filter_ = true;
  float pc_max_depth_ = 6.0;

  float uncertainty_th_ = -0.09;

 private:
  std::vector<std::shared_ptr<StereoRectify>> stereo_rectify_list_;

 private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::CompressedImage,
      //sensor_msgs::msg::Image,
      sensor_msgs::msg::CompressedImage
  >;
  message_filters::Subscriber<sensor_msgs::msg::CompressedImage> depth_subscriber_;
  message_filters::Subscriber<sensor_msgs::msg::Image> color_subscriber_;
  message_filters::Subscriber<sensor_msgs::msg::CompressedImage> compare_left_subscriber_;

  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  void sync_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& depth_msg,
      //const sensor_msgs::msg::Image::ConstSharedPtr& color_msg,
                     const sensor_msgs::msg::CompressedImage::ConstSharedPtr &rs_left_msg);

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr compare_visual_image_pub_;

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compare_left_sub_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  
  void d_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg);
  void c_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg);
  void camera_info_cb(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info_msg);

  cv::Mat compare_visual_;
  std::mutex compare_visual_mtx_;
  bool depth_compare = false;

 private:
  cv::Mat userColor_;
};
}
#endif //STEREONET_MODEL_INCLUDE_STEREONET_COMPONENT_H_
