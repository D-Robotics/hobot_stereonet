//
// Created by zhy on 7/1/24.
//

#include "blockqueue.h"
#include "stereonet_process.h"

#include <fstream>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cv_bridge/cv_bridge.h>


class StereoNetNode : public rclcpp::Node {
 public:

  enum sub_image_type {
    BGR,
    NV12
  };

  struct sub_image {
    std::string frame_id;
    cv::Mat image;
    sub_image_type image_type;
  };

  using inference_data_t = std::tuple<sub_image, sub_image, double>;
  using pub_data_t = std::tuple<sub_image, double, std::vector<float>>;

  StereoNetNode() : rclcpp::Node("StereoNetNode") {
    parameter_configuration();
    pub_sub_configuration();
  }

  void parameter_configuration();

  int inference(const inference_data_t&, std::vector<float> &points);
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

  void pub_sub_configuration();

  std::atomic_bool is_running_;

  std::vector<std::shared_ptr<std::thread>> work_thread_;

  blockqueue<inference_data_t> inference_que_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr stereo_image_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_,visual_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud2_pub_;

 private:
  std::shared_ptr<StereonetProcess> stereonet_process_;

 private:
  int depth_w_, depth_h_;
  int model_input_w_, model_input_h_;
  bool point_with_color_, pub_cloud_;
  float camera_cx, camera_cy, camera_fx, camera_fy, base_line;
  float depth_min_, depth_max_;
  std::string stereonet_model_file_path_ =
      "/mnt/nfs171/x5/cc_ws/tros_ws/install/share/stereonet_model/config/model.hbm",
  stereo_image_topic_,
  local_image_path_;
  int stereo_combine_mode_ = 1;
};

int StereoNetNode::inference(const inference_data_t& inference_data,
    std::vector<float> &points) {
  bool is_nv12;
  cv::Mat resized_left_img, resized_right_img;
  const cv::Mat &left_img = std::get<0>(inference_data).image;
  const cv::Mat &right_img = std::get<1>(inference_data).image;
  const double ts = std::get<2>(inference_data);
  is_nv12 = std::get<0>(inference_data).image_type == sub_image_type::NV12;
  if (is_nv12) {
    if (left_img.rows * 2 / 3 != model_input_h_ || left_img.cols != model_input_w_
     || right_img.rows * 2 / 3 != model_input_h_ || right_img.cols != model_input_w_) {
      RCLCPP_FATAL(this->get_logger(), "when encoding of image is nv12, "
                                       "the size(%d, %d) of image MUST equal to size(%d, %d) of model",
                   left_img.cols, left_img.rows,
                   model_input_w_, model_input_h_);
      return -1;
    }
    resized_left_img = left_img;
    resized_right_img = right_img;
  } else {
    if (left_img.rows != model_input_h_ || left_img.cols != model_input_w_) {
      cv::resize(left_img, resized_left_img, cv::Size(model_input_w_, model_input_h_));
      cv::resize(right_img, resized_right_img, cv::Size(model_input_w_, model_input_h_));
    } else {
      resized_left_img = left_img;
      resized_right_img = right_img;
    }
  }
  return stereonet_process_->stereonet_inference(resized_left_img, resized_right_img,
      is_nv12, points);
}

int StereoNetNode::pub_depth_image(const pub_data_t &pub_raw_data) {
  cv_bridge::CvImage img_bridge;
  std_msgs::msg::Header image_header;
  sensor_msgs::msg::Image depth_img_msg;
  const cv::Mat &image = std::get<0>(pub_raw_data).image;
  const double ts = std::get<1>(pub_raw_data);
  const std::vector<float> &points = std::get<2>(pub_raw_data);
  int img_origin_width;
  int img_origin_height;

  if (depth_image_pub_->get_subscription_count() < 1) return 0;

  if (std::get<0>(pub_raw_data).image_type == sub_image_type::NV12) {
    img_origin_width = image.cols;
    img_origin_height = image.rows * 2 / 3;
  } else {
    img_origin_width = image.cols;
    img_origin_height = image.rows;
  }

  assert(points.size() == img_origin_height * img_origin_width);
  cv::Mat depth_img(depth_h_, depth_w_, CV_16SC1);

  for (int y = 0; y < img_origin_height; ++y) {
    for (int x = 0; x < img_origin_width; ++x) {
      depth_img.at<uint16_t>(y, x) = 1000 *
          (camera_fx * base_line) / points[y * img_origin_width + x];
    }
  }
  image_header.frame_id = std::get<0>(pub_raw_data).frame_id;
  image_header.stamp.sec = ts;
  image_header.stamp.nanosec = (ts - image_header.stamp.sec) * 1e9;
  img_bridge = cv_bridge::CvImage(image_header, "mono16", depth_img);
  img_bridge.toImageMsg(depth_img_msg);
  depth_image_pub_->publish(depth_img_msg);
  return 0;
}

int StereoNetNode::pub_visual_image(const pub_data_t &pub_raw_data) {
  cv_bridge::CvImage img_bridge;
  std_msgs::msg::Header image_header;
  sensor_msgs::msg::Image visual_img_msg;
  const cv::Mat &image = std::get<0>(pub_raw_data).image;
  const double ts = std::get<1>(pub_raw_data);
  const std::vector<float> &points = std::get<2>(pub_raw_data);
  cv::Mat bgr_image;

  if (visual_image_pub_->get_subscription_count() < 1) return 0;

  if (std::get<0>(pub_raw_data).image_type == sub_image_type::NV12) {
    cv::cvtColor(image, bgr_image, cv::COLOR_YUV2BGR_NV12);
  } else {
    bgr_image = image;
  }

  cv::Mat visual_img(bgr_image.rows * 2, bgr_image.cols, CV_8UC3);
  bgr_image.copyTo(visual_img(cv::Rect(0, 0, bgr_image.cols, bgr_image.rows)));

  cv::Mat feat_mat(bgr_image.rows, bgr_image.cols, CV_32F, const_cast<float*>(points.data()));
  cv::Mat feat_visual;
  feat_mat.convertTo(feat_visual, CV_8U, 1, 0);
  cv::convertScaleAbs(feat_visual, feat_visual, 2);
  cv::applyColorMap(feat_visual,
      visual_img(cv::Rect(0, bgr_image.rows, bgr_image.cols, bgr_image.rows)),
      cv::COLORMAP_JET);

  image_header.frame_id = std::get<0>(pub_raw_data).frame_id;
  image_header.stamp.sec = ts;
  image_header.stamp.nanosec = (ts - image_header.stamp.sec) * 1e9;
  img_bridge = cv_bridge::CvImage(image_header, "bgr8", visual_img);
  img_bridge.toImageMsg(visual_img_msg);
  visual_image_pub_->publish(visual_img_msg);
  return 0;
}

int StereoNetNode::pub_pointcloud2(const pub_data_t &pub_raw_data) {
  int points_size = 0;
  const cv::Mat &image = std::get<0>(pub_raw_data).image;
  const double ts = std::get<1>(pub_raw_data);
  const std::vector<float> &points = std::get<2>(pub_raw_data);

  if (pointcloud2_pub_->get_subscription_count() < 1) return 0;

  sensor_msgs::msg::PointCloud2 point_cloud_msg;
  sensor_msgs::PointCloud2Modifier modifier(point_cloud_msg);

  int img_origin_width;
  int img_origin_height;

  if (std::get<0>(pub_raw_data).image_type == sub_image_type::NV12) {
    img_origin_width = image.cols;
    img_origin_height = image.rows * 2 / 3;
  } else {
    img_origin_width = image.cols;
    img_origin_height = image.rows;
  }

  modifier.resize(points.size() / 4);
  modifier.setPointCloud2Fields(3,
                                "x", 1, sensor_msgs::msg::PointField::FLOAT32,
                                "y", 1, sensor_msgs::msg::PointField::FLOAT32,
                                "z", 1, sensor_msgs::msg::PointField::FLOAT32);
  sensor_msgs::PointCloud2Iterator<float> iter_x(point_cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(point_cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(point_cloud_msg, "z");
  for (int y = 0; y < img_origin_height; y += 2) {
    for (int x = 0; x < img_origin_width; x += 2) {
      float depth = (camera_cx * base_line) / points[y * img_origin_width + x];
      //if (depth < depth_min_ || depth > depth_max_) continue;

      float X = (x - camera_cx) / camera_fx * depth;
      float Y = (y - camera_cy) / camera_fy * depth;

      *iter_x = X, *iter_y = Y, *iter_z = depth;
      ++iter_x, ++iter_y, ++iter_z;
      ++points_size;
    }
  }
  point_cloud_msg.header.frame_id = std::get<0>(pub_raw_data).frame_id;
  point_cloud_msg.header.stamp.sec = ts ;
  point_cloud_msg.header.stamp.nanosec = (ts - point_cloud_msg.header.stamp.sec) * 1e9;
  point_cloud_msg.height = 1;
  point_cloud_msg.is_dense = false;
  //point_cloud_msg.width = points_size;
  //point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width;
  pointcloud2_pub_->publish(point_cloud_msg);
  return 0;
}
//
//int StereoNetNode::pub_pointcloud2(const pub_data_t &pub_raw_data) {
//  const cv::Mat &image = std::get<0>(pub_raw_data).image;
//  const double ts = std::get<1>(pub_raw_data);
//  const std::vector<float> &points = std::get<2>(pub_raw_data);
//  std::vector<float> points_xyz;
//  int img_origin_width = image.cols;
//  int img_origin_height = image.rows;
//  sensor_msgs::msg::PointCloud2 point_cloud_msg;
//
//  point_cloud_msg.fields.resize(3);
//
//  point_cloud_msg.fields[0].name = "x";
//  point_cloud_msg.fields[0].offset = 0;
//  point_cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
//  point_cloud_msg.fields[0].count = 1;
//
//  point_cloud_msg.fields[1].name = "y";
//  point_cloud_msg.fields[1].offset = 4;
//  point_cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
//  point_cloud_msg.fields[1].count = 1;
//
//  point_cloud_msg.fields[2].name = "z";
//  point_cloud_msg.fields[2].offset = 8;
//  point_cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
//  point_cloud_msg.fields[2].count = 1;
//
//  points_xyz.reserve(points.size());
//
//  for (int y = 0; y < img_origin_height; ++y) {
//    for (int x = 0; x < img_origin_width; ++x) {
//      float depth = (camera_cx * base_line) / points[y * img_origin_width + x];
//      if (depth < depth_min_ || depth > depth_max_) continue;
//      float X = (x - camera_cx) / camera_fx * depth;
//      float Y = (y - camera_cy) / camera_fy * depth;
//      points_xyz.emplace_back(X);
//      points_xyz.emplace_back(Y);
//      points_xyz.emplace_back(depth);
//    }
//  }
//
//  point_cloud_msg.height = 1;
//  point_cloud_msg.is_bigendian = false;
//  point_cloud_msg.point_step = 12;
//  point_cloud_msg.is_dense = false;
//  point_cloud_msg.width = points_xyz.size() / 3;
//  point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width;
//  point_cloud_msg.data.resize(point_cloud_msg.row_step * point_cloud_msg.height);
//
//
//  std::memcpy(point_cloud_msg.data.data(), points_xyz.data(), points_xyz.size() * 4);
//
//  pointcloud2_pub_->publish(point_cloud_msg);
//  return 0;
//}

void StereoNetNode::stereo_image_cb(const sensor_msgs::msg::Image::SharedPtr img) {
  double ts;
  cv::Mat stereo_img, left_img, right_img;
  sub_image left_sub_img, right_sub_img;
  const std::string &encoding = img->encoding;
  int stereo_img_width, stereo_img_height;

  RCLCPP_DEBUG(this->get_logger(),
      "Sub a stereo image, encoding: %s, width: %d, height: %d",
              encoding.c_str(), img->width, img->height);
  if (stereo_combine_mode_ == 0) {
    stereo_img_width = img->width / 2;
    stereo_img_height = img->height;
  } else if (stereo_combine_mode_ == 1) {
    stereo_img_width = img->width;
    stereo_img_height = img->height / 2;
  }

//  std::ofstream yuv("stereo.yuv", std::ios::out | std::ios::binary);
//  yuv.write(reinterpret_cast<const char *>(img->data.data()), img->width * img->height * 3/2);
//  std::exit(0);
  if (encoding == "nv12" || encoding == "NV12") {
    int ylen = stereo_img_height * stereo_img_width;
    int uvlen = ylen / 2;
    if (stereo_combine_mode_ == 0) {
      RCLCPP_FATAL(this->get_logger(),
          "when stereo_combine_mode is 0, the encoding of image must be bgr8!");
      return;
    }
    left_sub_img.image_type = sub_image_type::NV12;
    right_sub_img.image_type = sub_image_type::NV12;
    left_sub_img.image = cv::Mat(stereo_img_height * 3 / 2, stereo_img_width,CV_8UC1);
    right_sub_img.image = cv::Mat(stereo_img_height * 3 / 2, stereo_img_width,CV_8UC1);
    std::memcpy(left_sub_img.image.data, img->data.data(), ylen);
    std::memcpy(right_sub_img.image.data, img->data.data() + ylen, ylen);
    std::memcpy(left_sub_img.image.data + ylen, img->data.data() + 2 * ylen, uvlen);
    std::memcpy(right_sub_img.image.data + ylen, img->data.data() + 2 * ylen + uvlen, uvlen);
  } else if (encoding == "bgr8" || encoding == "BGR8") {
    stereo_img = cv_bridge::toCvShare(img)->image;
    if (stereo_combine_mode_ == 0) {
      left_img = stereo_img(
          cv::Rect(0, 0, stereo_img_width, stereo_img_height)).clone();
      right_img = stereo_img(
          cv::Rect(stereo_img_width, 0, stereo_img_width, stereo_img_height)).clone();
    } else if (stereo_combine_mode_ == 1) {
      left_img = stereo_img(
          cv::Rect(0, 0, stereo_img_width, stereo_img_height)).clone();
      right_img = stereo_img(
          cv::Rect(0, stereo_img_height, stereo_img_width, stereo_img_height)).clone();
    }
    left_sub_img.image_type = sub_image_type::BGR;
    right_sub_img.image_type = sub_image_type::BGR;
    left_sub_img.image = left_img;
    right_sub_img.image = right_img;
  }

  left_sub_img.frame_id = img->header.frame_id;
  right_sub_img.frame_id = img->header.frame_id;

  ts = img->header.stamp.sec;
  ts += img->header.stamp.nanosec * 1e-9;

  inference_data_t inference_data = std::make_tuple(left_sub_img, right_sub_img, ts);
  if (inference_que_.size() > 5) {
    RCLCPP_WARN(this->get_logger(), "inference que is full!");
    return;
  }
  inference_que_.put(inference_data);
}

void StereoNetNode::inference_func() {
  int ret = 0;
  while (is_running_) {
    inference_data_t inference_data;
    std::vector<float> points;
    if (inference_que_.get(inference_data)) {
      ret = inference(inference_data, points);
      if (ret != 0) {
        RCLCPP_ERROR(this->get_logger(), "inference failed.");
      } else {
        const sub_image& left_sub_img = std::get<0>(inference_data);
        const cv::Mat &left_img = left_sub_img.image;
        const double ts = std::get<2>(inference_data);
        pub_data_t pub_data = std::make_tuple(left_sub_img, ts, points);
        pub_func(pub_data);
      }
    }
  }
  inference_que_.clear();
}

void mapping_resolution(StereoNetNode::pub_data_t &pub_raw_data,
    int depth_w, int depth_h) {
  int img_origin_width, img_origin_height;
  const cv::Mat &image = std::get<0>(pub_raw_data).image;
  std::vector<float> &points = std::get<2>(pub_raw_data);
  std::vector<float> resized_points;
  if (std::get<0>(pub_raw_data).image_type == StereoNetNode::sub_image_type::NV12) {
    img_origin_width = image.cols;
    img_origin_height = image.rows * 2 / 3;
  } else {
    img_origin_width = image.cols;
    img_origin_height = image.rows;
  }
  if (img_origin_width != depth_w || img_origin_height != depth_h) {
    resized_points.resize(img_origin_width * img_origin_height);
    cv::Mat resized_mat(img_origin_height, img_origin_width, CV_32FC1,
        resized_points.data());
    cv::Mat origin_mat(depth_h, depth_w, CV_32FC1, points.data());
    cv::resize(origin_mat, resized_mat,
        cv::Size(img_origin_width, img_origin_height));
    points = std::move(resized_points);
  }
}

void StereoNetNode::pub_func(pub_data_t &pub_raw_data) {
  int ret = 0;
  {
    ScopeProcessTime t("mapping_resolution");
    mapping_resolution(pub_raw_data, depth_w_, depth_h_);
  }
  {
    ScopeProcessTime t("pub_depth_image");
    ret = pub_depth_image(pub_raw_data);
  }
  {
    ScopeProcessTime t("pub_pointcloud2");
    ret = pub_pointcloud2(pub_raw_data);
  }
  {
    ScopeProcessTime t("pub_visual");
    ret = pub_visual_image(pub_raw_data);
  }
  if (ret != 0) {
    RCLCPP_ERROR(this->get_logger(), "pub failed, ret: %d", ret);
  }
}

int StereoNetNode::start() {
  int ret = 0;
  stereonet_process_ = std::make_shared<StereonetProcess>();
  ret = stereonet_process_->stereonet_init(stereonet_model_file_path_);
  if (ret != 0) {
    RCLCPP_FATAL(this->get_logger(), "stereonet model init failed");
    stereonet_process_ = nullptr;
    return ret;
  } else {
    RCLCPP_INFO(this->get_logger(), "stereonet model init successed");
  }
  stereonet_process_->get_input_width_height(model_input_w_, model_input_h_);
  stereonet_process_->get_depth_width_height(depth_w_, depth_h_);
  is_running_ = true;
  work_thread_.emplace_back(std::make_shared<std::thread>(
      [this] { inference_func(); }));
  work_thread_.emplace_back(std::make_shared<std::thread>(
      [this] { inference_func(); }));

  return 0;
}

int StereoNetNode::stop() {
  is_running_ = false;
  for (auto &t : work_thread_) {
    t->join();
  }
  work_thread_.clear();
  stereonet_process_->stereonet_deinit();
  stereonet_process_ = nullptr;
  return 0;
}

void StereoNetNode::parameter_configuration() {
  this->declare_parameter("camera_cx", 480.0f);
  this->get_parameter("camera_cx", camera_cx);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_cx: " << camera_cx);

  this->declare_parameter("camera_fx", 300.0f);
  this->get_parameter("camera_fx", camera_fx);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_fx: " << camera_fx);

  this->declare_parameter("camera_cy", 270.0f);
  this->get_parameter("camera_cy", camera_cy);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_cy: " << camera_cy);

  this->declare_parameter("camera_fy", 300.0f);
  this->get_parameter("camera_fy", camera_fy);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_fy: " << camera_fy);

  this->declare_parameter("base_line", 0.06f);
  this->get_parameter("base_line", base_line);
  RCLCPP_INFO_STREAM(this->get_logger(), "base_line: " << base_line);

  this->declare_parameter("point_with_color", false);
  this->get_parameter("point_with_color", point_with_color_);
  RCLCPP_INFO_STREAM(this->get_logger(), "point_with_color_: " << point_with_color_);

  this->declare_parameter("stereonet_model_file_path", stereonet_model_file_path_);
  this->get_parameter("stereonet_model_file_path", stereonet_model_file_path_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereonet_model_file_path_: " << stereonet_model_file_path_);

  this->declare_parameter("stereo_image_topic", stereo_image_topic_);
  this->get_parameter("stereo_image_topic", stereo_image_topic_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereo_image_topic: " << stereo_image_topic_);

  this->declare_parameter("local_image_path", local_image_path_);
  this->get_parameter("local_image_path", local_image_path_);
  RCLCPP_INFO_STREAM(this->get_logger(), "local_image_path_: " << local_image_path_);

  this->declare_parameter("depth_min", depth_min_);
  this->get_parameter("depth_min", depth_min_);
  RCLCPP_INFO_STREAM(this->get_logger(), "depth_min_: " << depth_min_);

  this->declare_parameter("depth_max", depth_max_);
  this->get_parameter("depth_max", depth_max_);
  RCLCPP_INFO_STREAM(this->get_logger(), "depth_max: " << depth_max_);

  this->declare_parameter("stereo_combine_mode", stereo_combine_mode_);
  this->get_parameter("stereo_combine_mode", stereo_combine_mode_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereo_combine_mode: " << stereo_combine_mode_);
}

void StereoNetNode::inference_by_usb_camera() {
/*
  cv::VideoCapture *capture = nullptr;
  cv::Mat stereo_img, left_img, right_img;
  cv::Rect right_rect(0, 0, 640, 360),
           left_rect(0, 360, 640, 360);
  uint64_t current_ts;
  double ts;

  capture = new cv::VideoCapture("/dev/video0");
  assert(capture != nullptr);
  int fps = capture->get(cv::CAP_PROP_FPS);
  int width = capture->get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture->get(cv::CAP_PROP_FRAME_HEIGHT);

  RCLCPP_INFO(this->get_logger(),
      "usb camera fps: %d, width: %d, height: %d",
      fps, width, height);

  while (rclcpp::ok()) {
    if (capture->grab()) {
      current_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
      capture->retrieve(stereo_img);
      left_img = stereo_img(left_rect);
      right_img = stereo_img(right_rect);
      ts = current_ts * 1e-9;
      inference_data_t inference_data = std::make_tuple(left_img, right_img, ts);
      inference(inference_data);
    }
  }
  delete capture;
*/
}

void StereoNetNode::inference_by_image() {
  if (inference_que_.size() > 5) {
    RCLCPP_WARN(this->get_logger(), "inference que is full!");
    return;
  }
  sub_image left_sub_img, right_sub_img;
  double ts;
  uint64_t current_ts;
  std::vector<float> points;
  cv::Mat left_img = cv::imread(local_image_path_ + "/left.png");
  cv::Mat right_img = cv::imread(local_image_path_ + "/right.png");
  current_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  ts = current_ts * 1e-9;
  left_sub_img.image_type = sub_image_type::BGR;
  right_sub_img.image_type = sub_image_type::BGR;
  left_sub_img.image = left_img;
  right_sub_img.image = right_img;
  left_sub_img.frame_id = "default_cam";
  right_sub_img.frame_id = "default_cam";
  inference_data_t inference_data = std::make_tuple(left_sub_img, right_sub_img, ts);
  inference_que_.put(inference_data);
}

void StereoNetNode::pub_sub_configuration() {
  stereo_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      stereo_image_topic_, 10,
      std::bind(&StereoNetNode::stereo_image_cb, this, std::placeholders::_1));

  pointcloud2_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "~/stereonet_pointcloud2", 10);

  depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "~/stereonet_depth", 10);

  visual_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "~/stereonet_visual", 10);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<StereoNetNode>();
  RCLCPP_INFO(node->get_logger(), "This is example of D-Robotics stereonet");

  bool use_usb_camera;
  node->declare_parameter("use_usb_camera", false);
  node->get_parameter("use_usb_camera", use_usb_camera);

  bool use_local_image;
  node->declare_parameter("use_local_image", false);
  node->get_parameter("use_local_image", use_local_image);

  if (node->start() != 0) {
    RCLCPP_FATAL(node->get_logger(), "Node start failed");
    return -1;
  }
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

  node->stop();
  node = nullptr;
  return 0;
}
