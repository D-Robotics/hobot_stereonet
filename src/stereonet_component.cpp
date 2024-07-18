//
// Created by zhy on 7/1/24.
//

#include <rclcpp_components/register_node_macro.hpp>
#include "stereonet_component.h"
namespace stereonet {
int StereoNetNode::inference(const inference_data_t &inference_data,
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

  cv::Mat feat_mat(bgr_image.rows, bgr_image.cols, CV_32F, const_cast<float *>(points.data()));
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
  point_cloud_msg.header.stamp.sec = ts;
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

void stereo_rectify(
    const cv::Mat &left_image,
    const cv::Mat &right_image,
    int origin_image_width, int origin_image_height,
    cv::Mat &Kl, cv::Mat &Kr, cv::Mat &Dl, cv::Mat &Dr, cv::Mat &R_rl, cv::Mat &t_rl,
    cv::Mat &rectified_left_image, cv::Mat &rectified_right_image,
    float &rectified_fx, float &rectified_cx, float &rectified_fy, float &rectified_cy, float &baseline) {
  cv::Mat Rl, Rr, Pl, Pr, Q;
  cv::Mat undistmap1l, undistmap2l, undistmap1r, undistmap2r;

  int width = left_image.cols;
  int height = left_image.rows;

  cv::stereoRectify(Kl, Dl, Kr, Dr,
                    cv::Size(width, height), R_rl, t_rl, Rl, Rr, Pl, Pr, Q,
                    cv::CALIB_ZERO_DISPARITY, 0);

  cv::initUndistortRectifyMap(Kl, Dl, Rl, Pl, cv::Size(width, height), CV_32FC1, undistmap1l, undistmap2l);
  cv::initUndistortRectifyMap(Kr, Dr, Rr, Pr, cv::Size(width, height), CV_32FC1, undistmap1r, undistmap2r);

  cv::remap(left_image, rectified_left_image, undistmap1l, undistmap2l, cv::INTER_LINEAR);
  cv::remap(right_image, rectified_right_image, undistmap1r, undistmap2r, cv::INTER_LINEAR);

  rectified_fx = Pl.at<double>(0, 0);
  rectified_cx = Pl.at<double>(0, 2);
  rectified_fy = Pl.at<double>(1, 1);
  rectified_cy = Pl.at<double>(1, 2);
  const cv::Mat t = Rr * t_rl;
  baseline = std::abs(t.at<double>(0, 0));
}

void dump_rectified_image(cv::Mat &left_img, cv::Mat &right_img,
                          cv::Mat &rectified_left_img, cv::Mat &rectified_right_img) {
  cv::Mat img_src, img_rtf;
  cv::hconcat(left_img, right_img, img_src);
  cv::hconcat(rectified_left_img, rectified_right_img, img_rtf);
  for (int i = 0; i < 10; ++i) {
    cv::Point a, b;
    a.x = 0;
    a.y = img_rtf.rows / 10 * i;
    b.x = img_rtf.cols;
    b.y = img_rtf.rows / 10 * i;
    cv::line(img_rtf, a, b, cv::Scalar(0, 255, 0), 2);
  }
  cv::imwrite("./before.jpg", img_src);
  cv::imwrite("./after.jpg", img_rtf);
}

void save_images(cv::Mat &left_img, cv::Mat &right_img, uint64_t ts) {
  static std::atomic_bool directory_created{false};
  if (!directory_created) {
    directory_created = true;
    system("mkdir -p ./images/cam0/data/ ./images/cam1/data/");
  }
  cv::imwrite("./images/cam0/data/" + std::to_string(ts) + ".png", left_img);
  cv::imwrite("./images/cam1/data/" + std::to_string(ts) + ".png", right_img);
}

void StereoNetNode::stereo_image_cb(const sensor_msgs::msg::Image::SharedPtr img) {
  double now = std::chrono::high_resolution_clock::now().time_since_epoch().count() * 1e-9;
  double ts;
  cv::Mat stereo_img, left_img, right_img;
  sub_image left_sub_img, right_sub_img;
  const std::string &encoding = img->encoding;
  int stereo_img_width, stereo_img_height;
  ts = img->header.stamp.sec;
  ts += img->header.stamp.nanosec * 1e-9;
  RCLCPP_DEBUG(this->get_logger(),
              "we received stereo msg at: %f, timestamp of stereo is: %f, latency is %f, "
              "encoding: %s, width: %d, height: %d",
              now, ts, now - ts, encoding.c_str(), img->width, img->height);
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
    left_sub_img.image = cv::Mat(stereo_img_height * 3 / 2, stereo_img_width, CV_8UC1);
    right_sub_img.image = cv::Mat(stereo_img_height * 3 / 2, stereo_img_width, CV_8UC1);
    std::memcpy(right_sub_img.image.data, img->data.data(), ylen);
    std::memcpy(left_sub_img.image.data, img->data.data() + ylen, ylen);
    std::memcpy(right_sub_img.image.data + ylen, img->data.data() + 2 * ylen, uvlen);
    std::memcpy(left_sub_img.image.data + ylen, img->data.data() + 2 * ylen + uvlen, uvlen);


    {
      ScopeProcessTime t("nv12->bgr");
      left_sub_img.image_type = sub_image_type::BGR;
      right_sub_img.image_type = sub_image_type::BGR;
      cv::cvtColor(left_sub_img.image, left_sub_img.image, CV_YUV2BGR_NV12);
      cv::cvtColor(right_sub_img.image, right_sub_img.image, CV_YUV2BGR_NV12);
    }

  } else if (encoding == "bgr8" || encoding == "BGR8") {
    {
      ScopeProcessTime t("cv_bridge::toCvShare");
      stereo_img = cv_bridge::toCvShare(img)->image;
    }
    {
      ScopeProcessTime t("stereo_img split and clone");
      if (stereo_combine_mode_ == 0) {
        left_img = stereo_img(
            cv::Rect(0, 0, stereo_img_width, stereo_img_height)).clone();
        right_img = stereo_img(
            cv::Rect(stereo_img_width, 0, stereo_img_width, stereo_img_height)).clone();
      } else if (stereo_combine_mode_ == 1) {
        right_img = stereo_img(
            cv::Rect(0, 0, stereo_img_width, stereo_img_height)).clone();
        left_img = stereo_img(
            cv::Rect(0, stereo_img_height, stereo_img_width, stereo_img_height)).clone();
      }
    }
    left_sub_img.image_type = sub_image_type::BGR;
    right_sub_img.image_type = sub_image_type::BGR;
    left_sub_img.image = left_img;
    right_sub_img.image = right_img;
    if (save_image_) {
      save_images(left_sub_img.image, right_sub_img.image, ts * 1e9);
      return;
    }
  }

  left_sub_img.frame_id = img->header.frame_id;
  right_sub_img.frame_id = img->header.frame_id;

  inference_data_t inference_data = std::make_tuple(left_sub_img, right_sub_img, ts);
  if (inference_que_.size() > 5) {
    RCLCPP_WARN(this->get_logger(), "inference que is full!");
    return;
  }
  inference_que_.put(inference_data);
}

void StereoNetNode::inference_func() {
  int ret = 0;
  cv::Mat rectified_left_image, rectified_right_image;
  while (is_running_) {
    inference_data_t inference_data;
    std::vector<float> points;
    if (inference_que_.get(inference_data)) {
      if (need_rectify_) {
        cv::Mat &left_image = std::get<0>(inference_data).image;
        cv::Mat &right_image = std::get<1>(inference_data).image;
        ScopeProcessTime t("stereo_rectify");
        stereo_rectify(left_image, right_image,
                       origin_image_width_, origin_image_height_,
                       Kl, Kr, Dl, Dr, R_rl, t_rl,
                       rectified_left_image, rectified_right_image,
                       camera_fx, camera_cx, camera_fy, camera_cy, base_line);
        left_image = rectified_left_image;
        right_image = rectified_right_image;
        RCLCPP_INFO_ONCE(this->get_logger(),
            "rectified fx: %f, fy: %f, cx: %f, cy: %f, base_line: :%f",
            camera_fx, camera_fy, camera_cx, camera_cy, base_line);
        //  return dump_rectified_image(left_img, right_img, rectified_left_img, rectified_right_img);
      }

      ret = inference(inference_data, points);
      if (ret != 0) {
        RCLCPP_ERROR(this->get_logger(), "inference failed.");
      } else {
        const sub_image &left_sub_img = std::get<0>(inference_data);
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
  camera_config_parse(stereo_calib_file_path_,
                      model_input_w_, model_input_h_);
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

void StereoNetNode::camera_config_parse(const std::string &file_path,
                                        int model_input_w, int model_input_h) {
  float width_scale, height_scale;
  cv::FileStorage fs(file_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    RCLCPP_WARN_STREAM(this->get_logger(), "Failed to open " << file_path);
    return;
  }

  // Reading cam0 data
  std::vector<double> cam0_distortion_coeffs;
  std::vector<double> cam0_intrinsics;
  std::vector<int> cam0_resolution;

  fs["cam0"]["distortion_coeffs"] >> cam0_distortion_coeffs;
  fs["cam0"]["intrinsics"] >> cam0_intrinsics;
  fs["cam0"]["resolution"] >> cam0_resolution;

  width_scale = model_input_w_ / static_cast<float>(cam0_resolution[0]);
  height_scale = model_input_h_ / static_cast<float>(cam0_resolution[1]);

  // Reading cam1 data
  std::vector<std::vector<double>> cam1_T_cn_cnm1;
  std::vector<double> cam1_distortion_coeffs;
  std::vector<double> cam1_intrinsics;
  std::vector<int> cam1_resolution;

  fs["cam1"]["T_cn_cnm1"] >> cam1_T_cn_cnm1;
  fs["cam1"]["distortion_coeffs"] >> cam1_distortion_coeffs;
  fs["cam1"]["intrinsics"] >> cam1_intrinsics;
  fs["cam1"]["resolution"] >> cam1_resolution;

  fs.release();

  Dl = cv::Mat(1, 4, CV_64F, cam0_distortion_coeffs.data()).clone();
  Kl = cv::Mat::zeros(3, 3, CV_64F);
  Kl.at<double>(0, 0) = cam0_intrinsics[0] * width_scale;
  Kl.at<double>(0, 2) = cam0_intrinsics[2] * width_scale;
  Kl.at<double>(1, 1) = cam0_intrinsics[1] * height_scale;
  Kl.at<double>(1, 2) = cam0_intrinsics[3] * height_scale;
  Kl.at<double>(2, 2) = 1;

  R_rl = cv::Mat::zeros(3, 3, CV_64F);
  t_rl = cv::Mat::zeros(3, 1, CV_64F);

  R_rl.at<double>(0, 0) = cam1_T_cn_cnm1[0][0];
  R_rl.at<double>(0, 1) = cam1_T_cn_cnm1[0][1];
  R_rl.at<double>(0, 2) = cam1_T_cn_cnm1[0][2];
  R_rl.at<double>(1, 0) = cam1_T_cn_cnm1[1][0];
  R_rl.at<double>(1, 1) = cam1_T_cn_cnm1[1][1];
  R_rl.at<double>(1, 2) = cam1_T_cn_cnm1[1][2];
  R_rl.at<double>(2, 0) = cam1_T_cn_cnm1[2][0];
  R_rl.at<double>(2, 1) = cam1_T_cn_cnm1[2][1];
  R_rl.at<double>(2, 2) = cam1_T_cn_cnm1[2][2];

  t_rl.at<double>(0, 0) = cam1_T_cn_cnm1[0][3];
  t_rl.at<double>(1, 0) = cam1_T_cn_cnm1[1][3];
  t_rl.at<double>(2, 0) = cam1_T_cn_cnm1[2][3];

  Dr = cv::Mat(1, 4, CV_64F, cam1_distortion_coeffs.data()).clone();

  Kr = cv::Mat::zeros(3, 3, CV_64F);
  Kr.at<double>(0, 0) = cam1_intrinsics[0] * width_scale;
  Kr.at<double>(0, 2) = cam1_intrinsics[2] * width_scale;
  Kr.at<double>(1, 1) = cam1_intrinsics[1] * height_scale;
  Kr.at<double>(1, 2) = cam1_intrinsics[3] * height_scale;
  Kr.at<double>(2, 2) = 1;
  RCLCPP_INFO_STREAM(this->get_logger(),
      "\nKl: \n" << Kl << "\nDl:\n" << Dl <<
                "\nKr: \n" << Kr << "\nDr:\n" << Dr <<
                "\nR, t: \n" << R_rl << "\n" << t_rl <<
                "\norigin width, height: " << cam0_resolution[0] << ", " << cam0_resolution[1]);
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

  this->declare_parameter("need_rectify", true);
  this->get_parameter("need_rectify", need_rectify_);
  RCLCPP_INFO_STREAM(this->get_logger(), "need_rectify: " << need_rectify_);

  this->declare_parameter("save_image", false);
  this->get_parameter("save_image", save_image_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_image: " << save_image_);

  this->declare_parameter("base_line", 0.06f);
  this->get_parameter("base_line", base_line);
  RCLCPP_INFO_STREAM(this->get_logger(), "base_line: " << base_line);

  this->declare_parameter("point_with_color", false);
  this->get_parameter("point_with_color", point_with_color_);
  RCLCPP_INFO_STREAM(this->get_logger(), "point_with_color_: " << point_with_color_);

  this->declare_parameter("stereonet_model_file_path", "./config/model.hbm");
  this->get_parameter("stereonet_model_file_path", stereonet_model_file_path_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereonet_model_file_path: " << stereonet_model_file_path_);

  this->declare_parameter("stereo_calib_file_path", "./config/stereo.yaml");
  this->get_parameter("stereo_calib_file_path", stereo_calib_file_path_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereo_calib_file_path: " << stereo_calib_file_path_);

  this->declare_parameter("stereo_image_topic", "/stereo_image");
  this->get_parameter("stereo_image_topic", stereo_image_topic_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereo_image_topic: " << stereo_image_topic_);

  this->declare_parameter("local_image_path", "./config/");
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
}

RCLCPP_COMPONENTS_REGISTER_NODE(stereonet::StereoNetNode)