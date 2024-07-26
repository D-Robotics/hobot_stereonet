//
// Created by zhy on 7/1/24.
//
#include <arm_neon.h>
#include <rclcpp_components/register_node_macro.hpp>
#include "stereonet_component.h"
#include "pcl_filter.h"

namespace stereonet {
int StereoNetNode::inference(const inference_data_t &inference_data,
                             std::vector<float> &points) {
  bool is_nv12;
  cv::Mat resized_left_img, resized_right_img;
  const cv::Mat &left_img = inference_data.left_sub_img.image;
  const cv::Mat &right_img = inference_data.right_sub_img.image;
  is_nv12 = inference_data.left_sub_img.image_type == sub_image_type::NV12;
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
  sensor_msgs::msg::Image depth_img_msg;
  const cv::Mat &depth_img = pub_raw_data.depth_img;

  if (depth_image_pub_->get_subscription_count() < 1) return 0;

  img_bridge = cv_bridge::CvImage(pub_raw_data.left_sub_img.header,
      "mono16", depth_img);
  img_bridge.toImageMsg(depth_img_msg);
  depth_image_pub_->publish(depth_img_msg);
  return 0;
}

int StereoNetNode::pub_visual_image(const pub_data_t &pub_raw_data) {
  cv_bridge::CvImage img_bridge;
  sensor_msgs::msg::Image visual_img_msg;
  const cv::Mat &image = pub_raw_data.left_sub_img.image;
  const std::vector<float> &points = pub_raw_data.points;
  cv::Mat bgr_image;

  if (visual_image_pub_->get_subscription_count() < 1) return 0;

  if (pub_raw_data.left_sub_img.image_type == sub_image_type::NV12) {
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

  img_bridge = cv_bridge::CvImage(pub_raw_data.left_sub_img.header,
      "bgr8", visual_img);
  img_bridge.toImageMsg(visual_img_msg);
  visual_image_pub_->publish(visual_img_msg);
  return 0;
}

int StereoNetNode::pub_rectified_image(const pub_data_t &pub_raw_data) {
  if (rectified_image_pub_->get_subscription_count() < 1) return 0;
  RCLCPP_WARN_ONCE(this->get_logger(),
    "pub rectified image with topic name [%s]",
    rectified_image_topic_.data());

  const cv::Mat &image = pub_raw_data.left_sub_img.image;
  int height = image.rows;
  int width = image.cols;
  const uint8_t* nv12_data_ptr = nullptr;
  cv::Mat nv12_image;
  if (pub_raw_data.left_sub_img.image_type == sub_image_type::NV12) {
    nv12_data_ptr = image.ptr<uint8_t>();
  } else {
    nv12_image = cv::Mat(height * 3 / 2, width, CV_8UC1);
    image_conversion::bgr24_to_nv12_neon(image.data, nv12_image.data, width, height);
    nv12_data_ptr = nv12_image.ptr<uint8_t>();
  }
  sensor_msgs::msg::Image pub_img_msg;
  pub_img_msg.header = pub_raw_data.left_sub_img.header;
  pub_img_msg.height = height;
  pub_img_msg.width = width;
  pub_img_msg.encoding = "nv12";
  pub_img_msg.step = width;
  size_t data_len = pub_img_msg.width * pub_img_msg.height * 3 / 2;
  pub_img_msg.data.resize(data_len);
  memcpy(pub_img_msg.data.data(), nv12_data_ptr, data_len);
  rectified_image_pub_->publish(pub_img_msg);

  return 0;
}

int StereoNetNode::pub_pointcloud2(const pub_data_t &pub_raw_data) {
  uint32_t point_size = 0;
  const cv::Mat &image = pub_raw_data.left_sub_img.image;
  const cv::Mat &depth_img = pub_raw_data.depth_img;
  uint16_t *depth_ptr = reinterpret_cast<uint16_t *>(depth_img.data);

  if (pointcloud2_pub_->get_subscription_count() < 1) return 0;

  sensor_msgs::msg::PointCloud2 point_cloud_msg;
  sensor_msgs::PointCloud2Modifier modifier(point_cloud_msg);

  int img_origin_width;
  int img_origin_height;

  if (pub_raw_data.left_sub_img.image_type == sub_image_type::NV12) {
    img_origin_width = image.cols;
    img_origin_height = image.rows * 2 / 3;
  } else {
    img_origin_width = image.cols;
    img_origin_height = image.rows;
  }

  point_cloud_msg.header = pub_raw_data.left_sub_img.header;
  point_cloud_msg.is_dense = false;
  point_cloud_msg.fields.resize(3);
  point_cloud_msg.fields[0].name = "x";
  point_cloud_msg.fields[0].offset = 0;
  point_cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  point_cloud_msg.fields[0].count = 1;

  point_cloud_msg.fields[1].name = "y";
  point_cloud_msg.fields[1].offset = 4;
  point_cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  point_cloud_msg.fields[1].count = 1;

  point_cloud_msg.fields[2].name = "z";
  point_cloud_msg.fields[2].offset = 8;
  point_cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  point_cloud_msg.fields[2].count = 1;
  point_cloud_msg.height = 1;
  point_cloud_msg.point_step = 12;

  //  point_cloud_msg.width = (img_origin_width / 2) * (img_origin_height / 2);
  //  point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width;
  point_cloud_msg.data.resize(
      (img_origin_width / 2) * (img_origin_height / 2) * point_cloud_msg.point_step *
      point_cloud_msg.height);

  float *pcd_data_ptr = reinterpret_cast<float *>(point_cloud_msg.data.data());
  float fy;
  for (int y = 0; y < img_origin_height; y += 2) {
    fy = (camera_cy  - y) / camera_fy;
    for (int x = 0; x < img_origin_width; x += 2) {
      float depth = depth_ptr[y * img_origin_width + x] / 1000.0f;
      //if (depth < height_min_ || depth > height_max_) continue;
      float X = (camera_cx - x) / camera_fx * depth;
      float Y = fy * depth;
      if (Y < height_min_ || Y > height_max_) {
        continue;
      }
      *pcd_data_ptr++ = depth;
      *pcd_data_ptr++ = X;
      *pcd_data_ptr++ = Y;
      point_size++;
    }
  }
  point_cloud_msg.width = point_size;
  point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width;
  point_cloud_msg.data.resize(point_size * point_cloud_msg.point_step *
          point_cloud_msg.height);
  {
    ScopeProcessTime t("pcl_filter");
    pcl_filter::applyfilter(point_cloud_msg,
                            leaf_size_, KMean_, stdv_);
  }

//  float32x4_t fx_vec = vdupq_n_f32(1 / camera_fx);
//  float32x4_t fy_vec = vdupq_n_f32(1 / camera_fy);
//  float32x4_t cx_vec = vdupq_n_f32(camera_cx);
//  float32x4_t cy_vec = vdupq_n_f32(camera_cy);
//  float32x4_t v1000 = vdupq_n_f32(0.001);
//  for (uint32_t y = 0; y < img_origin_height; y += 2) {
//    float32x4_t y_f32 = vdupq_n_f32(static_cast<float>(y));
//    for (uint32_t x = 0; x < img_origin_width; x += 8) {
//      uint32_t idx = y * img_origin_width + x;
//      uint32_t xx[4] = {x, x + 2, x + 4, x + 6};
//      uint16x4x2_t d = vld2_u16(&depth_ptr[idx]);
//      float32x4_t depth_f32 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(d.val[0])), v1000);
//      float32x4_t x_f32 = vcvtq_f32_u32(vld1q_u32(xx));
//      x_f32 = vmulq_f32(vsubq_f32(x_f32, cx_vec), fx_vec);
//      y_f32 = vmulq_f32(vsubq_f32(y_f32, cy_vec), fy_vec);
//      float32x4x3_t pts = {vmulq_f32(x_f32, depth_f32),
//                           vmulq_f32(y_f32, depth_f32),
//                           depth_f32};
//      vst3q_f32(pcd_data_ptr, pts);
//      pcd_data_ptr += 12;
//    }
//  }
  {
    ScopeProcessTime t("pcd publisher");
    pointcloud2_pub_->publish(point_cloud_msg);
  }
  return 0;
}
//
//int StereoNetNode::pub_pointcloud2(const pub_data_t &pub_raw_data) {
//  const cv::Mat &image = pub_raw_data.left_sub_img.image;
//  const std::vector<float> &points = pub_raw_data.points;
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
//      if (depth < height_min_ || depth > height_max_) continue;
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
                    cv::CALIB_ZERO_DISPARITY);

  cv::initUndistortRectifyMap(Kl, Dl, Rl, Pl, cv::Size(width, height), CV_32FC1, undistmap1l, undistmap2l);
  cv::initUndistortRectifyMap(Kr, Dr, Rr, Pr, cv::Size(width, height), CV_32FC1, undistmap1r, undistmap2r);

  cv::remap(left_image, rectified_left_image, undistmap1l, undistmap2l, cv::INTER_LINEAR);
  cv::remap(right_image, rectified_right_image, undistmap1r, undistmap2r, cv::INTER_LINEAR);

  rectified_fx = Q.at<double>(2, 3);
  rectified_fy = Q.at<double>(2, 3);
  rectified_cx = -Q.at<double>(0, 3);
  rectified_cy = -Q.at<double>(1, 3);
  //  const cv::Mat t = Rr * t_rl;
  baseline = std::abs(1 / Q.at<double>(3, 2));
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
  cv::Mat stereo_img, left_img, right_img;
  sub_image left_sub_img, right_sub_img;
  const std::string &encoding = img->encoding;
  int stereo_img_width, stereo_img_height;
  builtin_interfaces::msg::Time now = this->get_clock()->now();
  RCLCPP_DEBUG(this->get_logger(),
              "we have received stereo msg at: %ld.%ld,\n"
              "timestamp of stereo is: %ld.%ld, latency is %f sec,\n"
              "encoding: %s, width: %d, height: %d",
              now.sec, now.nanosec,
              img->header.stamp.sec, img->header.stamp.nanosec,
              (rclcpp::Time(now) - rclcpp::Time(img->header.stamp)).seconds(),
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
    if (stereo_combine_mode_ == 0) {
      RCLCPP_FATAL(this->get_logger(),
                   "when stereo_combine_mode is 0, the encoding of image must be bgr8!");
      return;
    }
    {
      ScopeProcessTime t("nv12->bgr");
      cv::Mat bgr(img->height, img->width, CV_8UC3);
//      cv::Mat nv12(img->height * 3 / 2, img->width, CV_8UC1, img->data.data());
//      cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
      image_conversion::nv12_to_bgr24_neon(img->data.data(), bgr.data, img->width, img->height);
      right_img = bgr(
          cv::Rect(0, 0, stereo_img_width, stereo_img_height)).clone();
      left_img = bgr(
          cv::Rect(0, stereo_img_height, stereo_img_width, stereo_img_height)).clone();
      left_sub_img.image_type = sub_image_type::BGR;
      right_sub_img.image_type = sub_image_type::BGR;
      left_sub_img.image = left_img;
      right_sub_img.image = right_img;
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
      save_images(left_sub_img.image, right_sub_img.image, img->header.stamp.sec * 1e9 + img->header.stamp.nanosec);
      return;
    }
  }

  left_sub_img.header = img->header;
  right_sub_img.header = img->header;

  inference_data_t inference_data {left_sub_img, right_sub_img};
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
        cv::Mat &left_image = inference_data.left_sub_img.image;
        cv::Mat &right_image = inference_data.right_sub_img.image;
        ScopeProcessTime t("stereo_rectify");
        stereo_rectify(left_image, right_image,
                       origin_image_width_, origin_image_height_,
                       Kl, Kr, Dl, Dr, R_rl, t_rl,
                       rectified_left_image, rectified_right_image,
                       camera_fx, camera_cx, camera_fy, camera_cy, base_line);
        left_image = rectified_left_image;
        right_image = rectified_right_image;
        RCLCPP_WARN_ONCE(this->get_logger(),
            "rectified fx: %f, fy: %f, cx: %f, cy: %f, base_line: :%f",
            camera_fx, camera_fy, camera_cx, camera_cy, base_line);
        //  return dump_rectified_image(left_img, right_img, rectified_left_img, rectified_right_img);
      }

      ret = inference(inference_data, points);
      if (ret != 0) {
        RCLCPP_ERROR(this->get_logger(), "inference failed.");
      } else {
        const sub_image &left_sub_img = inference_data.left_sub_img;
        const cv::Mat &left_img = left_sub_img.image;
        cv::Mat depth;
        pub_data_t pub_data{left_sub_img, points, depth};
        pub_func(pub_data);
      }
    }
  }
  inference_que_.clear();
}

void StereoNetNode::convert_depth(pub_data_t &pub_raw_data) {
  int img_origin_width, img_origin_height;
  const cv::Mat &image = pub_raw_data.left_sub_img.image;
  std::vector<float> &points = pub_raw_data.points;
  cv::Mat &depth_img = pub_raw_data.depth_img;
  std::vector<float> resized_points;
  if (pub_raw_data.left_sub_img.image_type == StereoNetNode::sub_image_type::NV12) {
    img_origin_width = image.cols;
    img_origin_height = image.rows * 2 / 3;
  } else {
    img_origin_width = image.cols;
    img_origin_height = image.rows;
  }
  if (img_origin_width != depth_w_ || img_origin_height != depth_h_) {
    resized_points.resize(img_origin_width * img_origin_height);
    cv::Mat resized_mat(img_origin_height, img_origin_width, CV_32FC1,
                        resized_points.data());
    cv::Mat origin_mat(depth_h_, depth_w_, CV_32FC1, points.data());
    cv::resize(origin_mat, resized_mat,
               cv::Size(img_origin_width, img_origin_height));
    points = std::move(resized_points);
  }
  depth_img = cv::Mat(img_origin_height, img_origin_width, CV_16UC1);
  uint16_t *depth_data = (uint16_t *)depth_img.data;
  float factor = 1000 * (camera_fx * base_line);
  uint32_t num_pixels = img_origin_height * img_origin_width;
  for (uint32_t i = 0; i < num_pixels; ++i) {
    depth_data[i] = factor / points[i];
  }
//  float32x4_t zero_vec = vdupq_n_f32(0.01f);
//  float32x4_t factor_vector = vdupq_n_f32(factor);
//  for (uint32_t i = 0; i < num_pixels; i += 4) {
//    float32x4_t points_vec = vmaxq_f32(vld1q_f32(&points[i]), zero_vec);
//    float32x4_t depth_vec = vdivq_f32(factor_vector, points_vec);
//    uint16x4_t depth_int16_vec = vmovn_u32(vcvtq_u32_f32(depth_vec));
//    vst1_u16(&depth_data[i], depth_int16_vec);
//  }
}

void StereoNetNode::pub_func(pub_data_t &pub_raw_data) {
  int ret = 0;
  {
    ScopeProcessTime t("convert to depth");
    convert_depth(pub_raw_data);
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
  {
    ScopeProcessTime t("pub_rectified");
    ret = pub_rectified_image(pub_raw_data);
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
  this->declare_parameter("camera_cx", 479.5f);
  this->get_parameter("camera_cx", camera_cx);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_cx: " << camera_cx);

  this->declare_parameter("camera_fx", 450.0f);
  this->get_parameter("camera_fx", camera_fx);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_fx: " << camera_fx);

  this->declare_parameter("camera_cy", 269.5f);
  this->get_parameter("camera_cy", camera_cy);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_cy: " << camera_cy);

  this->declare_parameter("camera_fy", 450.0f);
  this->get_parameter("camera_fy", camera_fy);
  RCLCPP_INFO_STREAM(this->get_logger(), "camera_fy: " << camera_fy);

  this->declare_parameter("need_rectify", true);
  this->get_parameter("need_rectify", need_rectify_);
  RCLCPP_INFO_STREAM(this->get_logger(), "need_rectify: " << need_rectify_);

  this->declare_parameter("save_image", false);
  this->get_parameter("save_image", save_image_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_image: " << save_image_);

  this->declare_parameter("base_line", 0.1f);
  this->get_parameter("base_line", base_line);
  RCLCPP_INFO_STREAM(this->get_logger(), "base_line: " << base_line);

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

  this->declare_parameter("height_min", -0.2);
  this->get_parameter("height_min", height_min_);
  RCLCPP_INFO_STREAM(this->get_logger(), "height_min_: " << height_min_);

  this->declare_parameter("height_max", 1.f);
  this->get_parameter("height_max", height_max_);
  RCLCPP_INFO_STREAM(this->get_logger(), "height_max: " << height_max_);

  this->declare_parameter("stereo_combine_mode", stereo_combine_mode_);
  this->get_parameter("stereo_combine_mode", stereo_combine_mode_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stereo_combine_mode: " << stereo_combine_mode_);

  this->declare_parameter("leaf_size", leaf_size_);
  this->get_parameter("leaf_size", leaf_size_);
  RCLCPP_INFO_STREAM(this->get_logger(), "leaf_size: " << leaf_size_);

  this->declare_parameter("KMean", KMean_);
  this->get_parameter("KMean", KMean_);
  RCLCPP_INFO_STREAM(this->get_logger(), "KMean: " << KMean_);

  this->declare_parameter("stdv", stdv_);
  this->get_parameter("stdv", stdv_);
  RCLCPP_INFO_STREAM(this->get_logger(), "stdv: " << stdv_);
}

void StereoNetNode::inference_by_usb_camera() {
/*
  cv::VideoCapture *capture = nullptr;
  cv::Mat stereo_img, left_img, right_img;
  cv::Rect right_rect(0, 0, 640, 360),
           left_rect(0, 360, 640, 360);
  uint64_t current_ts;

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
      inference_data_t inference_data = std::make_tuple(left_img, right_img);
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
  std_msgs::msg::Header image_header;
  sub_image left_sub_img, right_sub_img;
  uint64_t current_ts;
  std::vector<float> points;
  cv::Mat left_img = cv::imread(local_image_path_ + "/left.png");
  cv::Mat right_img = cv::imread(local_image_path_ + "/right.png");
  image_header.frame_id =  "default_cam";
  image_header.stamp = this->now();
  current_ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  left_sub_img.image_type = sub_image_type::BGR;
  right_sub_img.image_type = sub_image_type::BGR;
  left_sub_img.image = left_img;
  right_sub_img.image = right_img;
  left_sub_img.header = image_header;
  right_sub_img.header = image_header;
  inference_data_t inference_data {left_sub_img, right_sub_img};
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
      
  rectified_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      rectified_image_topic_, 10);
}
}

RCLCPP_COMPONENTS_REGISTER_NODE(stereonet::StereoNetNode)