//
// Created by zhy on 7/1/24.
//
#include <arm_neon.h>
#include <dirent.h>
#include <rclcpp_components/register_node_macro.hpp>
#include "stereonet_component.h"
#include "pcl_filter.h"


namespace stereonet {

void StereoNetNode::save_mat_to_bin(const cv::Mat &mat, const std::string &filename)
{
  if (mat.empty())
  {
    RCLCPP_ERROR(this->get_logger(), "=> The input matrix is empty!");
    return;
  }

  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open())
  {
    RCLCPP_ERROR(this->get_logger(), "=> Failed to open file for writing!");
    return;
  }

  const uchar *dataPtr = mat.data;
  size_t dataSize = mat.total() * mat.elemSize();
  // RCLCPP_INFO(this->get_logger(), "=> dataSize: %ld, mat.total(): %ld, mat.elemSize(): %ld", dataSize, mat.total(), mat.elemSize());

  ofs.write(reinterpret_cast<const char *>(dataPtr), dataSize);

  ofs.close();
}

int StereoNetNode::inference(inference_data_t &inference_data,
                             std::vector<float> &points) {
  bool is_nv12;
 // cv::Mat resized_left_img, resized_right_img;
  cv::Mat &left_img = inference_data.left_sub_img.image;
  cv::Mat &right_img = inference_data.right_sub_img.image;
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
    //  resized_left_img = left_img;
    //  resized_right_img = right_img;
  } else {
    if (left_img.rows != model_input_h_ || left_img.cols != model_input_w_) {
      // RCLCPP_INFO(this->get_logger(), "\033[31m=> resize img [%d, %d] to [%d, %d]\033[0m", left_img.cols, left_img.rows, model_input_w_, model_input_h_);
      cv::resize(left_img, left_img, cv::Size(model_input_w_, model_input_h_));
      cv::resize(right_img, right_img, cv::Size(model_input_w_, model_input_h_));
    } else {
      //  resized_left_img = left_img;
      //  resized_right_img = right_img;
    }
  }

  inference_data.left_sub_img.origin_height = left_img.rows;
  inference_data.left_sub_img.origin_width = left_img.cols;
  inference_data.right_sub_img.origin_height = right_img.rows;
  inference_data.right_sub_img.origin_width = right_img.cols;

  if (save_image_all_ && !is_nv12) {
    if (!directory_created_) {
      directory_created_ = true;
      system("mkdir -p ./stereonet_images");
    }

    save_images_with_nv12(left_img, right_img, image_format_);

    if (save_cnt_ == 1)
    {
      std::stringstream ss;
      ss << "[fx, fy, cx, cy, baseline] = [" << camera_fx << ", "<< camera_fy << ", "<< camera_cx << ", "<< camera_cy << ", " << base_line * 1000 << "]" << std::endl;
      std::string result = ss.str();
      std::ofstream outFile("./stereonet_images/calib_param.txt");
      if (outFile.is_open()) {
        outFile << result;
        outFile.close();
        RCLCPP_WARN_STREAM(this->get_logger(), "=> calib param save to: ./stereonet_images/calib_param.txt");
      }
    }
  }

  return stereonet_process_->stereonet_inference(left_img, right_img,
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
  const cv::Mat &depth_img = pub_raw_data.model_depth_img;
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
  feat_mat.convertTo(feat_visual, CV_8U, visual_alpha_, visual_beta_);

  //  cv::convertScaleAbs(feat_visual, feat_visual, 2);
  cv::applyColorMap(feat_visual,
                    visual_img(cv::Rect(0, bgr_image.rows, bgr_image.cols, bgr_image.rows)),
                    cv::COLORMAP_JET);

  int step_num = 6;
  int x_step = bgr_image.cols / step_num;
  int y_step = bgr_image.rows / step_num;
  // RCLCPP_WARN_ONCE(this->get_logger(), "=> x_step: %d, y_step: %d", x_step, y_step);

  for (int i = 1; i < step_num; i++)
  {
    for (int j = 1; j < step_num; j++)
    {
      // 横线
      cv::line(visual_img, cv::Point2i(0, bgr_image.rows + i * y_step),
               cv::Point2i(bgr_image.cols, bgr_image.rows + i * y_step),
               cv::Scalar(255, 255, 255), 1);
      // 竖线
      cv::line(visual_img, cv::Point2i(j * x_step, bgr_image.rows),
               cv::Point2i(j * x_step, bgr_image.rows * 2),
               cv::Scalar(255, 255, 255), 1);

      // 横线
      cv::line(visual_img, cv::Point2i(0, i * y_step),
               cv::Point2i(bgr_image.cols, i * y_step),
               cv::Scalar(255, 255, 255), 1);
      // 竖线
      cv::line(visual_img, cv::Point2i(j * x_step, 0),
               cv::Point2i(j * x_step, bgr_image.rows),
               cv::Scalar(255, 255, 255), 1);
      // 取出Z值
      uint16_t Z = depth_img.at<uint16_t>(i * y_step, j * x_step);
      // mm -> m
      double distance = static_cast<double>(Z) / 1000.0;
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(2) << distance << "m";
      double font_scale = 1.0;
      if (postprocess_ == "v2") font_scale = 0.5;
      cv::putText(visual_img, ss.str(), cv::Point2i(j * x_step + 3,
                                                    bgr_image.rows + i * y_step - 3),
                  cv::FONT_HERSHEY_SIMPLEX, font_scale,
                  cv::Scalar(255, 255, 255), 2);

      cv::putText(visual_img, ss.str(), cv::Point2i(j * x_step + 3,
                                                    i * y_step - 3),
                  cv::FONT_HERSHEY_SIMPLEX, font_scale,
                  cv::Scalar(255, 255, 255), 2);
    }
  }

  if (depth_compare) {
    cv::putText(visual_img, "camera", cv::Point2i(3, 38),
                cv::FONT_HERSHEY_SIMPLEX, 1.5,
                cv::Scalar(0, 255, 0), 3);

    cv::putText(visual_img, "AI-depth", cv::Point2i(3, bgr_image.rows + 38),
                cv::FONT_HERSHEY_SIMPLEX, 1.5,
                cv::Scalar(0, 255, 0), 3);
    {
      std::lock_guard<std::mutex> lck(compare_visual_mtx_);

      cv::hconcat(visual_img, compare_visual_, visual_img);
    }
  }

  img_bridge = cv_bridge::CvImage(pub_raw_data.left_sub_img.header,
                                  "bgr8", visual_img);
  img_bridge.toImageMsg(visual_img_msg);
  visual_image_pub_->publish(visual_img_msg);
//  static uint32_t i = 0;
//  std::ofstream result(std::to_string(i) + ".txt", std::ios::out);
//  cv::imwrite(std::to_string(i) + ".jpg", visual_img);
//  i++;
//  result << pub_raw_data.depth_img;
  return 0;
}

int StereoNetNode::pub_rectified_image(const pub_data_t &pub_raw_data) {
  if (rectified_image_pub_->get_subscription_count() > 0) {
    sensor_msgs::msg::Image pub_img_msg;
    const cv::Mat &image = pub_raw_data.left_sub_img.image;
    int height = image.rows;
    int width = image.cols;
    const uint8_t* nv12_data_ptr = nullptr;
    cv::Mat nv12_image;

    pub_img_msg.header = pub_raw_data.left_sub_img.header;
    pub_img_msg.height = height;
    pub_img_msg.width = width;

    if (pub_rectified_bgr_) {
      pub_img_msg.encoding = "bgr8";
      pub_img_msg.step = width * 3;
      size_t data_len = pub_img_msg.width * pub_img_msg.height * 3;
      pub_img_msg.data.resize(data_len);
      memcpy(pub_img_msg.data.data(), image.data, data_len);
    } else {
      pub_img_msg.encoding = "nv12";
      pub_img_msg.step = width;
      if (pub_raw_data.left_sub_img.image_type == sub_image_type::NV12) {
        nv12_data_ptr = image.ptr<uint8_t>();
      } else {
        nv12_image = cv::Mat(height * 3 / 2, width, CV_8UC1);
        image_conversion::bgr24_to_nv12_neon(image.data, nv12_image.data, width, height);
        nv12_data_ptr = nv12_image.ptr<uint8_t>();
      }
      size_t data_len = pub_img_msg.width * pub_img_msg.height * 3 / 2;
      pub_img_msg.data.resize(data_len);
      memcpy(pub_img_msg.data.data(), nv12_data_ptr, data_len);
    }
    rectified_image_pub_->publish(pub_img_msg);
  }

  if (rectified_right_image_pub_->get_subscription_count() > 0) {
    sensor_msgs::msg::Image pub_img_msg;
    const cv::Mat &image = pub_raw_data.right_sub_img.image;
    int height = image.rows;
    int width = image.cols;
    const uint8_t* nv12_data_ptr = nullptr;
    cv::Mat nv12_image;

    pub_img_msg.header = pub_raw_data.right_sub_img.header;
    pub_img_msg.height = height;
    pub_img_msg.width = width;

    if (pub_rectified_bgr_) {
      pub_img_msg.encoding = "bgr8";
      pub_img_msg.step = width * 3;
      size_t data_len = pub_img_msg.width * pub_img_msg.height * 3;
      pub_img_msg.data.resize(data_len);
      memcpy(pub_img_msg.data.data(), image.data, data_len);
    } else {
      pub_img_msg.encoding = "nv12";
      pub_img_msg.step = width;
      if (pub_raw_data.left_sub_img.image_type == sub_image_type::NV12) {
        nv12_data_ptr = image.ptr<uint8_t>();
      } else {
        nv12_image = cv::Mat(height * 3 / 2, width, CV_8UC1);
        image_conversion::bgr24_to_nv12_neon(image.data, nv12_image.data, width, height);
        nv12_data_ptr = nv12_image.ptr<uint8_t>();
      }
      size_t data_len = pub_img_msg.width * pub_img_msg.height * 3 / 2;
      pub_img_msg.data.resize(data_len);
      memcpy(pub_img_msg.data.data(), nv12_data_ptr, data_len);
    }
    rectified_right_image_pub_->publish(pub_img_msg);
  }
  RCLCPP_WARN_ONCE(this->get_logger(),
                   "pub rectified image with topic name [%s]",
                   rectified_image_topic_.data());
  return 0;
}

int StereoNetNode::pub_pointcloud2(const pub_data_t &pub_raw_data) {
  uint32_t point_size = 0;
  const cv::Mat &image = pub_raw_data.left_sub_img.image;
  const cv::Mat &depth_img = pub_raw_data.model_depth_img;
  uint16_t *depth_ptr = reinterpret_cast<uint16_t *>(depth_img.data);

  if (pointcloud2_pub_->get_subscription_count() < 1) return 0;

  sensor_msgs::msg::PointCloud2 point_cloud_msg;
  sensor_msgs::PointCloud2Modifier modifier(point_cloud_msg);

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
      (depth_w_ / 2) * (depth_h_ / 2) * point_cloud_msg.point_step *
          point_cloud_msg.height);

  float *pcd_data_ptr = reinterpret_cast<float *>(point_cloud_msg.data.data());
  float fy;
  for (int y = 0; y < depth_h_; y += 2) {
    fy = (camera_cy  - y) / camera_fy;
    for (int x = 0; x < depth_w_; x += 2) {
      float depth = depth_ptr[y * depth_w_ + x] / 1000.0f;
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

  if (need_pcl_filter_) {
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

void dump_rectified_image(cv::Mat &left_img, cv::Mat &right_img,
                          cv::Mat &rectified_left_img, cv::Mat &rectified_right_img) {
  std::stringstream iss;
  static std::atomic_int iii {0};
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
  iss << std::setw(6) << std::setfill('0') << iii++;
  auto image_seq = iss.str();
  cv::imwrite("./230ai_data/left" + image_seq + "_rectify.png", rectified_left_img);
  cv::imwrite("./230ai_data/right" + image_seq + "_rectify.png", rectified_right_img);
  cv::imwrite("./before.jpg", img_src);
  cv::imwrite("./after.jpg", img_rtf);
}

void StereoNetNode::save_images_with_nv12(cv::Mat &left_img, cv::Mat &right_img, const std::string &image_format) {
  std::stringstream iss;
  iss << std::setw(6) << std::setfill('0') << save_cnt_;
  auto image_seq = iss.str();
  cv::imwrite("./stereonet_images/left" + image_seq + "." + image_format, left_img);
  cv::imwrite("./stereonet_images/right" + image_seq + "." + image_format, right_img);
  if (save_image_to_nv12_)
  {
    cv::Mat left_img_nv12, right_img_nv12;
    image_conversion::bgr_to_nv12(left_img, left_img_nv12);
    image_conversion::bgr_to_nv12(right_img, right_img_nv12);
    save_mat_to_bin(left_img_nv12, "./stereonet_images/left" + image_seq + ".nv12");
    save_mat_to_bin(right_img_nv12, "./stereonet_images/right" + image_seq + ".nv12");
  }
}

void save_images(cv::Mat &left_img, cv::Mat &right_img, uint64_t ts,
    const std::string &image_format) {
  static std::atomic_bool directory_created{false};
  static std::atomic_int i {0};
  std::stringstream iss;
  cv::Mat image_combine;
  if (!directory_created) {
    directory_created = true;
    system("mkdir -p"
           " ./images/cam0/data/"
           " ./images/cam1/data/"
           " ./images/cam_combine/data/");
  }
  iss << std::setw(3) << std::setfill('0') << i++;
  auto image_seq = iss.str();
  cv::imwrite("./images/cam0/data/" + std::to_string(ts) + "." + image_format, left_img);
  cv::imwrite("./images/cam1/data/" + std::to_string(ts) + "." + image_format, right_img);
  //cv::vconcat(left_img, right_img, image_combine);
  //cv::imwrite("./images/cam_combine/data/combine_" + image_seq + image_format, image_combine);
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
    ScopeProcessTime t("nv12->bgr");
    stereo_img = cv::Mat(img->height, img->width, CV_8UC3);
//      cv::Mat nv12(img->height * 3 / 2, img->width, CV_8UC1, img->data.data());
//      cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
    image_conversion::nv12_to_bgr24_neon(img->data.data(), stereo_img.data, img->width, img->height);
  } else if (encoding == "bgr8" || encoding == "BGR8") {
    ScopeProcessTime t("cv_bridge::toCvShare");
    stereo_img = cv_bridge::toCvShare(img)->image;
  }

  if (stereo_combine_mode_ == 0) {
    left_img = stereo_img(
        cv::Rect(0, 0, stereo_img_width, stereo_img_height));
    right_img = stereo_img(
        cv::Rect(stereo_img_width, 0, stereo_img_width, stereo_img_height));
  } else if (stereo_combine_mode_ == 1) {
    left_img = stereo_img(
        cv::Rect(0, 0, stereo_img_width, stereo_img_height));
    right_img = stereo_img(
        cv::Rect(0, stereo_img_height, stereo_img_width, stereo_img_height));
  }

  left_sub_img.image_type = sub_image_type::BGR;
  right_sub_img.image_type = sub_image_type::BGR;

  left_sub_img.image = left_img.clone();
  right_sub_img.image = right_img.clone();

  left_sub_img.header = img->header;
  right_sub_img.header = img->header;

  left_sub_img.origin_height = left_sub_img.image.rows;
  left_sub_img.origin_width = left_sub_img.image.cols;
  right_sub_img.origin_height = right_sub_img.image.rows;
  right_sub_img.origin_width = right_sub_img.image.cols;

  inference_data_t inference_data {left_sub_img, right_sub_img};
  if (inference_que_.size() > 5) {
    RCLCPP_WARN_THROTTLE(this->get_logger(),
                         *this->get_clock(), 5000, "inference que is full!");
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
      cv::Mat &left_image = inference_data.left_sub_img.image;
      cv::Mat &right_image = inference_data.right_sub_img.image;
      if (left_image.cols != model_input_w_ || left_image.rows != model_input_h_) {
        cv::resize(left_image, left_image, cv::Size(model_input_w_, model_input_h_));
        cv::resize(right_image, right_image, cv::Size(model_input_w_, model_input_h_));
      }
      if (need_rectify_) {
        ScopeProcessTime t("stereo_rectify");
        for (auto & s : stereo_rectify_list_) {
          s->Rectify(left_image, right_image, rectified_left_image, rectified_right_image);
          left_image = rectified_left_image;
          right_image = rectified_right_image;
        }
        //  dump_rectified_image(left_image, right_image, rectified_left_image, rectified_right_image);
      }

      if (save_image_) {
        save_images(inference_data.left_sub_img.image,
                    inference_data.right_sub_img.image,
                    inference_data.left_sub_img.header.stamp.sec * 1e9
                     + inference_data.left_sub_img.header.stamp.nanosec,
                     image_format_);
      }

      ret = inference(inference_data, points);
      if (ret != 0) {
        RCLCPP_ERROR(this->get_logger(), "inference failed.");
      } else {
        const sub_image &left_sub_img = inference_data.left_sub_img;
        const sub_image &right_sub_img = inference_data.right_sub_img;
        const cv::Mat &left_img = left_sub_img.image;
        cv::Mat depth;
        pub_data_t pub_data{left_sub_img, right_sub_img, points, depth};
        pub_func(pub_data);
//        dump_one_point_disparity(pub_data,
//            inference_data.right_sub_img.image, 659, 301);
      }
    }
  }
  inference_que_.clear();
}

void StereoNetNode::dump_one_point_disparity(
    pub_data_t &pub_raw_data, const cv::Mat &right_image,
    int x, int y) {
  cv::Mat left_image = pub_raw_data.left_sub_img.image.clone();
  cv::Mat right_image2 = right_image.clone();
  std::vector<float> &points = pub_raw_data.points;
  auto disparity = points[y * left_image.cols + x];
  cv::circle(left_image, cv::Point(x, y), 10,
             cv::Scalar(255, 0, 0), 3);
  cv::circle(right_image2, cv::Point(x - disparity, y), 10,
             cv::Scalar(255, 0, 0), 3);
  RCLCPP_INFO(this->get_logger(), "[x: %d, y: %d, disp: %f]\n", x, y, disparity);
  cv::imwrite("one_point_disparity_left.jpeg", left_image);
  cv::imwrite("one_point_disparity_right.jpeg", right_image2);
}

void StereoNetNode::convert_depth(pub_data_t &pub_raw_data) {
  int img_origin_width, img_origin_height;
  const cv::Mat &image = pub_raw_data.left_sub_img.image;
  std::vector<float> &points = pub_raw_data.points;
  std::vector<float> &image_size_points = pub_raw_data.image_size_points;
  cv::Mat &depth_img = pub_raw_data.depth_img;
  cv::Mat model_depth_img;
  std::vector<float> resized_points;
  img_origin_width = pub_raw_data.left_sub_img.origin_width;
  img_origin_height = pub_raw_data.left_sub_img.origin_height;

  model_depth_img = cv::Mat(depth_h_, depth_w_, CV_16UC1);
  uint16_t *depth_data = (uint16_t *)model_depth_img.data;
  float factor = 1000 * (camera_fx * base_line);
  uint32_t num_pixels = points.size();
  for (uint32_t i = 0; i < num_pixels; ++i) {
    depth_data[i] = factor / points[i];
  }

  pub_raw_data.model_depth_img = model_depth_img;

  if (img_origin_width != depth_w_ || img_origin_height != depth_h_) {
    cv::resize(model_depth_img, depth_img,
               cv::Size(img_origin_width, img_origin_height));
    image_size_points.resize(img_origin_width * img_origin_height);
    cv::Mat image_size_points_mat(img_origin_height, img_origin_width,
                                  CV_32FC1, image_size_points.data());
    cv::Mat points_mat(depth_h_, depth_w_, CV_32FC1, points.data());
    cv::resize(points_mat, image_size_points_mat,
               cv::Size(img_origin_width, img_origin_height));
  } else {
    depth_img = model_depth_img;
    image_size_points = points;
  }

  if (save_image_all_)
  {
    RCLCPP_WARN(this->get_logger(), "=> img_origin_height: %d,  img_origin_width: %d", img_origin_height, img_origin_width);
    cv::Mat disp_img = cv::Mat(img_origin_height, img_origin_width, CV_32FC1);
    float *disp_data = (float *)disp_img.data;
    for (uint32_t i = 0; i < num_pixels; ++i) {
      disp_data[i] = points[i];
    }
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << save_cnt_;
    std::string disp_path = "./stereonet_images/disp" + ss.str() + ".pfm";
    std::string depth_path = "./stereonet_images/depth" + ss.str() + ".png";
    cv::imwrite(disp_path, disp_img);
    cv::imwrite(depth_path, depth_img);
    RCLCPP_WARN_STREAM(this->get_logger(), "=> save to: " << disp_path);

    save_cnt_++;
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
  ret = stereonet_process_->stereonet_init(stereonet_model_file_path_, max_disp_, postprocess_);
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
  RCLCPP_WARN(this->get_logger(), "\033[31m=> rectified fx: %f, fy: %f, cx: %f, cy: %f, base_line: :%f\033[0m", camera_fx, camera_fy, camera_cx, camera_cy, base_line);
  compare_visual_ = cv::Mat::zeros(cv::Size(depth_w_, depth_h_ * 2), CV_8UC3);

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
  int i = 0;
  cv::FileStorage fs(file_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    RCLCPP_WARN_STREAM(this->get_logger(), "Failed to open " << file_path);
    return;
  }

  do {
    std::string stereo_no = "stereo" + std::to_string(i++);
    if (!fs[stereo_no].empty()) {
      RCLCPP_WARN_STREAM(this->get_logger(), "Add StereoRectify Instance: " << stereo_no);
      stereo_rectify_list_.emplace_back(std::make_shared<StereoRectify>(
          fs[stereo_no], model_input_w, model_input_h));
    } else {
      break;
    }
  } while(true);

  if (need_rectify_) {
    stereo_rectify_list_.back()->GetIntrinsic(camera_cx, camera_cy, camera_fx, camera_fy, base_line);
    RCLCPP_WARN(this->get_logger(), "rectified fx: %f, fy: %f, cx: %f, cy: %f, base_line: :%f",
                camera_fx, camera_fy, camera_cx, camera_cy, base_line);
  }

  fs.release();
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

  this->declare_parameter("need_pcl_filter", false);
  this->get_parameter("need_pcl_filter", need_pcl_filter_);
  RCLCPP_INFO_STREAM(this->get_logger(), "need_pcl_filter: " << need_pcl_filter_);

  this->declare_parameter("save_image", false);
  this->get_parameter("save_image", save_image_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_image: " << save_image_);

  this->declare_parameter("save_image_all", false);
  this->get_parameter("save_image_all", save_image_all_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_image_all: " << save_image_all_);

  this->declare_parameter("save_image_to_nv12", false);
  this->get_parameter("save_image_to_nv12", save_image_to_nv12_);
  RCLCPP_INFO_STREAM(this->get_logger(), "save_image_to_nv12: " << save_image_to_nv12_);

  this->declare_parameter("postprocess", "v1");
  this->get_parameter("postprocess", postprocess_);
  RCLCPP_INFO_STREAM(this->get_logger(), "postprocess: " << postprocess_);

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

  this->declare_parameter("visual_beta", visual_beta_);
  this->get_parameter("visual_beta", visual_beta_);
  RCLCPP_INFO_STREAM(this->get_logger(), "visual_beta: " << visual_beta_);

  this->declare_parameter("alpha", visual_alpha_);
  this->get_parameter("alpha", visual_alpha_);
  RCLCPP_INFO_STREAM(this->get_logger(), "visual_alpha: " << visual_alpha_);

  this->declare_parameter("max_disp", max_disp_);
  this->get_parameter("max_disp", max_disp_);
  RCLCPP_INFO_STREAM(this->get_logger(), "max_disp: " << max_disp_);

  this->declare_parameter("rectify_bgr", pub_rectified_bgr_);
  this->get_parameter("rectify_bgr", pub_rectified_bgr_);
  RCLCPP_INFO_STREAM(this->get_logger(), "pub_rectified_bgr: " << pub_rectified_bgr_);

  this->declare_parameter("image_format", image_format_);
  this->get_parameter("image_format", image_format_);
  RCLCPP_INFO_STREAM(this->get_logger(), "image_format: " << image_format_);

  this->declare_parameter("image_sleep", image_inference_sleep_ms_);
  this->get_parameter("image_sleep", image_inference_sleep_ms_);
  RCLCPP_INFO_STREAM(this->get_logger(), "image_inference_sleep_ms: " << image_inference_sleep_ms_);
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

int get_image(const std::string &image_path,
              cv::Mat &left_img, cv::Mat &right_img, int64_t &ts,
              const std::string &image_format) {
  static uint32_t i_num = 0;
  std::stringstream iss;
  std::string image_seq;
  iss << std::setw(6) << std::setfill('0') << i_num++;
  image_seq = iss.str();
  std::string left_img_path = image_path + "/left" + image_seq + "." + image_format;
  std::string right_img_path = image_path + "/right"+ image_seq + "." + image_format;
  left_img = cv::imread(left_img_path);
  right_img = cv::imread(right_img_path);
  ts = 0;
  if (left_img.empty() || right_img.empty()) {
    return -1;
  }
  RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "=> left_img_path: " << left_img_path << ", right_img_path: " << right_img_path);
  return 0;
}

void get_image_file_list(const std::string &image_path,
                         std::vector<std::string> &file_names) {
  DIR *pDir;
  struct dirent *ptr;
  if (!(pDir = opendir(image_path.c_str()))) {
    RCLCPP_ERROR(rclcpp::get_logger(""),
                 "image path is not existed: %s ", image_path.c_str());
    return;
  }
  while ((ptr = readdir(pDir)) != 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
      std::string file_name = ptr->d_name;
      size_t lastDot = file_name.find_last_of('.');
      if (lastDot != std::string::npos) {
        auto extension = file_name.substr(lastDot + 1);
        std::transform(extension.begin(),
                       extension.end(), extension.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (extension == "png" ||
            extension == "jpg" || extension == "jpeg") {
          file_names.push_back(file_name);
        }
      }
    }
  }
  sort(file_names.begin(), file_names.end());
  closedir(pDir);
}

int get_image2(const std::string &image_path, cv::Mat &left_img,
               cv::Mat &right_img, int64_t &ts, const std::string &image_format) {
  static std::vector<std::string> left_file_names, right_file_names;
  static uint32_t i_num = 0;
  if (i_num == 0) {
    get_image_file_list(image_path + "/cam0/data/", left_file_names);
    get_image_file_list(image_path + "/cam1/data/", right_file_names);
  }
  if (i_num < left_file_names.size()) {
    std::string file_name;
    size_t lastDot = left_file_names[i_num].find_last_of('.');
    // 分离文件名和后缀
    if (lastDot == std::string::npos) {
      file_name = left_file_names[i_num];
    } else {
      file_name = left_file_names[i_num].substr(0, lastDot);
    }
    ts = std::atoll(file_name.c_str());
    left_img = cv::imread(image_path + "/cam0/data/" + left_file_names[i_num]);
    right_img = cv::imread(image_path + "/cam1/data/"+ right_file_names[i_num]);
    i_num++;
    return 0;
  }
  i_num = 0;
  return -1;
}

void StereoNetNode::inference_by_image() {
  std_msgs::msg::Header image_header;
  sub_image left_sub_img, right_sub_img;
  int64_t ts;
  while (rclcpp::ok()) {
    if (inference_que_.size() > 5) {
      RCLCPP_WARN_THROTTLE(this->get_logger(),
                           *this->get_clock(), 2000, "inference que is full!");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    if (-1 == get_image(local_image_path_, left_sub_img.image,
        right_sub_img.image, ts, image_format_)) {
//    if (-1 == get_image2(local_image_path_, left_sub_img.image,
//                         right_sub_img.image, ts, image_format_)) {
      continue;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(image_inference_sleep_ms_));
    image_header.frame_id =  "default_cam";
    if (ts != 0) {
      image_header.stamp = rclcpp::Time(ts);
    } else {
      image_header.stamp = this->now();
    }
    left_sub_img.image_type = sub_image_type::BGR;
    right_sub_img.image_type = sub_image_type::BGR;
    left_sub_img.header = image_header;
    right_sub_img.header = image_header;
    left_sub_img.origin_height = left_sub_img.image.rows;
    left_sub_img.origin_width = left_sub_img.image.cols;
    right_sub_img.origin_height = right_sub_img.image.rows;
    right_sub_img.origin_width = right_sub_img.image.cols;
    inference_que_.put({left_sub_img, right_sub_img});
  }
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

  rectified_right_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      rectified_right_image_topic_, 10);

  depth_compare = false;
  std::string compare_depth_topic = "~/stereonet_depth";
  std::string visual_topic = "~/stereonet_visual";
  std::string compare_image_topic = "~/rectified_image";

  depth_compare = this->declare_parameter("depth_compare", depth_compare);
  RCLCPP_INFO_STREAM(this->get_logger(), "depth_compare: " << depth_compare);

  compare_depth_topic = this->declare_parameter("compare_depth_topic", compare_depth_topic);
  RCLCPP_INFO_STREAM(this->get_logger(), "compare_depth_topic: " << compare_depth_topic);

  visual_topic = this->declare_parameter("visual_topic", visual_topic);
  RCLCPP_INFO_STREAM(this->get_logger(), "visual_topic: " << visual_topic);

  compare_image_topic = this->declare_parameter("compare_image_topic", compare_image_topic);
  RCLCPP_INFO_STREAM(this->get_logger(), "compare_image_topic: " << compare_image_topic);

  if (depth_compare) {
    depth_subscriber_.subscribe(this, compare_depth_topic);
    //color_subscriber_.subscribe(this, visual_topic);
    compare_left_subscriber_.subscribe(this, compare_image_topic);

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        SyncPolicy(10),
        depth_subscriber_,
        //color_subscriber_,
        compare_left_subscriber_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
    sync_->registerCallback(std::bind(&StereoNetNode::sync_callback,
                                      this, std::placeholders::_1,
        //std::placeholders::_2,
                                      std::placeholders::_2
    ));

    compare_visual_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "~/compare_stereonet_visual", 10);

/*
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    qos.reliability(rclcpp::ReliabilityPolicy::Reliable);
    qos.durability(rclcpp::DurabilityPolicy::TransientLocal);

     depth_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
       compare_depth_topic, qos,
       std::bind(&StereoNetNode::d_callback, this, std::placeholders::_1));


     compare_left_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
       compare_image_topic, qos,
       std::bind(&StereoNetNode::c_callback, this, std::placeholders::_1));
      */
  }
}


void StereoNetNode::d_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
  std::cout << "d cb" << std::endl;

}

void StereoNetNode::c_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg) {
  std::cout << "c cb" << std::endl;

}

void StereoNetNode::sync_callback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &depth_msg,
    //const sensor_msgs::msg::Image::ConstSharedPtr &color_msg,
                                  const sensor_msgs::msg::CompressedImage::ConstSharedPtr &rs_msg) {
  sensor_msgs::msg::Image visual_image_msg;
  cv_bridge::CvImage img_bridge;

  const float fx_bl = 37050;
  int width, height;
  std::vector<unsigned char> compressed_data(depth_msg->data.begin() + 12, depth_msg->data.end());
  cv::Mat depth_image = cv::imdecode(compressed_data, cv::IMREAD_UNCHANGED);
  //  cv::Mat depth_image = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::MONO16)->image;
  cv::Mat rs_image = cv_bridge::toCvCopy(rs_msg, sensor_msgs::image_encodings::BGR8)->image;
  width = depth_w_;
  height = depth_h_;

  if (depth_image.empty() || rs_image.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to decode compressed image");
    return;
  }

  if (depth_image.cols != width || depth_image.rows != height) {
    cv::resize(depth_image, depth_image, cv::Size(width, height));
  }
  if (rs_image.cols != width || rs_image.rows != height) {
    cv::resize(rs_image, rs_image, cv::Size(width, height));
  }

  cv::Mat dis_image = fx_bl / depth_image;
  dis_image.convertTo(dis_image, CV_8U);
  cv::applyColorMap(dis_image, dis_image,userColor_);

  int step_num = 6;
  int x_step = width / step_num;
  int y_step = height / step_num;

  for (int i = 1; i < step_num; i++) {
    for (int j = 1; j < step_num; j++) {
      cv::line(rs_image, cv::Point2i(0, i * y_step),
               cv::Point2i(width, i * y_step),
               cv::Scalar(255, 255, 255), 1);
      cv::line(rs_image, cv::Point2i(j * x_step, 0),
               cv::Point2i(j * x_step, height),
               cv::Scalar(255, 255, 255), 1);

      cv::line(dis_image, cv::Point2i(0, i * y_step),
               cv::Point2i(width, i * y_step),
               cv::Scalar(255, 255, 255), 1);
      cv::line(dis_image, cv::Point2i(j * x_step, 0),
               cv::Point2i(j * x_step, height),
               cv::Scalar(255, 255, 255), 1);
      uint16_t Z = depth_image.at<uint16_t>(i * y_step, j * x_step);
      double distance = static_cast<double>(Z) / 1000.0;
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(2) << distance << "m";

      cv::putText(rs_image, ss.str(), cv::Point2i(j * x_step + 3,
                                                  i * y_step - 3),
                  cv::FONT_HERSHEY_SIMPLEX, 1,
                  cv::Scalar(255, 255, 255), 2);

      cv::putText(dis_image, ss.str(), cv::Point2i(j * x_step + 3,
                                                   i * y_step - 3),
                  cv::FONT_HERSHEY_SIMPLEX, 1,
                  cv::Scalar(255, 255, 255), 2);

    }
  }

  cv::putText(rs_image, "realsense", cv::Point2i( 3, 38),
              cv::FONT_HERSHEY_SIMPLEX, 1.5,
              cv::Scalar(0, 255, 0), 3);

  cv::putText(dis_image, "relasense-depth", cv::Point2i(3, 38),
              cv::FONT_HERSHEY_SIMPLEX, 1.5,
              cv::Scalar(0, 255, 0), 3);

  {
    std::lock_guard<std::mutex> lck(compare_visual_mtx_);
    cv::vconcat(rs_image, dis_image, compare_visual_);
  }
}

}

RCLCPP_COMPONENTS_REGISTER_NODE(stereonet::StereoNetNode)
