// Copyright (c) 2022，Horizon Robotics.
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

#include <memory>
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "parser.h"

using hobot::dnn_node::DNNTensor;

namespace hobot {
namespace stereonet {

void ParseTensor(std::shared_ptr<DNNTensor> tensor,
                 std::vector<std::shared_ptr<StereonetResult>> &results);

int get_tensor_hw(std::shared_ptr<DNNTensor> tensor, int *height, int *width, int *chn);

void ParseTensor(std::shared_ptr<DNNTensor> tensor,
                 std::vector<std::shared_ptr<StereonetResult>> &results) {
  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

  //  int *shape = tensor->data_shape.d;
  int height, width, chn;
  auto ret = get_tensor_hw(tensor, &height, &width, &chn);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo5_detection_parser"),
                 "get_tensor_hw failed");
  }

  auto *data = reinterpret_cast<int32_t *>(tensor->sysMem[0].virAddr);
  // std::string ofname = "dump_output_"
  //     + std::to_string(width)
  //     + "_" + std::to_string(height)
  //     + "_" + std::to_string(chn)
  //     + "_" + std::to_string(tensor->sysMem[0].memSize) + ".log";
  // RCLCPP_INFO_STREAM(rclcpp::get_logger(""),
  //               "height: " << height
  //               << ", width: " << width
  //               << ", chn: " << chn
  //               << ", sysMem[0] memSize: " << tensor->sysMem[0].memSize
  //               << ", ofname: " << ofname);
  // std::ofstream ofs("output.bin");
  // ofs.write((const char*)data, tensor->sysMem[0].memSize);

  auto scale = tensor->properties.scale.scaleData;

  auto stereonet_result = std::make_shared<StereonetResult>();

  RCLCPP_DEBUG_STREAM(rclcpp::get_logger(""),
    "data[0]: " << data[0]
    << ", scale[0]: " << scale[0]
    << ", width: " << width
    << ", height: " << height);

  float f = 527.1931762695312; // 相机的焦距
  float B = 119.89382172; // 相机的baseline
  // Z = f*B/dis/1000
  // 深度信息，单位米
  std::vector<float>& result = stereonet_result->results;
  std::stringstream ss_depth;
  // 模型输出的定点转成float后的数据
  std::stringstream ss_float;
  std::stringstream ss_int;
  for (int c = 0; c < chn; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        int offset = c * (height * width) + h * width + w;
        // Dequantize
        float dis = static_cast<float>(data[offset]) * scale[c];
        // convert to depth
        result.push_back(f * B / (dis * 16.0 * 12.0) / 1000.0);
        ss_float << dis << " ";
        ss_depth << result.back() << " ";
        ss_int << data[offset] << "\n";
      }
      ss_float << "\n";
      ss_depth << "\n";
    }
  }
  
  // {
  //   std::ofstream ofs("out_data_int.txt");
  //   ofs << ss_int.str();
  // }
  // {
  //   std::ofstream ofs("out_data_float.txt");
  //   ofs << ss_float.str();
  // }
  // {
  //   std::ofstream ofs("out_data_depth.txt");
  //   ofs << ss_depth.str();
  // }


  // render
  cv::Mat img_out_f = cv::Mat(height, width, CV_32FC1);
  auto *data_img_out_f = img_out_f.ptr<float>();
  memcpy(data_img_out_f, result.data(), result.size() * sizeof(float));

  cv::Mat img_out_scale_abs;
  cv::convertScaleAbs(img_out_f, img_out_scale_abs, 11);
  cv::Mat img_out_color_map;
  cv::applyColorMap(img_out_scale_abs, img_out_color_map, cv::COLORMAP_JET);
  
  return;
  // {
  //   auto& data = img_out_color_map;
  //   cv::Mat nv12(height * 3 / 2, width, CV_8UC1, (char*)data.ptr<uint8_t>());
  //   cv::Mat bgr_mat;
  //   cv::cvtColor(nv12, bgr_mat, CV_YUV2BGR_NV12);  //  nv12 to bgr
  //   cv::imwrite("img_out_color_map.jpg", bgr_mat);

  //   // 使用opencv的imencode接口将mat转成vector，获取图片size
  //   std::vector<int> param;
  //   imencode(".jpg", bgr_mat, stereonet_result->jpeg, param);
    
  //   results.push_back(stereonet_result);
  //   return;
  // }


  {
    auto& data = img_out_f;
    RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
      "img_out_f mat col: " << data.cols
      << ", row: " << data.rows
      << ", dims: " << data.dims
      << ", channels: " << data.channels()
      << ", elemSize: " << data.elemSize()
      << ", data size: " << data.cols * data.rows * data.elemSize()
    );
  }
  {
    auto& data = img_out_color_map;
    RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
      "img_out_color_map mat col: " << data.cols
      << ", row: " << data.rows
      << ", dims: " << data.dims
      << ", channels: " << data.channels()
      << ", elemSize: " << data.elemSize()
      << ", data size: " << data.cols * data.rows * data.elemSize()
    );
    
    std::ofstream ofs("img_out_color_map.yuv");
    ofs.write((const char*)data.ptr<uint8_t>(), data.cols * data.rows * data.elemSize());
    
    cv::Mat nv12(height * 3 / 2, width, CV_8UC1, (char*)data.ptr<uint8_t>());
    cv::Mat bgr_mat;
    cv::cvtColor(nv12, bgr_mat, CV_YUV2BGR_NV12);  //  nv12 to bgr
    cv::imwrite("img_out_color_map.jpg", bgr_mat);
  }
}

int get_tensor_hw(std::shared_ptr<DNNTensor> tensor, int *height, int *width, int *chn) {
  int h_index = 0;
  int w_index = 0;
  int c_index = 0;
  if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    h_index = 1;
    w_index = 2;
    c_index = 3;
  } else if (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    c_index = 1;
    h_index = 2;
    w_index = 3;
  } else {
    return -1;
  }
  *height = tensor->properties.validShape.dimensionSize[h_index];
  *width = tensor->properties.validShape.dimensionSize[w_index];
  *chn = tensor->properties.validShape.dimensionSize[c_index];
  return 0;
}

int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::vector<std::shared_ptr<StereonetResult>> &results) {
  RCLCPP_DEBUG_STREAM(rclcpp::get_logger(""),
  "node_output->output_tensors.size(): " << node_output->output_tensors.size());
  ParseTensor(
      node_output->output_tensors[0], results);

  return 0;
}

}  // namespace stereonet
}  // namespace hobot