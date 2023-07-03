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

#include "preprocess.h"

namespace hobot {
namespace stereonet {

#include <iostream>  
#include <vector>  
#include <algorithm>  
#include <iterator>  
  
// 定义输入数组类型  
using Array = std::vector<double>;  
  
// 实现 numpy.concatenate() 函数  
Array concatenate(Array arr, Array arr2) {  
    std::copy(std::begin(arr2), std::end(arr2), std::back_inserter(arr));  
    return arr;  
}  
  
PreProcess::PreProcess(const std::string &config_file) {
}

int PreProcess::CvtImgData2Tensors(
    std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
    Model *pmodel,
    const std::shared_ptr<FeedbackData>& sp_feedback_data
    ) {
  if (!pmodel || !sp_feedback_data ||
      sp_feedback_data->image_files.size() != 2) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "Invalid input data");
    return -1;
  }

  hbDNNTensorProperties properties;
  pmodel->GetInputTensorProperties(properties, 0);
  int h_index = 1;
  int w_index = 2;
  int c_index = 3;
  if (properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    h_index = 1;
    w_index = 2;
    c_index = 3;
  } else if (properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    c_index = 1;
    h_index = 2;
    w_index = 3;
  }
  int in_h = properties.validShape.dimensionSize[h_index];
  int in_w = properties.validShape.dimensionSize[w_index];
  int c_stride = properties.validShape.dimensionSize[c_index];
  // auto w_stride = ALIGN_16(in_w);
  // int uv_height = in_h / 2;
  // int uv_width = in_w / 2;
  int y_size = in_h * in_w;
  int one_data_size = 1;

  RCLCPP_INFO_STREAM(rclcpp::get_logger("stereonet_node"),
  "The model in_w: " << in_w
  << ", in_h: " << in_h
  << ", c_stride: " << c_stride);

  std::shared_ptr<DNNTensor> dnn_tensor =
    std::shared_ptr<DNNTensor>(new DNNTensor(),
      [this](DNNTensor* tensors_ptr) {
        if (tensors_ptr) {
          if (tensors_ptr->sysMem[0].memSize > 0) {
            hbSysMem mem = {tensors_ptr->sysMem[0].phyAddr,
                            tensors_ptr->sysMem[0].virAddr,
                            tensors_ptr->sysMem[0].memSize};
            hbSysFreeMem(&mem);
          }

          delete tensors_ptr;
          tensors_ptr = nullptr;
        }
      });
  dnn_tensor->properties = properties;

  // tensor type is HB_DNN_TENSOR_TYPE_S8(val is 8)
  // TODO hbSysFreeMem
  hbSysAllocCachedMem(&dnn_tensor->sysMem[0], y_size * c_stride * one_data_size);
  
  // std::ofstream ofs("dump.bin");
  // ofs.write(reinterpret_cast<const char*>(dnn_tensor->sysMem[0].virAddr), y_size * c_stride * one_data_size);

  // hbSysAllocCachedMem(&dnn_tensor->sysMem[1], y_size * c_stride / 2);
  //内存初始化
  memset(dnn_tensor->sysMem[0].virAddr, 0, y_size * c_stride * one_data_size);
  // memset(dnn_tensor->sysMem[1].virAddr, 0, y_size / 2);


  // {
  //   // 示例输入数组  
  //   Array arr1 = {1, 2, 3};  
  //   Array arr2 = {4, 5, 6};  
  //   Array arr3 = {7, 8, 9};  
  //   Array arr4 = {10, 11, 12};  
  //   Array arr5 = {13, 14, 15};  
  
  //   // 使用 numpy.concatenate() 函数连接数组  
  //   Array result = concatenate(concatenate(arr1, arr2), concatenate(arr3, arr4));  
  //   result = concatenate(result, arr5);  
  
  //   // 输出结果  
  //   std::copy(std::begin(result), std::end(result), std::ostream_iterator<double>(std::cout, " "));  
  //   std::cout << std::endl;  
    
  //   rclcpp::shutdown();
  //   return -1;
  // }

  if (0)
  {
    for (size_t idx = 0; idx < sp_feedback_data->image_files.size(); idx++) {
      const auto& image_file = sp_feedback_data->image_files.at(idx);
      if (access(image_file.c_str(), F_OK) != 0) {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "File is not exist! image_file: " << image_file);
        return -1;
      }
      cv::Mat mat_tmp = cv::imread(image_file, cv::IMREAD_COLOR);
      if (mat_tmp.empty()) {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "Read file failed! image_file: " << image_file);
        return -1;
      }

      cv::Mat bgr_mat;
      bgr_mat.create(in_h, in_w, mat_tmp.type());
      cv::resize(mat_tmp, bgr_mat, bgr_mat.size(), 0, 0);
      // cv::imwrite("resized_img.jpg", bgr_mat);

      cv::Mat i420_mat;
      cv::cvtColor(bgr_mat, i420_mat, cv::COLOR_BGR2YUV_I420);
      {
        std::ofstream ofs("i420_mat.yuv");
        ofs.write((const char*)i420_mat.ptr<uint8_t>(), i420_mat.cols * i420_mat.rows * i420_mat.elemSize());
      }

      RCLCPP_INFO_STREAM(rclcpp::get_logger("hobot_stereonet"),
        "I420 mat col: " << i420_mat.cols
        << ", row: " << i420_mat.rows
        << ", dims: " << i420_mat.dims
        << ", channels: " << i420_mat.channels()
        << ", elemSize: " << i420_mat.elemSize());

      cv::Mat i444_mat;
      cv::cvtColor(bgr_mat, i444_mat, cv::COLOR_BGR2YUV);
      {
        std::ofstream ofs("i444_mat.yuv");
        ofs.write((const char*)i444_mat.ptr<uint8_t>(), i444_mat.cols * i444_mat.rows * i444_mat.elemSize());
      }


      RCLCPP_INFO_STREAM(rclcpp::get_logger("hobot_stereonet"),
        "I444 mat col: " << i444_mat.cols
        << ", row: " << i444_mat.rows
        << ", dims: " << i444_mat.dims
        << ", channels: " << i444_mat.channels()
        << ", elemSize: " << i444_mat.elemSize());

      // 将左右两张图像合并为一个图像  
      // cv::Mat img_concat;  
      // // cv::concatenate(i444_mat, i444_mat, img_concat, 2);
      // // CV_EXPORTS void hconcat(InputArray src1, InputArray src2, OutputArray dst);
      // cv::hconcat(i444_mat, i444_mat, img_concat);
      // RCLCPP_INFO_STREAM(rclcpp::get_logger("hobot_stereonet"),
      //   "hconcat mat col: " << img_concat.cols
      //   << ", row: " << img_concat.rows
      //   << ", dims: " << img_concat.dims
      //   << ", channels: " << img_concat.channels()
      //   << ", elemSize: " << img_concat.elemSize());

      // {
      //   // 将左右两张图像合并为一个图像  
      //   cv::Mat img_concat;  
      //   // cv::concatenate(i444_mat, i444_mat, img_concat, 2);
      //   // CV_EXPORTS void hconcat(InputArray src1, InputArray src2, OutputArray dst);
      //   cv::vconcat(i444_mat, i444_mat, img_concat);
      //   RCLCPP_INFO_STREAM(rclcpp::get_logger("hobot_stereonet"),
      //     "vconcat mat col: " << img_concat.cols
      //     << ", row: " << img_concat.rows
      //     << ", dims: " << img_concat.dims
      //     << ", channels: " << img_concat.channels()
      //     << ", elemSize: " << img_concat.elemSize());
      // }


      {
        using namespace cv;
        Mat& img = i444_mat;
        int dimY = i444_mat.rows;
        int dimX = i444_mat.cols;
        // bgr_mat.create(in_h, in_w, mat_tmp.type());
        // Mat data = Mat(dimY, dimX, CV_8UC(6));
        uint8_t *rawData = i444_mat.ptr<uint8_t>();

        std::vector<Mat> split_channels;
        split(i444_mat, split_channels);

        std::vector<Mat> channels{ split_channels[0], split_channels[1], split_channels[2], split_channels[0], split_channels[1], split_channels[2] };
        Mat outputMat;
        merge(channels, outputMat);
        
        auto& data = outputMat;
        
        // // cv::imwrite("merge.yuv", outputMat);
        std::ofstream ofs("merge.yuv");
        ofs.write((const char*)data.ptr<uint8_t>(), data.cols * data.rows * data.elemSize());
        // uint8_t *rawData = outputMat.ptr<uint8_t>();

        RCLCPP_INFO_STREAM(rclcpp::get_logger("hobot_stereonet"),
          "data mat col: " << data.cols
          << ", row: " << data.rows
          << ", dims: " << data.dims
          << ", channels: " << data.channels()
          << ", elemSize: " << data.elemSize()
          << ", split_channels size: " << split_channels.size());

      }

      
      rclcpp::shutdown();
      return -1;


      // auto *yuv = yuv_mat.ptr<uint8_t>();
      // cv::Mat img_nv12 = cv::Mat(in_h * 3 / 2, in_w, CV_8UC1);
      // auto *data = img_nv12.ptr<uint8_t>();

      // // copy y data
      // memcpy(data, yuv, y_size);

      // // copy uv data
      // uint8_t *nv12 = data + y_size;
      // uint8_t *u_data = yuv + y_size;
      // uint8_t *v_data = u_data + uv_height * uv_width;

      // for (int i = 0; i < uv_width * uv_height; i++) {
      //   *nv12++ = *u_data++;
      //   *nv12++ = *v_data++;
      // }
    }
  }


  std::vector<cv::Mat> split_channels;
  // split_channels.resize(c_stride);
  for (size_t idx = 0; idx < sp_feedback_data->image_files.size(); idx++) {
    const auto& image_file = sp_feedback_data->image_files.at(idx);
    if (access(image_file.c_str(), F_OK) != 0) {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "File is not exist! image_file: " << image_file);
      return -1;
    }
    // GetNV12Tensor(image_file, dnn_tensor, in_h, in_w);
    
    cv::Mat mat_tmp = cv::imread(image_file, cv::IMREAD_COLOR);
    if (mat_tmp.empty()) {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "Read file failed! image_file: " << image_file);
      return -1;
    }

    cv::Mat bgr_mat;
    bgr_mat.create(in_h, in_w, mat_tmp.type());
    cv::resize(mat_tmp, bgr_mat, bgr_mat.size(), 0, 0);
    // cv::imwrite("resized_img.jpg", bgr_mat);

    cv::Mat yuv_mat;
    cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV);
    {
      auto& data = yuv_mat;
      RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
        "data mat col: " << data.cols
        << ", row: " << data.rows
        << ", dims: " << data.dims
        << ", channels: " << data.channels()
        << ", elemSize: " << data.elemSize()
        << ", data size: " << data.cols * data.rows * data.elemSize()
      );
    }

    {
      // Convert image to float32 type
      cv::Mat image_float;
      yuv_mat.convertTo(image_float, CV_32F);
      {
        auto& data = image_float;
        RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
          "data mat col: " << data.cols
          << ", row: " << data.rows
          << ", dims: " << data.dims
          << ", channels: " << data.channels()
          << ", elemSize: " << data.elemSize()
          << ", data size: " << data.cols * data.rows * data.elemSize()
        );
      }
      // // Subtract mean from image
      // cv::Mat normalized_image = image_float - mean;
      // // Divide image by standard deviation
      // normalized_image = normalized_image / std;
      // return normalized_image;

      auto norm_channel = NormalizeImage(yuv_mat, 128.0, 128.0);
      {
        auto& data = norm_channel;
        RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
          "data mat col: " << data.cols
          << ", row: " << data.rows
          << ", dims: " << data.dims
          << ", channels: " << data.channels()
          << ", elemSize: " << data.elemSize()
          << ", data size: " << data.cols * data.rows * data.elemSize()
        );
      }
    }
    
    std::vector<cv::Mat> channels;
    cv::split(yuv_mat, channels);
    
    //归一化
    for (auto& channel : channels) {
      auto norm_channel = NormalizeImage(channel, 128.0, 128.0);
      {
        auto& data = channel;
        RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
          "data mat col: " << data.cols
          << ", row: " << data.rows
          << ", dims: " << data.dims
          << ", channels: " << data.channels()
          << ", elemSize: " << data.elemSize()
          << ", data size: " << data.cols * data.rows * data.elemSize()
        );
      }
      {
        auto& data = norm_channel;
        RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
          "data mat col: " << data.cols
          << ", row: " << data.rows
          << ", dims: " << data.dims
          << ", channels: " << data.channels()
          << ", elemSize: " << data.elemSize()
          << ", data size: " << data.cols * data.rows * data.elemSize()
        );
      }

      split_channels.emplace_back(norm_channel);
    }
    // split_channels.insert(split_channels.end(), channels.begin(), channels.end());
  }

  cv::Mat data;
  cv::merge(split_channels, data);
  // std::ofstream ofs("merge.yuv");
  // ofs.write((const char*)data.ptr<uint8_t>(), data.cols * data.rows * data.elemSize());

  RCLCPP_INFO_STREAM(rclcpp::get_logger("hobot_stereonet"),
    "data mat col: " << data.cols
    << ", row: " << data.rows
    << ", dims: " << data.dims
    << ", channels: " << data.channels()
    << ", elemSize: " << data.elemSize()
    << ", split_channels size: " << split_channels.size()
    << ", data size: " << data.cols * data.rows * data.elemSize()
    << ", " << dnn_tensor->sysMem[0].memSize
  );

  // auto tp_start = std::chrono::system_clock::now();
  // //归一化
  // // cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
  // data = NormalizeImage(data, 128.0, 128.0);

  // auto tp_now = std::chrono::system_clock::now();
  // auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
  //                     tp_now - tp_start)
  //                     .count();
  // RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "NormalizeImage time cost %d ms", interval);

  auto *y = reinterpret_cast<uint8_t *>(dnn_tensor->sysMem[0].virAddr);
  // memcpy(y, data.data, data.cols * data.rows * data.elemSize());
  memcpy(y, data.data, dnn_tensor->sysMem[0].memSize);

  hbSysFlushMem(&dnn_tensor->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  // hbSysFlushMem(&dnn_tensor->sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
  
  std::ofstream ofs("input_nchw.bin");
  ofs.write((const char*)dnn_tensor->sysMem[0].virAddr, dnn_tensor->sysMem[0].memSize);

  input_tensors.emplace_back(dnn_tensor);
  
  return 0;
}

int read_binary_file(std::string &file_path, char **bin, int *length) {
  std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs) {
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  *length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  *bin = new char[sizeof(char) * (*length)];
  ifs.read(*bin, *length);
  ifs.close();
  return 0;
}

void prepare_tensor_data(std::shared_ptr<hobot::easy_dnn::DNNTensor> tensor, char *data) {
  auto &tensor_property = tensor->properties;
  auto data_type = tensor_property.tensorType;
  char *vir_addr = reinterpret_cast<char *>(tensor->sysMem[0].virAddr);
  auto data_length = tensor->sysMem[0].memSize;
  memcpy(vir_addr, data, data_length);
  // flush_tensor(tensor);
}

int PreProcess::CvtBinData2Tensors(
    std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
    Model *pmodel,
    const std::shared_ptr<FeedbackData>& sp_feedback_data
    ) {
  if (!pmodel || !sp_feedback_data) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "Invalid input data");
    return -1;
  }

  if (access(sp_feedback_data->bin_file.c_str(), F_OK) != 0) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"),
    "File is not exist! bin_file: " << sp_feedback_data->bin_file);
    return -1;
  }

  RCLCPP_WARN_STREAM(rclcpp::get_logger("hobot_stereonet"),
  "Read data from bin_file: " << sp_feedback_data->bin_file);
  
  hbDNNTensorProperties properties;
  pmodel->GetInputTensorProperties(properties, 0);
  int h_index = 1;
  int w_index = 2;
  int c_index = 3;
  if (properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    h_index = 1;
    w_index = 2;
    c_index = 3;
  } else if (properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    c_index = 1;
    h_index = 2;
    w_index = 3;
  }
  int in_h = properties.validShape.dimensionSize[h_index];
  int in_w = properties.validShape.dimensionSize[w_index];
  int c_stride = properties.validShape.dimensionSize[c_index];
  // auto w_stride = ALIGN_16(in_w);
  // int uv_height = in_h / 2;
  // int uv_width = in_w / 2;
  int y_size = in_h * in_w;
  int one_data_size = 1;

  RCLCPP_INFO_STREAM(rclcpp::get_logger("stereonet_node"),
  "The model in_w: " << in_w
  << ", in_h: " << in_h
  << ", c_stride: " << c_stride);

  std::shared_ptr<DNNTensor> dnn_tensor =
    std::shared_ptr<DNNTensor>(new DNNTensor(),
      [this](DNNTensor* tensors_ptr) {
        if (tensors_ptr) {
          if (tensors_ptr->sysMem[0].memSize > 0) {
            hbSysMem mem = {tensors_ptr->sysMem[0].phyAddr,
                            tensors_ptr->sysMem[0].virAddr,
                            tensors_ptr->sysMem[0].memSize};
            hbSysFreeMem(&mem);
          }

          delete tensors_ptr;
          tensors_ptr = nullptr;
        }
      });
  dnn_tensor->properties = properties;

  hbSysAllocCachedMem(&dnn_tensor->sysMem[0], y_size * c_stride * one_data_size);
  memset(dnn_tensor->sysMem[0].virAddr, 0, y_size * c_stride * one_data_size);

  std::ifstream ifs(sp_feedback_data->bin_file, std::ios::in | std::ios::binary);
  if (!ifs) {
    return -1;
  }
  ifs.seekg(0, std::ios::end);
  int length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  printf("file len: %d\n", length);

  std::vector<int8_t> quantize_vals;

  // data in bin is float
  if (1)
  {
    std::vector<float32_t> f_vals;
    f_vals.resize(length / 4);
    // // float32_t bin;
    // // // while (!ifs.eof()) {
    // // for (int idx = 0; idx < length / 4; idx++) {
    // //   ifs.read((char*)&bin, 4);
    // //   f_vals.push_back(bin);
    // }
    
    ifs.read((char*)(&f_vals[0]), length);

    ifs.close();
    printf("f_vals len: %d\n", f_vals.size());
    
    std::stringstream ss_float;
    std::stringstream ss_int;
    
    int count = 0;
    for (const auto f_val : f_vals) {
      int8_t quant_val = Quantize(f_val);
      ss_float << f_val << " ";
      ss_int << quant_val << " ";
      quantize_vals.push_back(quant_val);
      if (++count >= 1280*720) {
        count = 0;
        ss_float << "\n";
        ss_int << "\n";
      }
    }
    
    {
      std::ofstream ofs("in_data_float.txt");
      ofs << ss_float.str();
    }
    {
      std::ofstream ofs("in_data_int.txt");
      ofs << ss_int.str();
    }
  }
  // data in bin is int
  else
  {
    quantize_vals.resize(length);
    ifs.read((char*)(&quantize_vals[0]), length);
  }

  auto scale = dnn_tensor->properties.scale.scaleData;
  printf("scaleLen: %d\n",
    dnn_tensor->properties.scale.scaleLen);
  for (int idx = 0; idx < dnn_tensor->properties.scale.scaleLen; idx++) {
    printf("%f ", scale[idx]);
  }
  printf("\n");

  uint8_t *y = reinterpret_cast<uint8_t *>(dnn_tensor->sysMem[0].virAddr);
  memcpy(y, quantize_vals.data(), quantize_vals.size());
  {
    std::ofstream ofs("in_data_nchw.bin");
    ofs.write((const char*)quantize_vals.data(), quantize_vals.size());
  }

  hbSysFlushMem(&dnn_tensor->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  input_tensors.emplace_back(dnn_tensor);


  // std::vector<int8_t> data_nhwc;
  // ncwh2nhwc(data_nhwc, quantize_vals.data(), 6, 720, 1280);
  // {
  //   std::ofstream ofs("in_data_nhwc.bin");
  //   ofs.write((const char*)data_nhwc.data(), data_nhwc.size());
  // }

  return 0;
}



int PreProcess::CvtBinData2Tensors(
    std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
    Model *pmodel,
    const std::string& img_l,
    const std::string& img_r
    ) {
  if (!pmodel) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "Invalid input data");
    return -1;
  }
  if (access(img_l.c_str(), F_OK) != 0) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "File is not exist! image_file: " << img_l);
    return -1;
  }
  if (access(img_r.c_str(), F_OK) != 0) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "File is not exist! image_file: " << img_r);
    return -1;
  }

  hbDNNTensorProperties properties;
  pmodel->GetInputTensorProperties(properties, 0);
  int h_index = 1;
  int w_index = 2;
  int c_index = 3;
  if (properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    h_index = 1;
    w_index = 2;
    c_index = 3;
  } else if (properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    c_index = 1;
    h_index = 2;
    w_index = 3;
  }
  int in_h = properties.validShape.dimensionSize[h_index];
  int in_w = properties.validShape.dimensionSize[w_index];
  int c_stride = properties.validShape.dimensionSize[c_index];
  // auto w_stride = ALIGN_16(in_w);
  // int uv_height = in_h / 2;
  // int uv_width = in_w / 2;
  int y_size = in_h * in_w;
  int one_data_size = 1;

  RCLCPP_INFO_STREAM(rclcpp::get_logger("stereonet_node"),
  "The model in_w: " << in_w
  << ", in_h: " << in_h
  << ", c_stride: " << c_stride);

  std::shared_ptr<DNNTensor> dnn_tensor =
    std::shared_ptr<DNNTensor>(new DNNTensor(),
      [this](DNNTensor* tensors_ptr) {
        if (tensors_ptr) {
          if (tensors_ptr->sysMem[0].memSize > 0) {
            hbSysMem mem = {tensors_ptr->sysMem[0].phyAddr,
                            tensors_ptr->sysMem[0].virAddr,
                            tensors_ptr->sysMem[0].memSize};
            hbSysFreeMem(&mem);
          }

          delete tensors_ptr;
          tensors_ptr = nullptr;
        }
      });
  dnn_tensor->properties = properties;

  // TODO hbSysFreeMem
  hbSysAllocCachedMem(&dnn_tensor->sysMem[0], y_size * c_stride * one_data_size);
  memset(dnn_tensor->sysMem[0].virAddr, 0, y_size * c_stride * one_data_size);

  std::vector<uint8_t> i_l_vals;
  std::vector<uint8_t> i_r_vals;
  {
    // std::vector<float> f_l_vals;
    std::ifstream ifs(img_l, std::ios::in | std::ios::binary);
    if (!ifs) {
      return -1;
    }
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    printf("file len: %d\n", length);
    
    char* buf = new char[length];
    ifs.read((char*)(buf), length);
    float* f_vals = (float*)buf;
    for (int idx = 0; idx < length/sizeof(float); idx++) {
      i_l_vals.push_back(f_vals[idx]);
    }
    delete []buf;
  }
  {
    // std::vector<float> f_r_vals;
    std::ifstream ifs(img_r, std::ios::in | std::ios::binary);
    if (!ifs) {
      return -1;
    }
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    printf("file len: %d\n", length);
    
    char* buf = new char[length];
    ifs.read((char*)(buf), length);
    float* f_vals = (float*)buf;
    for (int idx = 0; idx < length/sizeof(float); idx++) {
      i_r_vals.push_back(f_vals[idx]);
    }
    delete []buf;
  }


  cv::Mat img_l_yuv444 = cv::Mat(720, 1280, CV_8UC3);
  cv::Mat img_r_yuv444 = cv::Mat(720, 1280, CV_8UC3);
  {
    auto *data = img_l_yuv444.ptr<uint8_t>();
    memcpy(data, i_l_vals.data(), i_l_vals.size());
    printf("img_l_yuv444 size: %d, data : %d %d %d %d\n",
    i_l_vals.size(), data[0], data[1], data[2], data[3]);
  }
  {
    auto *data = img_r_yuv444.ptr<uint8_t>();
    memcpy(data, i_r_vals.data(), i_r_vals.size());
    printf("img_r_yuv444: %d %d %d %d\n", data[0], data[1], data[2], data[3]);
  }

  // dump merge c=6 img
  cv::Mat data_merge_nchw = cv::Mat(img_l_yuv444.rows, img_l_yuv444.cols, CV_8UC(6));
  auto *ptr_data_merge_nchw = data_merge_nchw.ptr<uint8_t>();
  memcpy(ptr_data_merge_nchw, img_l_yuv444.ptr<uint8_t>(), 1280*720*3);
  ptr_data_merge_nchw += 1280*720*3;
  memcpy(ptr_data_merge_nchw, img_r_yuv444.ptr<uint8_t>(), 1280*720*3);

  uint8_t* img_data_nchw = data_merge_nchw.data;
  int height = data_merge_nchw.rows;
  int width = data_merge_nchw.cols;
  int channels = data_merge_nchw.channels();
  int len = height * width * channels;

  printf("data_merge_nchw height: %d, width: %d, channels: %d\n\n", height, width, channels);

  std::vector<uint8_t> v_merge_nchw;
  v_merge_nchw.resize(len);
  memcpy(&v_merge_nchw[0], img_data_nchw, len);
    
  {
    std::ofstream ofs("in_data_merge_int_yuv444.bin");
    ofs.write((const char*)img_data_nchw, len);
  }

  {
      std::stringstream ss_int;
      int count = 0;
      for (auto& data : v_merge_nchw) {
        ss_int << (float)data << " ";
        if (++count >= 1280*720) {
          count = 0;
          ss_int << "\n";
        }
      }
      ss_int << "\n";
      std::ofstream ofs("in_data_merge_int_yuv444.txt");
      ofs << ss_int.str();
  }

  std::stringstream ss_norm;
  std::stringstream ss_quant;
  int count = 0;

  // 归一化
  count = 0;
  std::vector<float> v_merge_nchw_norm;
  printf("%d, %f\n", v_merge_nchw[0], ((float)v_merge_nchw[0] - 127.0) / 128.0);

  for (auto& data : v_merge_nchw) {
    v_merge_nchw_norm.push_back(((float)data - 128.0) / 128.0);
    ss_norm << v_merge_nchw_norm.back() << " ";
    
    if (++count >= 1280*720) {
      count = 0;
      ss_norm << "\n";
    }
  }
  ss_norm << "\n";
  printf("v_merge_nchw_norm: ");
  for (int idx = 0; idx < 12; idx++) {
    printf("%f ", v_merge_nchw_norm[idx]);
  }
  printf("\n\n");

  {
    std::ofstream ofs("in_data_merge_norm.bin");
    ofs.write((const char*)v_merge_nchw_norm.data(), v_merge_nchw_norm.size() * sizeof(float));
  }

  // 量化
  count = 0;
  std::vector<int8_t> v_merge_nchw_norm_quant;
  for (auto& data : v_merge_nchw_norm) {
    v_merge_nchw_norm_quant.push_back(Quantize(data));
    ss_quant << (float)v_merge_nchw_norm_quant.back() << " ";
    if (++count >= 1280*720) {
      count = 0;
      ss_quant << "\n";
    }
  }
  ss_quant << "\n";

  printf("v_merge_nchw_norm_quant: ");
  for (int idx = 0; idx < 12; idx++) {
    printf("%d ", v_merge_nchw_norm_quant[idx]);
  }
  printf("\n\n");

  {
    std::ofstream ofs("in_data_merge_norm.txt");
    ofs << ss_norm.str();
  }
  {
    std::ofstream ofs("in_data_merge_quant.txt");
    ofs << ss_quant.str();
  }
  uint8_t *y = reinterpret_cast<uint8_t *>(dnn_tensor->sysMem[0].virAddr);
  memcpy(y, v_merge_nchw_norm_quant.data(), v_merge_nchw_norm_quant.size());
  {
    std::ofstream ofs("in_data_merge_quant.bin");
    ofs.write((const char*)v_merge_nchw_norm_quant.data(), v_merge_nchw_norm_quant.size());
  }

  hbSysFlushMem(&dnn_tensor->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  input_tensors.emplace_back(dnn_tensor);

  return 0;
}


void PreProcess::GetNV12Tensor(const std::string &image_file,
                  std::shared_ptr<DNNTensor> &dnn_tensor,
                  int scaled_img_height, int scaled_img_width) {
  hbDNNTensorProperties properties = dnn_tensor->properties;
  cv::Mat mat_tmp = cv::imread(image_file, cv::IMREAD_COLOR);
  if (mat_tmp.empty()) {
    std::cout << "image file not exist!" << std::endl;
    return;
  }

  cv::Mat bgr_mat;
  bgr_mat.create(scaled_img_height, scaled_img_width, mat_tmp.type());
  cv::resize(mat_tmp, bgr_mat, bgr_mat.size(), 0, 0);
  cv::imwrite("resized_img.jpg", bgr_mat);

  auto height = bgr_mat.rows;
  auto width = bgr_mat.cols;

  cv::Mat yuv_mat;
  cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);

  auto tp_start = std::chrono::system_clock::now();
  //归一化
  // cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
  yuv_mat = NormalizeImage(yuv_mat, 128.0, 128.0);

  auto tp_now = std::chrono::system_clock::now();
  auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                      tp_now - tp_start)
                      .count();
  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "NormalizeImage time cost %d ms", interval);

  auto *yuv = yuv_mat.ptr<uint8_t>();
  cv::Mat img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
  auto *data = img_nv12.ptr<uint8_t>();

  int uv_height = height / 2;
  int uv_width = width / 2;

  // copy y data
  int y_size = height * width;
  memcpy(data, yuv, y_size);

  // copy uv data
  uint8_t *nv12 = data + y_size;
  uint8_t *u_data = yuv + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }

  int one_data_size = 1;
  int c_stride = properties.validShape.dimensionSize[1];
  auto w_stride = ALIGN_16(scaled_img_width);

  RCLCPP_INFO_STREAM(rclcpp::get_logger("stereonet_node"),
  "w_stride: " << w_stride
  << ", height: " << height
  << ", c_stride: " << c_stride);

  // tensor type is HB_DNN_TENSOR_TYPE_S8(val is 8)
  // TODO hbSysFreeMem
  hbSysAllocCachedMem(&dnn_tensor->sysMem[0], y_size * c_stride * one_data_size);
  
  // std::ofstream ofs("dump.bin");
  // ofs.write(reinterpret_cast<const char*>(dnn_tensor->sysMem[0].virAddr), y_size * c_stride * one_data_size);

  // hbSysAllocCachedMem(&dnn_tensor->sysMem[1], y_size * c_stride / 2);
  //内存初始化
  memset(dnn_tensor->sysMem[0].virAddr, 0, y_size * c_stride * one_data_size);
  // memset(dnn_tensor->sysMem[1].virAddr, 0, y_size / 2);

  auto stride = properties.alignedShape.dimensionSize[3];
  auto *y = reinterpret_cast<uint8_t *>(dnn_tensor->sysMem[0].virAddr);
  memcpy(y, img_nv12.data, height * width * 3 / 2);
  
  
  // char *vir_addr = reinterpret_cast<char *>(tensor->sysMem[0].virAddr);
  // auto data_length = tensor->sysMem[0].memSize;
  // memcpy(vir_addr, data, data_length);
  // flush_tensor(tensor);

  // auto *y_2 = y + height * width * 3;
  // memcpy(y_2, img_nv12.data, height * width * 3 / 2);

  hbSysFlushMem(&dnn_tensor->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  // hbSysFlushMem(&dnn_tensor->sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
}


int PreProcess::CvtNV12Data2Tensors(
      std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
      Model *pmodel,
      const unsigned char* img_l,
      const unsigned char* img_r
      ) {
  if (!pmodel || !img_l || !img_r) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "Invalid input data");
    return -1;
  }

  hbDNNTensorProperties properties;
  pmodel->GetInputTensorProperties(properties, 0);
  int h_index = 1;
  int w_index = 2;
  int c_index = 3;
  if (properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
    h_index = 1;
    w_index = 2;
    c_index = 3;
  } else if (properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
    c_index = 1;
    h_index = 2;
    w_index = 3;
  }
  int in_h = properties.validShape.dimensionSize[h_index];
  int in_w = properties.validShape.dimensionSize[w_index];
  int c_stride = properties.validShape.dimensionSize[c_index];
  // auto w_stride = ALIGN_16(in_w);
  // int uv_height = in_h / 2;
  // int uv_width = in_w / 2;
  int y_size = in_h * in_w;
  int one_data_size = 1;

  RCLCPP_INFO_STREAM(rclcpp::get_logger("stereonet_node"),
  "The model in_w: " << in_w
  << ", in_h: " << in_h
  << ", c_stride: " << c_stride);

  std::shared_ptr<DNNTensor> dnn_tensor =
    std::shared_ptr<DNNTensor>(new DNNTensor(),
      [this](DNNTensor* tensors_ptr) {
        if (tensors_ptr) {
          if (tensors_ptr->sysMem[0].memSize > 0) {
            hbSysMem mem = {tensors_ptr->sysMem[0].phyAddr,
                            tensors_ptr->sysMem[0].virAddr,
                            tensors_ptr->sysMem[0].memSize};
            hbSysFreeMem(&mem);
          }

          delete tensors_ptr;
          tensors_ptr = nullptr;
        }
      });
  dnn_tensor->properties = properties;

  auto tp_start = std::chrono::system_clock::now();

  // TODO hbSysFreeMem
  hbSysAllocCachedMem(&dnn_tensor->sysMem[0], y_size * c_stride * one_data_size);
  memset(dnn_tensor->sysMem[0].virAddr, 0, y_size * c_stride * one_data_size);

  cv::Mat img_l_yuv444 = cv::Mat(model_in_h_, model_in_w_, CV_8UC3);
  cv::Mat img_r_yuv444 = cv::Mat(model_in_h_, model_in_w_, CV_8UC3);

  {
    auto *data = img_l_yuv444.ptr<uint8_t>();
    // memcpy(data, img_l, 1280 * 720 * 1.5);
    Tools::YUV420TOYUV444(img_l, data, model_in_w_, model_in_h_);
  }
  {
    auto *data = img_r_yuv444.ptr<uint8_t>();
    // memcpy(data, img_r, 1280 * 720 * 1.5);
    Tools::YUV420TOYUV444(img_r, data, model_in_w_, model_in_h_);
  }

  {
    auto tp_now = std::chrono::system_clock::now();
    auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                        tp_now - tp_start)
                        .count();
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Preprocess yuv444, time cost %d ms", interval);
    tp_start = std::chrono::system_clock::now();
  }

  // dump merge c=6 img
  cv::Mat data_merge_nchw = cv::Mat(img_l_yuv444.rows, img_l_yuv444.cols, CV_8UC(6));
  auto *ptr_data_merge_nchw = data_merge_nchw.ptr<uint8_t>();
  memcpy(ptr_data_merge_nchw, img_l_yuv444.ptr<uint8_t>(), 1280*720*3);
  ptr_data_merge_nchw += 1280*720*3;
  memcpy(ptr_data_merge_nchw, img_r_yuv444.ptr<uint8_t>(), 1280*720*3);

  uint8_t* img_data_nchw = data_merge_nchw.data;
  int height = data_merge_nchw.rows;
  int width = data_merge_nchw.cols;
  int channels = data_merge_nchw.channels();
  int len = height * width * channels;

  // printf("data_merge_nchw height: %d, width: %d, channels: %d\n\n", height, width, channels);

  std::vector<uint8_t> v_merge_nchw;
  v_merge_nchw.resize(len);
  memcpy(&v_merge_nchw[0], img_data_nchw, len);

  {
    auto tp_now = std::chrono::system_clock::now();
    auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                        tp_now - tp_start)
                        .count();
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Preprocess merge chn 6 nchw, time cost %d ms", interval);
    tp_start = std::chrono::system_clock::now();
  }

  // {
  //   std::ofstream ofs("in_data_merge_int_yuv444.bin");
  //   ofs.write((const char*)img_data_nchw, len);
  // }

  // {
  //     std::stringstream ss_int;
  //     int count = 0;
  //     for (auto& data : v_merge_nchw) {
  //       ss_int << (float)data << " ";
  //       if (++count >= 1280*720) {
  //         count = 0;
  //         ss_int << "\n";
  //       }
  //     }
  //     ss_int << "\n";
  //     std::ofstream ofs("in_data_merge_int_yuv444.txt");
  //     ofs << ss_int.str();
  // }

  std::stringstream ss_norm;
  std::stringstream ss_quant;
  int count = 0;

  std::vector<int8_t> v_merge_nchw_norm_quant;
  // 原始方法
  if (0)
  {
    // 归一化
    count = 0;
    std::vector<float> v_merge_nchw_norm;
    // printf("%d, %f\n", v_merge_nchw[0], ((float)v_merge_nchw[0] - 127.0) / 128.0);

    for (auto& data : v_merge_nchw) {
      v_merge_nchw_norm.push_back(((float)data - 128.0) / 128.0);
      // ss_norm << v_merge_nchw_norm.back() << " ";
      
      // if (++count >= 1280*720) {
      //   count = 0;
      //   ss_norm << "\n";
      // }
    }
    // ss_norm << "\n";
    printf("v_merge_nchw_norm: ");
    for (int idx = 0; idx < 12; idx++) {
      printf("%f ", v_merge_nchw_norm[idx]);
    }
    printf("\n\n");

    // {
    //   std::ofstream ofs("in_data_merge_norm.bin");
    //   ofs.write((const char*)v_merge_nchw_norm.data(), v_merge_nchw_norm.size() * sizeof(float));
    // }
    
    {
      auto tp_now = std::chrono::system_clock::now();
      auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                          tp_now - tp_start)
                          .count();
      RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Preprocess norm time cost %d ms", interval);
      tp_start = std::chrono::system_clock::now();
    }

    // 量化
    count = 0;
    // std::vector<int8_t> v_merge_nchw_norm_quant;
    for (auto& data : v_merge_nchw_norm) {
      v_merge_nchw_norm_quant.push_back(Quantize(data));
      // ss_quant << (float)v_merge_nchw_norm_quant.back() << " ";
      // if (++count >= 1280*720) {
      //   count = 0;
      //   ss_quant << "\n";
      // }
    }
    // ss_quant << "\n";

    printf("v_merge_nchw_norm_quant: ");
    for (int idx = 0; idx < 12; idx++) {
      printf("%d ", v_merge_nchw_norm_quant[idx]);
    }
    printf("\n\n");
  }


  // norm和quant同时计算
  if (0)
  {
    // 归一化 & 量化
    count = 0;
    for (auto& data : v_merge_nchw) {
      v_merge_nchw_norm_quant.push_back(Quantize(((float)data - 128.0) / 128.0));
    }

    printf("v_merge_nchw_norm_quant: ");
    for (int idx = 0; idx < 12; idx++) {
      printf("%d ", v_merge_nchw_norm_quant[idx]);
    }
    printf("\n\n");
  }
  
  // norm和quant同时计算，基于stl计算
  if (1)
  {
    // 归一化 & 量化
    count = 0;
    for (auto& data : v_merge_nchw) {
      v_merge_nchw_norm_quant.push_back(Quantize(((float)data - 128.0) / 128.0));
    }

    printf("v_merge_nchw_norm_quant: ");
    for (int idx = 0; idx < 12; idx++) {
      printf("%d ", v_merge_nchw_norm_quant[idx]);
    }
    printf("\n\n");
  }


  {
    auto tp_now = std::chrono::system_clock::now();
    auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                        tp_now - tp_start)
                        .count();
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Preprocess norm and quantize time cost %d ms", interval);
    tp_start = std::chrono::system_clock::now();
  }

  {
    std::ofstream ofs("in_data_merge_norm.txt");
    ofs << ss_norm.str();
  }
  {
    std::ofstream ofs("in_data_merge_quant.txt");
    ofs << ss_quant.str();
  }

  uint8_t *y = reinterpret_cast<uint8_t *>(dnn_tensor->sysMem[0].virAddr);
  memcpy(y, v_merge_nchw_norm_quant.data(), v_merge_nchw_norm_quant.size());
  // {
  //   std::ofstream ofs("in_data_merge_quant.bin");
  //   ofs.write((const char*)v_merge_nchw_norm_quant.data(), v_merge_nchw_norm_quant.size());
  // }

  hbSysFlushMem(&dnn_tensor->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  input_tensors.emplace_back(dnn_tensor);

  return 0;
}

int PreProcess::CvtNV12File2Tensors(
    std::vector<std::shared_ptr<DNNTensor>> &input_tensors,
    Model *pmodel,
    std::string img_l,
    std::string img_r
    ) {
  if (access(img_l.c_str(), F_OK) != 0 || access(img_r.c_str(), F_OK) != 0) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "File is not exist! img_l: " << img_l << ", img_r: " << img_r);
    return -1;
  }
  
  unsigned char* img_l_data;
  unsigned char* img_r_data;
  {
    std::ifstream ifs(img_l.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      return -1;
    }
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    img_l_data = new unsigned char[sizeof(unsigned char) * length];
    ifs.read((char*)(img_l_data), length);
  }
  {
    std::ifstream ifs(img_r.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      return -1;
    }
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    img_r_data = new unsigned char[sizeof(unsigned char) * length];
    ifs.read((char*)img_r_data, length);
    printf("length: %d\n", length);
  }

  CvtNV12Data2Tensors(input_tensors, pmodel, img_l_data, img_r_data);
  
  delete []img_l_data;
  delete []img_r_data;
  return 0;
}

std::shared_ptr<DNNTensor> PreProcess::GetNV12Pyramid(const std::string &image_file,
                                                      int scaled_img_height,
                                                      int scaled_img_width) {
  int original_img_height = 0, original_img_width = 0;
  return GetNV12Pyramid(image_file, scaled_img_height, scaled_img_width,
                        original_img_height, original_img_width);
}

std::shared_ptr<DNNTensor> PreProcess::GetNV12Pyramid(const std::string &image_file,
                                                      int scaled_img_height,
                                                      int scaled_img_width,
                                                      int &original_img_height,
                                                      int &original_img_width) {
  cv::Mat nv12_mat;
  cv::Mat bgr_mat = cv::imread(image_file, cv::IMREAD_COLOR);
  cv::Mat mat_tmp;
  mat_tmp.create(scaled_img_height, scaled_img_width, bgr_mat.type());
  cv::resize(bgr_mat, mat_tmp, mat_tmp.size());
  auto ret = PreProcess::BGRToNv12(mat_tmp, nv12_mat);
  if (ret) {
    std::cout << "get nv12 image failed " << std::endl;
    return nullptr;
  }
  original_img_height = bgr_mat.rows;
  original_img_width = bgr_mat.cols;

  auto *y = new hbSysMem;
  auto *uv = new hbSysMem;

  auto w_stride = ALIGN_16(scaled_img_width);
  hbSysAllocCachedMem(y, scaled_img_height * w_stride);
  hbSysAllocCachedMem(uv, scaled_img_height / 2 * w_stride);

  uint8_t *data = nv12_mat.data;
  auto *hb_y_addr = reinterpret_cast<uint8_t *>(y->virAddr);
  auto *hb_uv_addr = reinterpret_cast<uint8_t *>(uv->virAddr);

  // padding y
  for (int h = 0; h < scaled_img_height; ++h) {
    auto *raw = hb_y_addr + h * w_stride;
    for (int w = 0; w < scaled_img_width; ++w) {
      *raw++ = *data++;
    }
  }

  // padding uv
  auto uv_data = nv12_mat.data + scaled_img_height * scaled_img_width;
  for (int32_t h = 0; h < scaled_img_height / 2; ++h) {
    auto *raw = hb_uv_addr + h * w_stride;
    for (int32_t w = 0; w < scaled_img_width; ++w) {
      *raw++ = *uv_data++;
    }
  }

  hbSysFlushMem(y, HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(uv, HB_SYS_MEM_CACHE_CLEAN);
  auto input_tensor = new DNNTensor;
  input_tensor->sysMem[0].virAddr = reinterpret_cast<void *>(y->virAddr);
  input_tensor->sysMem[0].phyAddr = y->phyAddr;
  input_tensor->sysMem[0].memSize = scaled_img_height * scaled_img_width;
  input_tensor->sysMem[1].virAddr = reinterpret_cast<void *>(uv->virAddr);
  input_tensor->sysMem[1].phyAddr = uv->phyAddr;
  input_tensor->sysMem[1].memSize = scaled_img_height * scaled_img_width / 2;
  // auto pyramid = new NV12PyramidInput;
  // pyramid->width = scaled_img_width;
  // pyramid->height = scaled_img_height;
  // pyramid->y_vir_addr = y->virAddr;
  // pyramid->y_phy_addr = y->phyAddr;
  // pyramid->y_stride = w_stride;
  // pyramid->uv_vir_addr = uv->virAddr;
  // pyramid->uv_phy_addr = uv->phyAddr;
  // pyramid->uv_stride = w_stride;
  return std::shared_ptr<DNNTensor>(
      input_tensor, [y, uv](DNNTensor *input_tensor) {
        // Release memory after deletion
        std::cout << "Release input_tensor" << std::endl;
        hbSysFreeMem(y);
        hbSysFreeMem(uv);
        delete y;
        delete uv;
        delete input_tensor;
      });
}

int32_t PreProcess::BGRToNv12(cv::Mat &bgr_mat, cv::Mat &img_nv12) {
  auto height = bgr_mat.rows;
  auto width = bgr_mat.cols;

  if (height % 2 || width % 2) {
    std::cerr << "input img height and width must aligned by 2!";
    return -1;
  }
  cv::Mat yuv_mat;
  cv::cvtColor(bgr_mat, yuv_mat, cv::COLOR_BGR2YUV_I420);
  if (yuv_mat.data == nullptr) {
    std::cerr << "yuv_mat.data is null pointer" << std::endl;
    return -1;
  }

  auto *yuv = yuv_mat.ptr<uint8_t>();
  if (yuv == nullptr) {
    std::cerr << "yuv is null pointer" << std::endl;
    return -1;
  }
  img_nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
  auto *ynv12 = img_nv12.ptr<uint8_t>();

  int32_t uv_height = height / 2;
  int32_t uv_width = width / 2;

  // copy y data
  int32_t y_size = height * width;
  memcpy(ynv12, yuv, y_size);

  // copy uv data
  uint8_t *nv12 = ynv12 + y_size;
  uint8_t *u_data = yuv + y_size;
  uint8_t *v_data = u_data + uv_height * uv_width;

  for (int32_t i = 0; i < uv_width * uv_height; i++) {
    *nv12++ = *u_data++;
    *nv12++ = *v_data++;
  }
  return 0;
}

void PreProcess::FreeTensors(
    const std::vector<std::shared_ptr<DNNTensor>> &input_tensors) {
  for (auto i = 0; i < input_tensors.size(); i++) {
    auto input_tensor = input_tensors[i];
    hbSysFreeMem(&input_tensor->sysMem[0]);
    hbSysFreeMem(&input_tensor->sysMem[1]);
  }
  return;
}
// mean: 128.0
// std: 128.0
cv::Mat PreProcess::NormalizeImage(cv::Mat image, float mean, float std) {
  // Convert image to float32 type
  cv::Mat image_float;
  image.convertTo(image_float, CV_32F);
  // Subtract mean from image
  cv::Mat normalized_image = image_float - mean;
  // Divide image by standard deviation
  normalized_image = normalized_image / std;
  return normalized_image;
}

// scale: 0.0078125
// zero_point: 0.5
// min: -128
// max: 127
int8_t PreProcess::Quantize(float32_t value, float32_t const scale, float32_t const zero_point, float32_t const min, float32_t const max) {
  // value = std::round(value / scale + zero_point);
  value = std::floor(value / scale + zero_point);
  value = std::min(std::max(value, min), max);
  return static_cast<int8_t>(value);
}

}  // namespace stereonet
}  // namespace hobot
