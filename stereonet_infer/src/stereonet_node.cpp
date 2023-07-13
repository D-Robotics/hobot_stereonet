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

#include <iostream>
#include <string>

#include "stereonet_node.h"

namespace hobot {
namespace stereonet {

StereonetNode::StereonetNode(const std::string& node_name,
                             const rclcpp::NodeOptions& options)
    : hobot::dnn_node::DnnNode(node_name, options) {
  this->declare_parameter<std::string>("config_file", config_file_);
  this->declare_parameter<std::string>("model_file", model_file_);
  this->declare_parameter<std::string>("sub_hbmem_topic_name", sub_hbmem_topic_name_);
  this->declare_parameter<std::string>("ros_img_topic_name", ros_img_topic_name_);

  this->get_parameter<std::string>("config_file", config_file_);
  this->get_parameter<std::string>("model_file", model_file_);
  this->get_parameter<std::string>("sub_hbmem_topic_name", sub_hbmem_topic_name_);
  this->get_parameter<std::string>("ros_img_topic_name", ros_img_topic_name_);

  RCLCPP_WARN_STREAM(rclcpp::get_logger("stereonet_node"),
    "\n config_file: " << config_file_
    << "\n model_file: " << model_file_
    << "\n sub_hbmem_topic_name: " << sub_hbmem_topic_name_
    << "\n ros_img_topic_name: " << ros_img_topic_name_);

  // Init中使用StereonetNode子类实现的SetNodePara()方法进行算法推理的初始化
  if (Init() != 0 ||
      GetModelInputSize(0, model_input_width_, model_input_height_) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Node init fail!");
    rclcpp::shutdown();
    return;
  }

  model_ = GetModel();
  if (!model_) {
    RCLCPP_ERROR(rclcpp::get_logger(""), "Invalid model");
    rclcpp::shutdown();
    return;
  }
  auto model_input_count = model_->GetInputCount();
  RCLCPP_WARN_STREAM(rclcpp::get_logger("stereonet_node"),
    "model_input_count: " << model_input_count
    << ", model_input_width: " << model_input_width_
    << ", model_input_height: " << model_input_height_);

  for (int idx = 0; idx < model_input_count; idx++) {
    hbDNNTensorProperties properties;
    model_->GetInputTensorProperties(properties, idx);
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"),
    "properties: %d, %d, %d, %d",
    properties.validShape.dimensionSize[0],
    properties.validShape.dimensionSize[1],
    properties.validShape.dimensionSize[2],
    properties.validShape.dimensionSize[3]);
  }

  hbDNNHandle_t dnn_model_handle = model_->GetDNNHandle();
  int input_num = model_->GetInputCount();
  input_model_info_.resize(input_num);
  for (int input_idx = 0; input_idx < input_num; input_idx++) {
    hbDNNGetInputTensorProperties(
        &input_model_info_[input_idx], dnn_model_handle, input_idx);

    std::stringstream ss;
    ss << "input_idx: " << input_idx
       << ", tensorType = " << input_model_info_[input_idx].tensorType
       << ", tensorLayout = " << input_model_info_[input_idx].tensorLayout;
    RCLCPP_INFO(
        rclcpp::get_logger(""), "%s", ss.str().c_str());
  }



  int output_num = model_->GetOutputCount();
  output_model_info_.resize(output_num);
  for (int output_idx = 0; output_idx < output_num; output_idx++) {
    hbDNNGetOutputTensorProperties(
        &output_model_info_[output_idx], dnn_model_handle, output_idx);

    std::stringstream ss;
    ss << "output_idx: " << output_idx
       << ", tensorType = " << output_model_info_[output_idx].tensorType
       << ", tensorLayout = " << output_model_info_[output_idx].tensorLayout;
    RCLCPP_WARN(
        rclcpp::get_logger(""), "%s", ss.str().c_str());
  }

  sp_preprocess_ = std::make_shared<PreProcess>("");

  // 创建消息订阅者，从摄像头节点订阅图像消息
  subscription_hbmem_img_ =
      this->create_subscription_hbmem<hbm_img_msgs::msg::HbmMsg1080P>(
          sub_hbmem_topic_name_,
          10,
          std::bind(&StereonetNode::FeedImg, this, std::placeholders::_1));
  // 创建消息发布者，发布算法推理消息
  msg_publisher_ = this->create_publisher<ai_msgs::msg::PerceptionTargets>(
      "/Stereonet_node_sample", 10);

  ros_img_publisher_ =
      this->create_publisher<sensor_msgs::msg::Image>(ros_img_topic_name_, 10);

  // RunImglistFeedInfer("left_img.list", "right_img.list");

  // RunImgFeedInfer();
  // RunBinFeedInfer();
  // timer_ = this->create_wall_timer(
  //     std::chrono::milliseconds(static_cast<int64_t>(30)),
  //     std::bind(&StereonetNode::RunBinFeedInfer, this));
}

int StereonetNode::SetNodePara() {
  if (!dnn_node_para_ptr_) return -1;
  if (access(model_file_.c_str(), F_OK) != 0) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("hobot_stereonet"), "File is not exist! model_file: " << model_file_);
    return -1;
  }

  dnn_node_para_ptr_->model_file = model_file_;
  
  // 指定算法推理任务类型
  // 本示例使用的人体检测算法输入为单张图片，对应的算法推理任务类型为ModelInferType
  // 只有当算法输入为图片和roi（Region of
  // Interest，例如目标的检测框）时，算法推理任务类型为ModelRoiInferType
  dnn_node_para_ptr_->model_task_type =
      hobot::dnn_node::ModelTaskType::ModelInferType;
  dnn_node_para_ptr_->task_num = 4;

  return 0;
}

void StereonetNode::RunImgFeedInfer() {
  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Run infer start");

  auto model = GetModel();
  if (!model) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Invalid model!");
    return;
  }
 
  auto tp_start = std::chrono::system_clock::now();
  auto sp_feedback_data = std::make_shared<FeedbackData>();
  std::vector<std::shared_ptr<DNNTensor>> input_tensors;
  if (sp_preprocess_->CvtImgData2Tensors(input_tensors, model, sp_feedback_data) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Preprocess fail");
    rclcpp::shutdown();
    return;
  }
  
  auto tp_now = std::chrono::system_clock::now();
  auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                      tp_now - tp_start)
                      .count();
  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Preprocess done, time cost %d ms", interval);

  std::vector<std::shared_ptr<hobot::dnn_node::OutputDescription>> output_descs{};
  auto dnn_output = std::make_shared<StereonetNodeOutput>();
  dnn_output->preprocess_time_ms = interval;
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->image_files = sp_feedback_data->image_files;
  if (Run(input_tensors, output_descs, dnn_output, true, -1, -1) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Run infer fail!");
    return;
  }

  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Run infer done");
}

void Dump(std::string in_file, std::string out_file) {
  {
    std::vector<uint8_t> i_l_vals;
    std::ifstream ifs(in_file, std::ios::in | std::ios::binary);
    if (!ifs) {
      printf("Open file %s fail!\n", in_file.data());
      return;
    }
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    printf("file: %s\n", in_file.data());
    
    char* buf = new char[length];
    ifs.read((char*)(buf), length);
    float* f_vals = (float*)buf;
    std::stringstream ss;
    int count = 0;
    
    printf("f_vals: ");
    for (int idx = 0; idx < 12; idx++) {
      printf("%f ", f_vals[idx]);
    }
    printf("\n\n");
    
    for (int idx = 0; idx < length/sizeof(float); idx++) {
      i_l_vals.push_back(f_vals[idx]);
      int data = f_vals[idx];
      ss << (float)data << " ";
      if (++count >= 1280*720) {
        count = 0;
        ss << "\n";
      }
    }
    delete []buf;

    std::ofstream ofs(out_file);
    ofs << ss.str();
  }
}
void DumpFloat(std::string in_file, std::string out_file) {
  std::ifstream ifs(in_file, std::ios::in | std::ios::binary);
  if (!ifs) {
    printf("Open file %s fail!\n", in_file.data());
    return;
  }
  ifs.seekg(0, std::ios::end);
  int length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  printf("file: %s\n", in_file.data());
  
  char* buf = new char[length];
  ifs.read((char*)(buf), length);
  float* f_vals = (float*)buf;
  std::stringstream ss;
  int count = 0;
  
  printf("f_vals: ");
  for (int idx = 0; idx < 12; idx++) {
    printf("%f ", f_vals[idx]);
  }
  printf("\n\n");
  
  for (int idx = 0; idx < length/sizeof(float); idx++) {
    ss << (float)f_vals[idx] << " ";
    if (++count >= 1280*720) {
      count = 0;
      ss << "\n";
    }
  }
  delete []buf;

  std::ofstream ofs(out_file);
  ofs << ss.str();
}

void RenderOutput(std::string in_file, std::string out_file) {
  std::ifstream ifs(in_file, std::ios::in | std::ios::binary);
  if (!ifs) {
    printf("Open file %s fail!\n", in_file.data());
    return;
  }
  ifs.seekg(0, std::ios::end);
  int length = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  printf("file: %s\n", in_file.data());
  
  char* buf = new char[length];
  ifs.read((char*)(buf), length);


  auto tp_start = std::chrono::system_clock::now();


  float* f_data = (float*)buf;
  float f = 527.1931762695312; // 相机的焦距
  float B = 119.89382172; // 相机的baseline
  float scale = 0.00000260443857769133;
  std::vector<float> result{};
  for (int offset = 0; offset < length / sizeof(float); ++offset) {
    // Dequantize
    float dis = static_cast<float>(f_data[offset]) * scale;
    // convert to depth
    result.push_back(f * B / (dis * 16.0 * 12.0) / 1000.0);
  }

  int vec_len = length / sizeof(float);
  printf("f_data: %f %f %f   %f %f %f\n",
      f_data[0], f_data[1], f_data[2], f_data[vec_len - 3], f_data[vec_len - 2], f_data[vec_len - 1]);
  
  printf("result: %f %f %f   %f %f %f\n",
      result[0], result[1], result[2], result[vec_len - 3], result[vec_len - 2], result[vec_len - 1]);

  int height = 720;
  int width = 1280;
  
  // {
  //   cv::Mat img_8uc1 = cv::Mat(height, width, CV_8UC1);
  //   auto& data = img_8uc1;
  //   RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
  //     "img_8uc1 col: " << data.cols
  //     << ", row: " << data.rows
  //     << ", dims: " << data.dims
  //     << ", channels: " << data.channels()
  //     << ", elemSize: " << data.elemSize()
  //     << ", data size: " << data.cols * data.rows * data.elemSize()
  //   );
  // }


  cv::Mat img_out_f = cv::Mat(height, width, CV_32FC1);
  auto *data_img_out_f = img_out_f.ptr<float>();
  memcpy(data_img_out_f, result.data(), length);
  
  // {
  //   auto& data = data_img_out_f;
  //   RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
  //     "data_img_out_f col: " << data.cols
  //     << ", row: " << data.rows
  //     << ", dims: " << data.dims
  //     << ", channels: " << data.channels()
  //     << ", elemSize: " << data.elemSize()
  //     << ", data size: " << data.cols * data.rows * data.elemSize()
  //   );
  // }


  cv::Mat img_out_scale_abs;
  cv::convertScaleAbs(img_out_f, img_out_scale_abs, 11);

  {
    auto& data = img_out_scale_abs;
    RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
      "img_out_scale_abs col: " << data.cols
      << ", row: " << data.rows
      << ", dims: " << data.dims
      << ", channels: " << data.channels()
      << ", elemSize: " << data.elemSize()
      << ", data size: " << data.cols * data.rows * data.elemSize()
    );
  }
  {
    uint8_t* data = img_out_scale_abs.ptr<uint8_t>();
    int len = img_out_scale_abs.cols * img_out_scale_abs.rows * img_out_scale_abs.elemSize();
    printf("img_out_scale_abs: %d %d %d   %d %d %d\n",
        data[0], data[1], data[2], data[len - 3], data[len - 2], data[len - 1]);
  }

  // yuv444/IYU2
  cv::Mat img_out_color_map;
  cv::applyColorMap(img_out_scale_abs, img_out_color_map, cv::COLORMAP_JET);
  
  
  {
    auto tp_now = std::chrono::system_clock::now();
    auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                        tp_now - tp_start)
                        .count();
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "render time cost %d ms", interval);
  }

  {
    auto& data = img_out_color_map;
    RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
      "img_out_color_map col: " << data.cols
      << ", row: " << data.rows
      << ", dims: " << data.dims
      << ", channels: " << data.channels()
      << ", elemSize: " << data.elemSize()
      << ", data size: " << data.cols * data.rows * data.elemSize()
    );
  }
  {
    uint8_t* data = img_out_color_map.ptr<uint8_t>();
    int len = img_out_color_map.cols * img_out_color_map.rows * img_out_color_map.elemSize();
    printf("img_out_color_map: %d %d %d   %d %d %d\n",
        data[0], data[1], data[2], data[len - 3], data[len - 2], data[len - 1]);
  }

  {
    std::ofstream ofs(out_file);
    ofs.write((const char*)img_out_color_map.ptr<uint8_t>(), img_out_color_map.cols * img_out_color_map.rows * img_out_color_map.elemSize());
  }

  cv::Mat yuv420(height * 3 / 2, width, CV_8UC1);
  {
    auto& data = yuv420;
    RCLCPP_INFO_STREAM_ONCE(rclcpp::get_logger("hobot_stereonet"),
      "yuv420 col: " << data.cols
      << ", row: " << data.rows
      << ", dims: " << data.dims
      << ", channels: " << data.channels()
      << ", elemSize: " << data.elemSize()
      << ", data size: " << data.cols * data.rows * data.elemSize()
    );
  }
  Tools::YUV444TOYUV420((const unsigned char*)img_out_color_map.ptr<uint8_t>(),
      (unsigned char *)(yuv420.ptr<uint8_t>()), img_out_color_map.rows, img_out_color_map.cols);

  {
    std::ofstream ofs("yuv420.yuv");
    ofs.write((const char*)yuv420.ptr<uint8_t>(), yuv420.cols * yuv420.rows * yuv420.elemSize());
  }

  cv::Mat img_nv12(height * 3 / 2, width, CV_8UC1);
  // yuv420转nv12
  {
    auto *yuv = yuv420.ptr<uint8_t>();
    auto *ynv12 = img_nv12.ptr<uint8_t>();

    int32_t uv_height = height / 2;
    int32_t uv_width = width / 2;

    int32_t y_size = height * width;
    memcpy(ynv12, yuv, y_size);

    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;

    for (int32_t i = 0; i < uv_width * uv_height; i++) {
      *nv12++ = *u_data++;
      *nv12++ = *v_data++;
    }
  }

  {
    std::ofstream ofs("img.nv12");
    ofs.write((const char*)img_nv12.ptr<uint8_t>(), img_nv12.cols * img_nv12.rows * img_nv12.elemSize());
  }

  cv::Mat bgr_mat;
  cv::cvtColor(img_nv12, bgr_mat, CV_YUV2BGR_NV12);  //  nv12 to bgr
  cv::imwrite("img_out_color_map.jpg", bgr_mat);
}

void StereonetNode::RunBinFeedInfer() {
  
  // test
  // only publish nv12
  if (0)
  {
    auto msg = sensor_msgs::msg::Image();
    msg.height = 720;
    msg.width = 1280;
    msg.encoding = "jpeg";

    int infer_out_data_len = msg.width * msg.height * 4;
    char* infer_out_data = new char[infer_out_data_len];
    std::ifstream ifs_infer("output.bin");
    if (!ifs_infer) {
      RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"),
                  "read file fail!");
      return;
    }
    ifs_infer.read(infer_out_data, infer_out_data_len);

    int nv12_data_len = msg.width * msg.height *1.5;
    char* data = new char[nv12_data_len];
    std::ifstream ifs("frame_1280_720_left.nv12");
    if (!ifs) {
      RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"),
                  "read file fail!");
      return;
    }
    ifs.read(data, nv12_data_len);
    cv::Mat nv12(msg.height * 3 / 2, msg.width, CV_8UC1, data);
    cv::Mat bgr_mat;
    cv::cvtColor(nv12, bgr_mat, CV_YUV2BGR_NV12);  //  nv12 to bgr
    cv::imwrite("cvt_nv12.jpg", bgr_mat);
    delete []data;

    // 使用opencv的imencode接口将mat转成vector，获取图片size
    std::vector<uint8_t> jpeg;
    std::vector<int> param;
    imencode(".jpg", bgr_mat, jpeg, param);
    auto img_data = jpeg.data();

    auto data_len = infer_out_data_len + jpeg.size();
    msg.data.resize(data_len);
    msg.step = data_len;

    char* msg_data_buf = (char*)&msg.data[0];
    memcpy(msg_data_buf, infer_out_data, infer_out_data_len);
    msg_data_buf += infer_out_data_len;
    memcpy(msg_data_buf, img_data, jpeg.size());

    delete []infer_out_data;

    ros_img_publisher_->publish(std::move(msg));
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"),
                  "publish");
    return;
  }


  // image1_yuv444.bin  image2_yuv444.bin  image_yuv444_concat.bin  img_adapter.bin
  // Dump("numpy/image1_yuv444.bin", "image1_yuv444.txt");
  // Dump("numpy/image2_yuv444.bin", "image2_yuv444.txt");
  // Dump("numpy/image_yuv444_concat.bin", "image_yuv444_concat.txt");
  // DumpFloat("numpy/img_adapter.bin", "img_adapter.txt");
  // rclcpp::shutdown();
  // return;
  
  // RenderOutput("./output.bin", "img_out_color_map.yuv");
  // rclcpp::shutdown();
  // return;

  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Run infer start");

  auto model = GetModel();
  if (!model) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Invalid model!");
    return;
  }
 
  auto tp_start = std::chrono::system_clock::now();
  auto sp_feedback_data = std::make_shared<FeedbackData>();

  std::vector<std::shared_ptr<DNNTensor>> input_tensors;

  // if (sp_preprocess_->CvtBinData2Tensors(input_tensors, model, sp_feedback_data) < 0) {
  //   RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Preprocess fail");
  //   rclcpp::shutdown();
  //   return;
  // }

  // if (sp_preprocess_->CvtBinData2Tensors(input_tensors, model, "numpy/image1_yuv444.bin", "numpy/image2_yuv444.bin") < 0) {
  //   RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Preprocess fail");
  //   rclcpp::shutdown();
  //   return;
  // }

  
  if (sp_preprocess_->CvtNV12File2Tensors(input_tensors, model, "frame_1280_720_left.nv12", "frame_1280_720_right.nv12") < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Preprocess fail");
    rclcpp::shutdown();
    return;
  }

  auto tp_now = std::chrono::system_clock::now();
  auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                      tp_now - tp_start)
                      .count();
  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Preprocess done, time cost %d ms", interval);

  std::vector<std::shared_ptr<hobot::dnn_node::OutputDescription>> output_descs{};
  auto dnn_output = std::make_shared<StereonetNodeOutput>();
  dnn_output->preprocess_time_ms = interval;
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->image_files = sp_feedback_data->image_files;

  BinDataType* bin_data = new BinDataType();
  bin_data->len = model_input_width_ * model_input_height_ * 3 / 2;
  bin_data->data = new char[bin_data->len];
  dnn_output->sp_left_nv12 = std::shared_ptr<BinDataType>(bin_data, [&](BinDataType* p){
    if (p) {
      if (p->data) {
        delete [](p->data);
        p->data = nullptr;
      }
      delete p;
      p = nullptr;
    }
  });
  if (dnn_output->sp_left_nv12) {
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"),
                  "valid sp_left_nv12");
  } else {
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"),
                  "invalid sp_left_nv12");
    rclcpp::shutdown();
    return;
  }
  std::ifstream ifs("frame_1280_720_left.nv12");
  ifs.read(bin_data->data, bin_data->len);
  // memcpy(bin_data->data, img_l, bin_data->len);



  if (Run(input_tensors, output_descs, dnn_output, true, -1, -1) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Run infer fail!");
    return;
  }

  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Run infer done");
}

void StereonetNode::GetNV12Tensor(std::string &image_file,
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

  int c_stride = properties.validShape.dimensionSize[1];
  auto w_stride = ALIGN_16(scaled_img_width);
  // tensor type is HB_DNN_TENSOR_TYPE_S8(val is 8)
  // TODO hbSysFreeMem
  hbSysAllocCachedMem(&dnn_tensor->sysMem[0], y_size * c_stride);
  
  // hbSysAllocCachedMem(&dnn_tensor->sysMem[1], y_size * c_stride / 2);
  //内存初始化
  memset(dnn_tensor->sysMem[0].virAddr, 0, y_size * c_stride);
  // memset(dnn_tensor->sysMem[1].virAddr, 0, y_size / 2);

  auto stride = properties.alignedShape.dimensionSize[3];
  auto *y = reinterpret_cast<uint8_t *>(dnn_tensor->sysMem[0].virAddr);
  memcpy(y, img_nv12.data, height * width * 3 / 2);
  
  auto *y_2 = y + height * width * 3;
  memcpy(y_2, img_nv12.data, height * width * 3 / 2);

  hbSysFlushMem(&dnn_tensor->sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  // hbSysFlushMem(&dnn_tensor->sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
}


void StereonetNode::FeedImg(
    const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr img_msg) {
  if (!rclcpp::ok() || !img_msg) {
    return;
  }

  RCLCPP_INFO_STREAM(rclcpp::get_logger("stereonet_node"),
    "recved img msg index: " << img_msg->index
    << " h: " << img_msg->height
    << ", w: " << img_msg->width
    << ", encoding: " << img_msg->encoding.data()
    << ", ts: " << img_msg->time_stamp.sec << "." << img_msg->time_stamp.nanosec);

  // 1 对订阅到的图片消息进行验证，本示例只支持处理NV12格式图片数据
  // 如果是其他格式图片，订阅hobot_codec解码/转码后的图片消息
  if ("nv12" !=
      std::string(reinterpret_cast<const char*>(img_msg->encoding.data()))) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"),
                 "Only support nv12 img encoding! Using hobot codec to process "
                 "%d encoding img.",
                 img_msg->encoding.data());
    return;
  }

  // 双目图像分辨率宽是模型输入宽度的两倍，高度等于模型输入高度
  if (img_msg->height != static_cast<uint32_t>(model_input_height_) ||
      img_msg->width != static_cast<uint32_t>(model_input_width_) * 2) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet_node"),
      "recved img msg h: " << img_msg->height
      << ", w: " << img_msg->width
      << " is unmatch with model_input_width: " << model_input_width_
      << ", model_input_height: " << model_input_height_);
    return;
  }

  // 2 创建算法输出数据，填充消息头信息，用于推理完成后AI结果的发布
  auto dnn_output = std::make_shared<StereonetNodeOutput>();
  dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
  dnn_output->msg_header->set__frame_id(std::to_string(img_msg->index));
  dnn_output->msg_header->set__stamp(img_msg->time_stamp);

  // 3 算法前处理，即创建算法输入数据
  std::vector<std::shared_ptr<DNNTensor>> input_tensors;
  int offset = img_msg->height * img_msg->width * 1.5 * 0.5;
  auto tp_start = std::chrono::system_clock::now();
  int w = img_msg->width / 2;
  int h = img_msg->height;
  // left img
  unsigned char *left_buf = reinterpret_cast<unsigned char *>(calloc(1, w*h*1.5));
  {
    const unsigned char *tmp_src = reinterpret_cast<const unsigned char*>(img_msg->data.data());
    unsigned char *tmp_buf = left_buf;
    for (int idx_h = 0; idx_h < img_msg->height; idx_h++) {
      memcpy(tmp_buf, tmp_src, w);
      tmp_buf+= w;
      tmp_src+= img_msg->width;
    }
    for (int idx_h = 0; idx_h < img_msg->height / 2; idx_h++) {
      memcpy(tmp_buf, tmp_src, w);
      tmp_buf+= w;
      tmp_src+= img_msg->width;
    }
  }

  // right img
  unsigned char *right_buf = reinterpret_cast<unsigned char *>(calloc(1, w*h*1.5));
  {
    const unsigned char *tmp_src = reinterpret_cast<const unsigned char*>(img_msg->data.data());
    unsigned char *tmp_buf = right_buf;
    for (int idx_h = 0; idx_h < img_msg->height; idx_h++) {
      tmp_src += w;
      memcpy(tmp_buf, tmp_src, w);
      tmp_buf += w;
      tmp_src += w;
    }
    for (int idx_h = 0; idx_h < img_msg->height / 2; idx_h++) {
      tmp_src += w;
      memcpy(tmp_buf, tmp_src, w);
      tmp_buf += w;
      tmp_src += w;
    }
  }

  if (sp_preprocess_->CvtNV12Data2Tensors(input_tensors, model_, left_buf, right_buf) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Preprocess fail");
    rclcpp::shutdown();
    return;
  }

  free(left_buf);
  free(right_buf);
  
  if (enable_pub_output_) {
    // 将nv12格式的左图转成jpg格式
    // left img
    int w = img_msg->width / 2;
    int h = img_msg->height;
    unsigned char *left_nv12_data = reinterpret_cast<unsigned char *>(calloc(1, w*h*1.5));
    const unsigned char *tmp_src = reinterpret_cast<const unsigned char*>(img_msg->data.data());
    unsigned char *tmp_buf = left_nv12_data;
    for (int idx_h = 0; idx_h < img_msg->height; idx_h++) {
      memcpy(tmp_buf, tmp_src, w);
      tmp_buf+= w;
      tmp_src+= img_msg->width;
    }
    for (int idx_h = 0; idx_h < img_msg->height / 2; idx_h++) {
      memcpy(tmp_buf, tmp_src, w);
      tmp_buf+= w;
      tmp_src+= img_msg->width;
    }
    // std::ofstream ofs(std::string("recved")
    //     + "_" +  std::to_string(w)
    //     + "_" + std::to_string(h)
    //     + "_left.nv12");
    // ofs.write((const char*)left_nv12_data, w*h*1.5);
    cv::Mat nv12(h * 3 / 2, w, CV_8UC1, (char*)left_nv12_data);
    cv::Mat bgr_mat;
    cv::cvtColor(nv12, bgr_mat, CV_YUV2BGR_NV12);  //  nv12 to bgr
    // cv::imwrite(std::string("recved")
    //     + "_" +  std::to_string(w)
    //     + "_" + std::to_string(h)
    //     + "_left.jpg", bgr_mat);
    free(left_nv12_data);

    
    BinDataType* bin_data = new BinDataType();
    // 使用opencv的imencode接口将mat转成vector，获取图片size
    std::vector<int> param;
    imencode(".jpg", bgr_mat, bin_data->jpeg, param);

    dnn_output->sp_left_nv12 = std::shared_ptr<BinDataType>(bin_data, [&](BinDataType* p){
      if (p) {
        if (p->data) {
          delete [](p->data);
          p->data = nullptr;
        }
        delete p;
        p = nullptr;
      }
    });
    if (!dnn_output->sp_left_nv12) {
      RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"),
                    "invalid sp_left_nv12");
      rclcpp::shutdown();
      return;
    }
  }

  auto tp_now = std::chrono::system_clock::now();
  auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                      tp_now - tp_start)
                      .count();
  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Preprocess done, time cost %d ms", interval);
  dnn_output->preprocess_time_ms = interval;

  std::vector<std::shared_ptr<hobot::dnn_node::OutputDescription>> output_descs{};
  if (Run(input_tensors, output_descs, dnn_output, false, -1, -1) < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Run infer fail!");
    return;
  }

  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Run infer done");
}

void StereonetNode::RunImglistFeedInfer(std::string left_img_list, std::string right_img_list) {
  if (!rclcpp::ok()) {
    return;
  }

  RCLCPP_INFO_STREAM(rclcpp::get_logger("stereonet_node"),
    "Feedback with left_img_list: " << left_img_list
    << " right_img_list: " << right_img_list);

  std::vector<std::string> left_imgs;
  std::vector<std::string> right_imgs;

  {
    std::ifstream ifs(left_img_list);
    if (!ifs.good()) {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet_node"), "Open file failed: " << left_img_list);
      rclcpp::shutdown();
      return;
    }

    std::string image_path;
    while (std::getline(ifs, image_path)) {
      std::string img_name = image_path;
      if (access(img_name.c_str(), F_OK) != 0) {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet_node"), "File is not exist! img_name: " << img_name);
        rclcpp::shutdown();
        return;
      }
      left_imgs.push_back(img_name);
      if (!rclcpp::ok()) {
        return;
      }
    }
    ifs.close();
  }
  
  {
    std::ifstream ifs(right_img_list);
    if (!ifs.good()) {
      RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet_node"), "Open file failed: " << right_img_list);
      rclcpp::shutdown();
      return;
    }

    std::string image_path;
    while (std::getline(ifs, image_path)) {
      std::string img_name = image_path;
      if (access(img_name.c_str(), F_OK) != 0) {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet_node"), "File is not exist! img_name: " << img_name);
        rclcpp::shutdown();
        return;
      }
      right_imgs.push_back(img_name);
      if (!rclcpp::ok()) {
        return;
      }
    }
    ifs.close();
  }
  

  if (left_imgs.size() != right_imgs.size()) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet_node"),
      "Imgs size error! left_imgs.size: " << left_imgs.size()
      << ", right_imgs.size: " << right_imgs.size());
    rclcpp::shutdown();
    return;
  }

  // 等待可视化端启动成功
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  size_t img_list_len = left_imgs.size();
  for (size_t idx = 0; idx < img_list_len; idx++) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("stereonet_node"),
      "Feed " << idx << "/" << img_list_len);
      
    if (!rclcpp::ok()) {
      return;
    }
    cv::Mat left_nv12_mat;
    cv::Mat left_bgr_mat = cv::imread(left_imgs.at(idx), cv::IMREAD_COLOR);
    {
      auto ret = Tools::BGRToNv12(left_bgr_mat, left_nv12_mat);
      if (ret) {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet_node"),
          "BGRToNv12 Fail");
        rclcpp::shutdown();
        return;
      }
    }
    cv::Mat right_nv12_mat;
    {
      cv::Mat bgr_mat = cv::imread(right_imgs.at(idx), cv::IMREAD_COLOR);
      auto ret = Tools::BGRToNv12(bgr_mat, right_nv12_mat);
      if (ret) {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet_node"),
          "BGRToNv12 Fail");
        rclcpp::shutdown();
        return;
      }
    }

    auto dnn_output = std::make_shared<StereonetNodeOutput>();
    dnn_output->msg_header = std::make_shared<std_msgs::msg::Header>();
    dnn_output->msg_header->frame_id = std::to_string(idx);

    std::vector<std::shared_ptr<DNNTensor>> input_tensors;
    int w = model_input_width_;
    int h = model_input_height_;
    
    // left img
    unsigned char *left_buf = reinterpret_cast<unsigned char *>(left_nv12_mat.data);
    // right img
    unsigned char *right_buf = reinterpret_cast<unsigned char *>(right_nv12_mat.data);

    if (sp_preprocess_->CvtNV12Data2Tensors(input_tensors, model_, left_buf, right_buf) < 0) {
      RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Preprocess fail");
      rclcpp::shutdown();
      return;
    }

    if (enable_pub_output_) {
      // 将nv12格式的左图转成jpg格式
      // left img
      BinDataType* bin_data = new BinDataType();
      // 使用opencv的imencode接口将mat转成vector，获取图片size
      std::vector<int> param;
      imencode(".jpg", left_bgr_mat, bin_data->jpeg, param);

      dnn_output->sp_left_nv12 = std::shared_ptr<BinDataType>(bin_data, [&](BinDataType* p){
        if (p) {
          if (p->data) {
            delete [](p->data);
            p->data = nullptr;
          }
          delete p;
          p = nullptr;
        }
      });
      if (!dnn_output->sp_left_nv12) {
        RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"),
                      "invalid sp_left_nv12");
        rclcpp::shutdown();
        return;
      }
    }

    std::vector<std::shared_ptr<hobot::dnn_node::OutputDescription>> output_descs{};
    if (Run(input_tensors, output_descs, dnn_output, true, -1, -1) < 0) {
      RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"), "Run infer fail!");
      return;
    }

    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"), "Run infer done");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }
}


// 推理结果回调，解析算法输出，通过ROS Msg发布消息
int StereonetNode::PostProcess(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput>& node_output) {
  if (!rclcpp::ok()) {
    return 0;
  }

  RCLCPP_INFO(rclcpp::get_logger("stereonet_node"),
                "Parse node_output start");
                
  // 后处理开始时间
  auto tp_start = std::chrono::system_clock::now();

  // 使用自定义的Parse解析方法，解析算法输出的DNNTensor类型数据
  // 创建解析输出数据，输出StereonetNodeOutput是自定义的算法输出数据类型，results的维度等于检测出来的目标数
  std::vector<std::shared_ptr<StereonetResult>>
      results;

  // 开始解析
  // 不做后处理，直接将模型输出发布出去
  // if (hobot::stereonet::Parse(node_output, results) < 0) {
  //   RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"),
  //                "Parse node_output fail!");
  //   return -1;
  // }
  // {
  //   auto tp_now = std::chrono::system_clock::now();
  //   auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
  //                       tp_now - tp_start)
  //                       .count();
  //   RCLCPP_INFO(rclcpp::get_logger("stereonet_node"),
  //                 "Parse node_output complete, time cost ms: %d",
  //               interval);
  // }

  tp_start = std::chrono::system_clock::now();

  auto stereonet_node_output =
      std::dynamic_pointer_cast<StereonetNodeOutput>(node_output);
  if (!stereonet_node_output) {
    RCLCPP_ERROR(rclcpp::get_logger("stereonet_node"),
                 "Cast dnn node output fail!");
    return -1;
  }

  // 将双目图像的左图转成jpeg，和模型输出一起发布出去
  if (enable_pub_output_ && stereonet_node_output->sp_left_nv12) {
    auto msg = sensor_msgs::msg::Image();
    msg.height = stereonet_node_output->sp_left_nv12->h;
    msg.width = stereonet_node_output->sp_left_nv12->w;
    msg.encoding = "jpeg";
    
    msg.header = *stereonet_node_output->msg_header;

    char * infer_out_data = reinterpret_cast<char *>(stereonet_node_output->output_tensors[0]->sysMem[0].virAddr);
    int infer_out_data_len = stereonet_node_output->output_tensors[0]->sysMem[0].memSize;

    auto jpeg_img_data = stereonet_node_output->sp_left_nv12->jpeg.data();
    auto jpeg_img_data_len = stereonet_node_output->sp_left_nv12->jpeg.size();
    
    auto data_len = infer_out_data_len + jpeg_img_data_len;
    msg.step = data_len;
    msg.data.resize(data_len);

    char* msg_data_buf = (char*)&msg.data[0];
    // 先拷贝原始infer数据
    memcpy(msg_data_buf, infer_out_data, infer_out_data_len);
    msg_data_buf += infer_out_data_len;

    // 再拷贝jpeg左图数据
    memcpy(msg_data_buf, jpeg_img_data, jpeg_img_data_len);

    // msg.time_stamp.sec = frame.timestamp / 1e9;
    // msg.time_stamp.nanosec = frame.timestamp - msg.time_stamp.sec * 1e9;
    // msg.index = frame.frame_id;
  
    auto tp_now = std::chrono::system_clock::now();
    auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                        tp_now - tp_start)
                        .count();
                        
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"),
                "publish output with msg index: %s, topic: %s, time cost ms: %d",
                stereonet_node_output->msg_header->frame_id.data(), ros_img_topic_name_.data(), interval);

    ros_img_publisher_->publish(std::move(msg));
  } else {
    RCLCPP_INFO(rclcpp::get_logger("stereonet_node"),
                  "publish is unable");
  }
  

  if (node_output->rt_stat) {
    // 如果算法推理统计有更新，输出算法输入和输出的帧率统计、推理耗时
    if (node_output->rt_stat->fps_updated) {
      auto tp_now = std::chrono::system_clock::now();
      auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
                          tp_now - tp_start)
                          .count();
      RCLCPP_WARN(rclcpp::get_logger("stereonet_node"),
                  "input fps: %.2f, out fps: %.2f, preprocess time ms: %d, infer time ms: %d, msg preparation for pub time cost ms: %d",
                  node_output->rt_stat->input_fps,
                  node_output->rt_stat->output_fps,
                  stereonet_node_output->preprocess_time_ms,
                  node_output->rt_stat->infer_time_ms,
                  interval);
    }
  }

  return 0;
}

}  // namespace stereonet
}  // namespace hobot
