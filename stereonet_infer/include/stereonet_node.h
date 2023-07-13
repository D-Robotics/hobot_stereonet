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

#ifndef STEREONET_INCLUDE_STEREONET_STEREONET_NODE_H_
#define STEREONET_INCLUDE_STEREONET_STEREONET_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "ai_msgs/msg/perception_targets.hpp"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include "parser.h"
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "preprocess.h"

#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using hobot::dnn_node::DNNTensor;
using hobot::dnn_node::OutputDescription;
using hobot::dnn_node::DnnNodeOutput;

namespace hobot {
namespace stereonet {

struct BinDataType {
  char* data = nullptr;
  int len = 0;
  int w = 1280;
  int h = 720;

  std::vector<uint8_t> jpeg;
};

struct StereonetNodeOutput : public hobot::dnn_node::DnnNodeOutput {
  // 缩放比例系数，原图和模型输入分辨率的比例。
  float ratio = 1.0;
  std::shared_ptr<DNNTensor> pyramid = nullptr;
  std::vector<std::string> image_files;

  std::shared_ptr<cv::Mat> left_nv12 = nullptr;
  std::shared_ptr<BinDataType> sp_left_nv12 = nullptr;

  int preprocess_time_ms = 0;
};

class StereonetNode : public hobot::dnn_node::DnnNode {
 public:
  StereonetNode(const std::string& node_name = "stereonet_node",
                const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 protected:
  // 实现基类的纯虚接口，用于配置Node参数
  int SetNodePara() override;
  // 实现基类的虚接口，将解析后结构化的算法输出数据封装成ROS Msg后发布
  int PostProcess(const std::shared_ptr<hobot::dnn_node::DnnNodeOutput>&
                      node_output) override;

 private:
  // 使用hobotcv resize nv12格式图片，固定图片宽高比
  int ResizeNV12Img(const char* in_img_data,
                    const int& in_img_height,
                    const int& in_img_width,
                    const int& scaled_img_height,
                    const int& scaled_img_width,
                    cv::Mat& out_img,
                    float& ratio);

  void GetNV12Tensor(std::string &image_file,
                    std::shared_ptr<DNNTensor> &dnn_tensor,
                    int scaled_img_height, int scaled_img_width);

  Model *model_ = nullptr;
  // 算法输入图片数据的宽和高
  int model_input_width_ = -1;
  int model_input_height_ = -1;
  std::vector<hbDNNTensorProperties> input_model_info_;
  std::vector<hbDNNTensorProperties> output_model_info_;

  // 图片消息订阅者
  rclcpp::SubscriptionHbmem<hbm_img_msgs::msg::HbmMsg1080P>::ConstSharedPtr 
      subscription_hbmem_img_ = nullptr;
  std::string sub_hbmem_topic_name_ = "hbmem_stereo_img";

  // 算法推理结果消息发布者
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr msg_publisher_ =
      nullptr;

  //用于ros方式发布图片
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ros_img_publisher_ =
      nullptr;
  std::string ros_img_topic_name_ = "/stereonet_node_output";
  bool enable_pub_output_ = true;

  rclcpp::TimerBase::SharedPtr timer_ = nullptr;

  // 图片消息订阅回调
  void FeedImg(const hbm_img_msgs::msg::HbmMsg1080P::ConstSharedPtr msg);

  void RunImgFeedInfer();

  void RunBinFeedInfer();
  
  void RunImglistFeedInfer(std::string left_img_list, std::string right_img_list);

  std::string config_file_ = "config/hobot_stereonet_config.json";
  std::string model_file_ = "config/hobot_stereonet.hbm";

  std::shared_ptr<PreProcess> sp_preprocess_ = nullptr;


};  // class StereonetNode

}  // namespace stereonet
}  // namespace hobot

#endif  // STEREONET_INCLUDE_STEREONET_STEREONET_NODE_H_
