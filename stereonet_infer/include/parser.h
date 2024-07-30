// Copyright (c) 2024，D-Robotics.
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

#include "dnn_node/dnn_node_data.h"

namespace hobot {
namespace stereonet {
// 定义算法输出数据类型
struct StereonetResult {
  // 反量化后的结果
  // 1280x720
  // 深度信息，单位米
  std::vector<float> results;
  
  // // 模型输出转成jpeg图
  // std::vector<uint8_t> jpeg;
};

// 自定义的算法输出解析方法
// - 参数
//   - [in] node_output dnn node输出，包含算法推理输出
//          解析时，如果不需要使用前处理参数，可以直接使用DnnNodeOutput中的
//          std::vector<std::shared_ptr<DNNTensor>>
//          output_tensors成员作为Parse的入口参数
//   - [in/out] results 解析后的结构化数据，StereonetResult为自定义的算法输出数据类型
// - 返回值
//   - 0 成功
//   - -1 失败
int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::vector<std::shared_ptr<StereonetResult>> &results);

}  // namespace stereonet
}  // namespace hobot