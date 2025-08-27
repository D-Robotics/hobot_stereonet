// Copyright (c) 2025，D-Robotics.
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

#include "stereonet_process.h"

namespace stereonet {
StereonetProcess::StereonetProcess(const rclcpp::Logger &logger) : logger_(logger) {
}

int StereonetProcess::init(const std::string &model_path, const int &max_memory_count) {
  int ret_code = 0;
  RCLCPP_INFO(logger_, "=> ==================== init stereonet model start ====================");
  // load model
  model_path_ = model_path;
  const char *model_path_cstr = model_path_.c_str();
  ret_code = hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_path_cstr, 1);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNInitializeFromFiles failed");

  // get model name
  ret_code = hbDNNGetModelNameList(&model_name_list_, &model_count_, packed_dnn_handle_);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetModelNameList failed");

  // get model handle
  ret_code = hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_list_[0]);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetModelHandle failed");

  // get input count and output count
  ret_code = hbDNNGetInputCount(&input_count_, dnn_handle_);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetInputCount failed");
  ret_code = hbDNNGetOutputCount(&output_count_, dnn_handle_);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetOutputCount failed");
  RCLCPP_INFO_STREAM(logger_, "=> model name: " << model_name_list_[0]);
  RCLCPP_INFO_STREAM(logger_, "=> input_count: " << input_count_);
  RCLCPP_INFO_STREAM(logger_, "=> output_count: " << output_count_);

  max_memory_count_ = max_memory_count;
  for (int i = 0; i < max_memory_count_; i++) {
    idle_tensor_.emplace_back(true);
  }

  batch_input_tensors_.resize(max_memory_count_);
  for (int i = 0; i < max_memory_count_; i++) {
    ret_code = prepare_input_tensor(batch_input_tensors_[i]);
  }

  batch_output_tensors_.resize(max_memory_count_);
  for (int i = 0; i < max_memory_count_; ++i) {
    ret_code = prepare_output_tensor(batch_output_tensors_[i]);
  }

  return ret_code;
}

int StereonetProcess::forward(std::vector<uint8_t> &left_img_data, std::vector<uint8_t> &right_img_data,
                              const int &img_w, const int &img_h, const double &uncertainty_th, cv::Mat &disp,
                              cv::Mat &uncert) {
  RCLCPP_INFO_STREAM(logger_, "=> ==================== infer by model =======================");
  int ret_code = 0;
  if (img_w != model_input_w_ || img_h != model_input_h_) {
    RCLCPP_ERROR_STREAM(logger_, "=> input image size does not match model input size, expected: "
                                     << model_input_w_ << "x" << model_input_h_ << ", got: " << img_w << "x" << img_h);
    return -1;
  }

  int idle_tensor_id = get_idle_tensor();
  {
    ScopeProcessTime t(logger_, "fill_img_to_input_tensor");
    if (idle_tensor_id == -1) {
      RCLCPP_ERROR_STREAM(logger_, "=> no idle tensor");
      return -1;
    }
    ret_code =
        fill_img_to_input_tensor(batch_input_tensors_[idle_tensor_id], left_img_data.data(), right_img_data.data());
  }

  {
    ScopeProcessTime t(logger_, "infer");
    hbDNNTensor *output = batch_output_tensors_[idle_tensor_id].data();
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    hbDNNTaskHandle_t task_handle = nullptr;
    ret_code =
        hbDNNInfer(&task_handle, &output, batch_input_tensors_[idle_tensor_id].data(), dnn_handle_, &infer_ctrl_param);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNInfer failed");
    // wait task done
    ret_code = hbDNNWaitTaskDone(task_handle, 0);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNWaitTaskDone failed");
    ret_code = hbDNNReleaseTask(task_handle);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNReleaseTask failed");
    // make sure CPU read data from DDR before using output tensor data
    for (size_t i = 0; i < batch_output_tensors_[idle_tensor_id].size(); i++) {
#ifdef PLATFORM_X5
      ret_code = hbSysFlushMem(&(batch_output_tensors_[idle_tensor_id][i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
#endif
#ifdef PLATFORM_S100
      ret_code = hbSysFlushMem(&(batch_output_tensors_[idle_tensor_id][i].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
#endif
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    }
  }

  {
    ScopeProcessTime t(logger_, "postprocess");
    postprocess(batch_output_tensors_[idle_tensor_id], uncertainty_th, disp, uncert);
  }

  set_tensor_idle(idle_tensor_id);

  return ret_code;
}

int StereonetProcess::postprocess(const std::vector<hbDNNTensor> &output_tensors, const double &uncertainty_th,
                                  cv::Mat &disp, cv::Mat &uncert) {
  int ret_code = 0;
  ret_code = postprocess_convex_upsampling(output_tensors, disp);
  return ret_code;
}

int StereonetProcess::postprocess_convex_upsampling(const std::vector<hbDNNTensor> &tensors, cv::Mat &out_mat) {
  // get shape info
  const int32_t *disp_shape = tensors[0].properties.validShape.dimensionSize;
  int c_dim = disp_shape[1];
  int h_dim = disp_shape[2];
  int w_dim = disp_shape[3];

  // calc disp
  Eigen::MatrixXf result = Eigen::MatrixXf::Zero(h_dim, w_dim);
  if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
      tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    // get tensor info
#ifdef PLATFORM_X5
    auto disp = reinterpret_cast<float *>(tensors[0].sysMem[0].virAddr);
    auto spx = reinterpret_cast<float *>(tensors[1].sysMem[0].virAddr);
#endif
#ifdef PLATFORM_S100
    auto disp = reinterpret_cast<float *>(tensors[0].sysMem.virAddr);
    auto spx = reinterpret_cast<float *>(tensors[1].sysMem.virAddr);
#endif

    // multiply element-wise and then add in the c channel
    for (int i = 0; i < c_dim; ++i) {
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim,
                                                                                   w_dim);
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim,
                                                                                  w_dim);
      result.noalias() += matrix_disp.cwiseProduct(matrix_spx);
    }
  } else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
    // get tensor info
#ifdef PLATFORM_X5
    auto disp = reinterpret_cast<float *>(tensors[0].sysMem[0].virAddr);
    auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem[0].virAddr);
#endif
#ifdef PLATFORM_S100
    auto disp = reinterpret_cast<float *>(tensors[0].sysMem.virAddr);
    auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem.virAddr);
#endif

    // multiply element-wise and then add in the c channel
    for (int i = 0; i < c_dim; ++i) {
      Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim,
                                                                                   w_dim);
      Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim,
                                                                                    w_dim);
      result.noalias() += matrix_disp.cwiseProduct(matrix_spx.cast<float>());
    }
  } else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S32 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
    // get tensor info
#ifdef PLATFORM_X5
    auto disp = reinterpret_cast<int32_t *>(tensors[0].sysMem[0].virAddr);
    auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem[0].virAddr);
#endif
#ifdef PLATFORM_S100
    auto disp = reinterpret_cast<int32_t *>(tensors[0].sysMem.virAddr);
    auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem.virAddr);
#endif

    // multiply element-wise and then add in the c channel
    for (int i = 0; i < c_dim; ++i) {
      Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim,
                                                                                     w_dim);
      Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim,
                                                                                    w_dim);
      result.noalias() += matrix_disp.cast<float>().cwiseProduct(matrix_spx.cast<float>());
    }
  } else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S16 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
    // get tensor info
#ifdef PLATFORM_X5
    auto disp = reinterpret_cast<int16_t *>(tensors[0].sysMem[0].virAddr);
    auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem[0].virAddr);
#endif
#ifdef PLATFORM_S100
    auto disp = reinterpret_cast<int16_t *>(tensors[0].sysMem.virAddr);
    auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem.virAddr);
#endif

    // multiply element-wise and then add in the c channel
    for (int i = 0; i < c_dim; ++i) {
      Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim,
                                                                                     w_dim);
      Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim,
                                                                                    w_dim);
      result.noalias() += matrix_disp.cast<float>().cwiseProduct(matrix_spx.cast<float>());
    }
  } else {
    RCLCPP_ERROR_STREAM(logger_,
                        "=> output tensor type unsupported! tensor[0]: "
                            << magic_enum::enum_name(static_cast<hbDNNDataType>(tensors[0].properties.tensorType))
                            << ", tensor[1]: "
                            << magic_enum::enum_name(static_cast<hbDNNDataType>(tensors[1].properties.tensorType)));
    return -1;
  }
  // get scale info
  float scale_constant = 1.0;
  float scale_factor;
  float *disp_scale = &scale_constant;
  float *spx_scale = &scale_constant;
  if (tensors[0].properties.quantiType == SCALE) {
    disp_scale = tensors[0].properties.scale.scaleData;
  }
  if (tensors[1].properties.quantiType == SCALE) {
    spx_scale = tensors[1].properties.scale.scaleData;
  }
  scale_factor = (*disp_scale * *spx_scale);
  if (std::abs(scale_factor - 1.f) > 1e-2) {
    result *= (*disp_scale * *spx_scale);
  }
  out_mat = cv::Mat::zeros(h_dim, w_dim, CV_32FC1);
  memcpy(out_mat.data, result.data(), h_dim * w_dim * sizeof(float));
  return 0;
}

int StereonetProcess::prepare_input_tensor(std::vector<hbDNNTensor> &input_tensors) {
  int ret_code = 0;
  RCLCPP_INFO(logger_, "=> ----- prepare_input_tensor -----");

  // get model input size from input tensor[0]
  hbDNNTensorProperties properties;
  ret_code = hbDNNGetInputTensorProperties(&properties, dnn_handle_, 0);
#ifdef PLATFORM_S100
  properties.quantizeAxis = 3;
#endif
  hbGetInputTensorHW(properties, model_input_h_, model_input_w_);
  RCLCPP_INFO_STREAM(logger_, "=> model_input_h: " << model_input_h_ << ", model_input_w: " << model_input_w_);

  // allocate memory for input tensor
  input_tensors.resize(input_count_);
  for (int i = 0; i < input_count_; i++) {
    auto &tensor = input_tensors[i];
    // get input tensor properties
    ret_code = hbDNNGetInputTensorProperties(&properties, dnn_handle_, i);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetInputTensorProperties failed");
    RCLCPP_INFO_STREAM(logger_, "=> input tensor type is "
                                    << magic_enum::enum_name(static_cast<hbDNNDataType>(properties.tensorType)));
    input_tensor_type_ = properties.tensorType;

#ifdef PLATFORM_X5
    if ((properties.tensorType != HB_DNN_IMG_TYPE_NV12) && (properties.tensorType != HB_DNN_IMG_TYPE_NV12_SEPARATE)) {
      RCLCPP_ERROR(logger_, "=> input tensor type is not in [HB_DNN_IMG_TYPE_NV12, HB_DNN_IMG_TYPE_NV12_SEPARATE]");
      return -1;
    }
#endif

#ifdef PLATFORM_S100
    if ((properties.tensorType != HB_DNN_TENSOR_TYPE_U8)) {
      RCLCPP_ERROR(logger_, "=> input tensor type is not in [HB_DNN_TENSOR_TYPE_U8]");
      return -1;
    }
#endif

#ifdef PLATFORM_S100
    // properties.quantizeAxis = 3;
    properties.alignedByteSize = properties.validShape.dimensionSize[0] * properties.validShape.dimensionSize[1] *
                                 properties.validShape.dimensionSize[2] * properties.validShape.dimensionSize[3];
    auto dim_len = properties.validShape.numDimensions;
    for (int32_t dim_i = dim_len - 1; dim_i >= 0; --dim_i) {
      if (properties.stride[dim_i] == -1) {
        auto cur_stride = properties.stride[dim_i + 1] * properties.validShape.dimensionSize[dim_i + 1];
        properties.stride[dim_i] = ALIGN_32(cur_stride);
      }
    }
#endif

    tensor.properties = properties;
    tensor.properties.tensorType = properties.tensorType;

#ifdef PLATFORM_X5
    tensor.properties.validShape.numDimensions = 4;
    tensor.properties.validShape.dimensionSize[0] = 1;
    tensor.properties.validShape.dimensionSize[1] = 3;
    tensor.properties.validShape.dimensionSize[2] = model_input_h_;
    tensor.properties.validShape.dimensionSize[3] = model_input_w_;
    tensor.properties.alignedShape = tensor.properties.validShape;

    if (properties.tensorType == HB_DNN_IMG_TYPE_NV12) {
      ret_code = hbSysAllocCachedMem(&tensor.sysMem[0], (3 * model_input_h_ * model_input_w_) / 2);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      tensor.sysMem[0].memSize = (3 * model_input_h_ * model_input_w_) / 2;
      RCLCPP_INFO_STREAM(logger_, "=> input[" << i << "].memsize: " << tensor.sysMem[0].memSize);
    } else if (properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
      ret_code = hbSysAllocCachedMem(&tensor.sysMem[0], model_input_h_ * model_input_w_);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      tensor.sysMem[0].memSize = model_input_h_ * model_input_w_;

      ret_code = hbSysAllocCachedMem(&tensor.sysMem[1], model_input_h_ * model_input_w_ / 2);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      tensor.sysMem[1].memSize = model_input_h_ * model_input_w_ / 2;
      RCLCPP_INFO_STREAM(logger_, "=> input[" << i << "].memsize[0]: " << tensor.sysMem[0].memSize);
      RCLCPP_INFO_STREAM(logger_, "=> input[" << i << "].memsize[1]: " << tensor.sysMem[1].memSize);
    } else {
      return -1;
    }
#endif

#ifdef PLATFORM_S100
    if (properties.tensorType == HB_DNN_TENSOR_TYPE_U8) {
      ret_code = hbSysAllocCachedMem(&tensor.sysMem, properties.alignedByteSize);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      RCLCPP_INFO_STREAM(logger_, "=> input tensor size: " << tensor.sysMem.memSize);
    } else {
      return -1;
    }
#endif
  }
  return ret_code;
}

int StereonetProcess::prepare_output_tensor(std::vector<hbDNNTensor> &output_tensors) {
  int ret_code = 0;
  RCLCPP_INFO(logger_, "=> ----- prepare_output_tensor -----");
  output_tensors.resize(output_count_);
  for (int i = 0; i < output_count_; ++i) {
    ret_code = hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle_, i);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetOutputTensorProperties failed");
    RCLCPP_INFO_STREAM(logger_, "=> output tensor type is " << magic_enum::enum_name(
                                    static_cast<hbDNNDataType>(output_tensors[i].properties.tensorType)));
#ifdef PLATFORM_X5
    ret_code = hbSysAllocCachedMem(&output_tensors[i].sysMem[0], output_tensors[i].properties.alignedByteSize);
#endif
#ifdef PLATFORM_S100
    ret_code = hbSysAllocCachedMem(&output_tensors[i].sysMem, output_tensors[i].properties.alignedByteSize);
#endif
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
    RCLCPP_INFO_STREAM(logger_, "=> output[" << i << "].memsize: " << output_tensors[i].properties.alignedByteSize);
  }
  return ret_code;
}

int StereonetProcess::get_idle_tensor() {
  for (int i = 0; i < max_memory_count_; ++i) {
    if (idle_tensor_[i]) {
      idle_tensor_[i] = false;
      return i;
    }
  }
  return -1;
}

int StereonetProcess::set_tensor_idle(int tensor_id) {
  if (tensor_id >= 0 || tensor_id < max_memory_count_) {
    idle_tensor_[tensor_id] = true;
    return 0;
  }
  return -1;
}

int StereonetProcess::fill_img_to_input_tensor(std::vector<hbDNNTensor> &input_tensors, uint8_t *left_img_data,
                                               uint8_t *right_img_data) {
  int ret_code = 0;
#ifdef PLATFORM_X5
  hbDNNTensor &left_input_tensor = input_tensors[0];
  hbDNNTensor &right_input_tensor = input_tensors[1];

  if (input_tensor_type_ == HB_DNN_IMG_TYPE_NV12) {
    // RCLCPP_INFO(logger_, "=> fill image data into memory HB_DNN_IMG_TYPE_NV12");
    // fill image data into memory
    ret_code = hbSysWriteMem(&left_input_tensor.sysMem[0], (char *)left_img_data, left_input_tensor.sysMem[0].memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code =
        hbSysWriteMem(&right_input_tensor.sysMem[0], (char *)right_img_data, right_input_tensor.sysMem[0].memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");

    // make sure memory data is flushed to DDR before inference
    ret_code = hbSysFlushMem(&left_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&right_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
  } else if (input_tensor_type_ == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    // RCLCPP_INFO(logger_, "=>fill image data into memory HB_DNN_IMG_TYPE_NV12_SEPARATE");
    // fill image data into memory
    ret_code = hbSysWriteMem(&left_input_tensor.sysMem[0], (char *)left_img_data, left_input_tensor.sysMem[0].memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code = hbSysWriteMem(&left_input_tensor.sysMem[1], (char *)left_img_data + left_input_tensor.sysMem[0].memSize,
                             left_input_tensor.sysMem[1].memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code =
        hbSysWriteMem(&right_input_tensor.sysMem[0], (char *)right_img_data, right_input_tensor.sysMem[0].memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code =
        hbSysWriteMem(&right_input_tensor.sysMem[1], (char *)right_img_data + right_input_tensor.sysMem[0].memSize,
                      right_input_tensor.sysMem[1].memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");

    // make sure memory data is flushed to DDR before inference
    ret_code = hbSysFlushMem(&left_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&left_input_tensor.sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&right_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&right_input_tensor.sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
  } else {
    RCLCPP_ERROR(logger_, "=> input_tensor_type is not in [HB_DNN_IMG_TYPE_NV12, HB_DNN_IMG_TYPE_NV12_SEPARATE]");
    return -1;
  }
#endif

#ifdef PLATFORM_S100
  hbDNNTensor &left_input_y_tensor = input_tensors[0];
  hbDNNTensor &left_input_uv_tensor = input_tensors[1];
  hbDNNTensor &right_input_y_tensor = input_tensors[2];
  hbDNNTensor &right_input_uv_tensor = input_tensors[3];

  if (input_tensor_type_ == HB_DNN_TENSOR_TYPE_U8) {
    // RCLCPP_INFO(logger_, "=>fill image data into memory HB_DNN_TENSOR_TYPE_U8");
    // fill image data into memory
    ret_code = hbSysWriteMem(&left_input_y_tensor.sysMem, (char *)left_img_data, left_input_y_tensor.sysMem.memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code = hbSysWriteMem(&left_input_uv_tensor.sysMem, (char *)left_img_data + left_input_y_tensor.sysMem.memSize,
                             left_input_uv_tensor.sysMem.memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code = hbSysWriteMem(&right_input_y_tensor.sysMem, (char *)right_img_data, right_input_y_tensor.sysMem.memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");
    ret_code =
        hbSysWriteMem(&right_input_uv_tensor.sysMem, (char *)right_img_data + right_input_y_tensor.sysMem.memSize,
                      right_input_uv_tensor.sysMem.memSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysWriteMem failed");

    // make sure memory data is flushed to DDR before inference
    ret_code = hbSysFlushMem(&left_input_y_tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&left_input_uv_tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&right_input_y_tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    ret_code = hbSysFlushMem(&right_input_uv_tensor.sysMem, HB_SYS_MEM_CACHE_CLEAN);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
  } else {
    RCLCPP_ERROR(logger_, "=> input_tensor_type is not in [HB_DNN_TENSOR_TYPE_U8]");
    return -1;
  }
#endif

  return ret_code;
}

} // namespace stereonet