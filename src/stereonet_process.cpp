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

StereonetProcess::~StereonetProcess() {
  int ret_code = 0;
  // Free input memory
  for (int i = 0; i < max_memory_count_; i++) {
    for (size_t j = 0; j < batch_input_tensors_[i].size(); j++) {
      ret_code = hbSysFreeMem(&TENSOR_SYSMEM(batch_input_tensors_[i][j], 0));
#ifdef PLATFORM_X5
      if (input_tensor_type_ == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
        ret_code = hbSysFreeMem(&TENSOR_SYSMEM(batch_input_tensors_[i][j], 1));
      }
#endif
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFreeMem failed");
    }
  }
  // Free output memory
  for (int i = 0; i < max_memory_count_; i++) {
    for (size_t j = 0; j < batch_output_tensors_[i].size(); j++) {
      ret_code = hbSysFreeMem(&TENSOR_SYSMEM(batch_output_tensors_[i][j], 0));
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFreeMem failed");
    }
  }
  // Release dnn handle
  ret_code = hbDNNRelease(packed_dnn_handle_);
  HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNInfer failed");
  LOG_WARN(logger_, "=> release StereonetProcess");
}

int StereonetProcess::init(const std::string &model_path, const std::string &post_version,
                           const int &max_memory_count) {
  int ret_code = 0;
  // load model
  model_path_ = model_path;
  post_version_ = post_version;
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
  LOG_WARN(logger_, "=> ============ init model start ============");
  LOG_WARN(logger_, "=> model name: " << model_name_list_[0]);
  LOG_WARN(logger_, "=> input_count: " << input_count_);
  LOG_WARN(logger_, "=> output_count: " << output_count_);

  // get model input size from input tensor[0]
  hbDNNTensorProperties properties;
  ret_code = hbDNNGetInputTensorProperties(&properties, dnn_handle_, 0);
#if defined(PLATFORM_S100) || defined(PLATFORM_S600)
  properties.quantizeAxis = 3;
#endif
  hbGetInputTensorHW(properties, model_input_h_, model_input_w_);
  LOG_WARN(logger_, "=> model_input_h: " << model_input_h_ << ", model_input_w: " << model_input_w_);

  // prepare input tensor and output tensor
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
  LOG_WARN_ONCE(logger_, "=> ============ init model end ============");

  return ret_code;
}

int StereonetProcess::forward(uint8_t *left_img_data, uint8_t *right_img_data, InferenceHandle &handle) {
  int ret_code = 0;

  int idle_tensor_id = get_idle_tensor();
  {
    ScopeProcessTime t(logger_, "fill_img_to_input_tensor");
    if (idle_tensor_id == -1) {
      LOG_ERROR(logger_, "=> no idle tensor");
      return -1;
    }
    ret_code = fill_img_to_input_tensor(batch_input_tensors_[idle_tensor_id], left_img_data, right_img_data);
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
      ret_code =
          hbSysFlushMem(&TENSOR_SYSMEM(batch_output_tensors_[idle_tensor_id][i], 0), HB_SYS_MEM_CACHE_INVALIDATE);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysFlushMem failed");
    }
  }

  handle = idle_tensor_id;

  return ret_code;
}

int StereonetProcess::forward_sync(std::vector<uint8_t> &left_img_data, std::vector<uint8_t> &right_img_data,
                                   const double &uncertainty_th, cv::Mat &disp, cv::Mat &uncert) {
  int ret_code = 0;
  // forward
  int idle_tensor_id = 0;
  ret_code = forward(left_img_data.data(), right_img_data.data(), idle_tensor_id);
  if (ret_code != 0) return ret_code;
  // postprocess
  ret_code = postprocess(idle_tensor_id, uncertainty_th, disp, uncert);
  return ret_code;
}

#if HOBOT_HAS_RCLCPP
int StereonetProcess::forward_async(std::vector<uint8_t> &left_img_data, std::vector<uint8_t> &right_img_data,
                                    const double &uncertainty_th, std::shared_ptr<CameraIntrinsic> camera_intrinsic,
                                    const sensor_msgs::msg::Image::SharedPtr &stereo_msg,
                                    order_blockqueue<std::shared_ptr<PubData>> &pub_data_queue) {
  int ret_code = 0;

  if (postprocess_thread_pool_ptr_ == nullptr) postprocess_thread_pool_ptr_ = std::make_unique<BS::thread_pool<>>(1);

  // forward
  int idle_tensor_id = 0;
  ret_code = forward(left_img_data.data(), right_img_data.data(), idle_tensor_id);

  // postprocess
  postprocess_thread_pool_ptr_->detach_task([this, idle_tensor_id, uncertainty_th, left_img_data, right_img_data,
                                             camera_intrinsic, stereo_msg, &pub_data_queue]() {
    cv::Mat disp, uncert;
    postprocess(idle_tensor_id, uncertainty_th, disp, uncert);

    cv::Mat depth;
    disp_to_depth(disp, depth, *camera_intrinsic);

    auto pub_data = std::make_shared<PubData>();
    pub_data->timestamp = static_cast<uint64_t>(stereo_msg->header.stamp.sec) * 1'000'000'000 +
                          static_cast<uint64_t>(stereo_msg->header.stamp.nanosec);
    pub_data->header = stereo_msg->header;
    pub_data->origin_stereo_msg = stereo_msg;
    pub_data->disp = disp;
    pub_data->uncert = uncert;
    pub_data->depth = depth;
    pub_data->rectify_left_img_data = left_img_data;
    pub_data->rectify_right_img_data = right_img_data;
    cv::Mat left_bgr, right_bgr;
    ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_left_img_data.data(), left_bgr, pub_data->disp.cols,
                                     pub_data->disp.rows);
    ImgConvertUtils::nv12_to_bgr_mat(pub_data->rectify_right_img_data.data(), right_bgr, pub_data->disp.cols,
                                     pub_data->disp.rows);
    pub_data->left_bgr = left_bgr;
    pub_data->right_bgr = right_bgr;
    if (pub_data_queue.size() >= 1) {
      pub_data_queue.pop_front();
    }
    pub_data_queue.put(pub_data->timestamp, pub_data);
  });

  return ret_code;
}
#endif

int StereonetProcess::postprocess(const InferenceHandle &handle, const double &uncertainty_th, cv::Mat &disp,
                                  cv::Mat &uncert) {
  ScopeProcessTime t(logger_, "postprocess");
  int ret_code = 0;
  int idle_tensor_id = handle;

  // get shape info
  auto &outputs = batch_output_tensors_[idle_tensor_id];
  if (outputs.size() < 2) {
    LOG_ERROR(logger_, "=> not enough output tensors for postprocess, size=" << outputs.size());
    set_tensor_idle(idle_tensor_id);
    return -1;
  }

  auto disp_tensor = outputs[0];
  auto spx_tensor = outputs[1];
  const int32_t *disp_shape = disp_tensor.properties.validShape.dimensionSize;
  int disp_h_dim = disp_shape[2];
  int disp_w_dim = disp_shape[3];
  const int32_t *spx_shape = spx_tensor.properties.validShape.dimensionSize;
  int spx_h_dim = spx_shape[2];
  int spx_w_dim = spx_shape[3];

  // postprocess
  if (post_version_ == "v2.0" || (output_count_ == 2 && disp_h_dim == spx_h_dim && disp_w_dim == spx_w_dim)) {
    ret_code = postprocess_convex_upsampling(outputs, disp);
  } else if (post_version_ == "v2.2" || post_version_ == "v2.3" || post_version_ == "v2.4" ||
             (output_count_ == 2 && disp_h_dim * 4 == spx_h_dim && disp_w_dim * 4 == spx_w_dim)) {
    ret_code = postprocess_convex_upsampling_with_interp(outputs, disp);
  } else if (post_version_ == "v2.1" || (output_count_ == 4 && disp_h_dim == spx_h_dim && disp_w_dim == spx_w_dim)) {
    std::vector<hbDNNTensor> infer_disp_tensor(outputs.begin(), outputs.begin() + 2);
    ret_code = postprocess_convex_upsampling(infer_disp_tensor, disp);
    if (uncertainty_th > 0 && ret_code == 0) {
      std::vector<hbDNNTensor> init_disp_tensor(outputs.begin() + 2, outputs.begin() + 4);
      cv::Mat init_disp, mask;
      ret_code = postprocess_convex_upsampling(init_disp_tensor, init_disp);
      if (ret_code == 0) {
        uncert = cv::abs(init_disp - disp) / init_disp;
        cv::threshold(uncert, mask, uncertainty_th, 1, cv::THRESH_BINARY_INV);
        disp = disp.mul(mask);
      }
    }
  } else if (post_version_ == "v2.4_uncert" ||
             (output_count_ == 4 && disp_h_dim * 4 == spx_h_dim && disp_w_dim * 4 == spx_w_dim)) {
    std::vector<hbDNNTensor> infer_disp_tensor(outputs.begin(), outputs.begin() + 2);
    ret_code = postprocess_convex_upsampling_with_interp(infer_disp_tensor, disp);
    if (uncertainty_th > 0 && ret_code == 0) {
      std::vector<hbDNNTensor> init_disp_tensor(outputs.begin() + 2, outputs.begin() + 4);
      cv::Mat init_disp, mask;
      ret_code = postprocess_convex_upsampling_with_interp(init_disp_tensor, init_disp);
      if (ret_code == 0) {
        uncert = cv::abs(init_disp - disp) / init_disp;
        cv::threshold(uncert, mask, uncertainty_th, 1, cv::THRESH_BINARY_INV);
        disp = disp.mul(mask);
      }
    }
  } else {
    LOG_ERROR(logger_, "\033[31m=> not support postprocess! output_count: "
                           << output_count_ << ", disp dim [" << disp_h_dim << ", " << disp_w_dim << "], spx dim ["
                           << spx_h_dim << ", " << spx_w_dim << "]\033[0m");
    ret_code = -1;
  }

  // reset idle tensor
  set_tensor_idle(idle_tensor_id);

  return ret_code;
}

int StereonetProcess::postprocess_out_disp_depth(const int idle_tensor_id, const double &uncertainty_th,
                                                 const CameraIntrinsic &camera_intrinsic, cv::Mat &disp,
                                                 cv::Mat &uncert, cv::Mat &depth) {
  int ret_code = 0;
  ret_code = postprocess(idle_tensor_id, uncertainty_th, disp, uncert);
  disp_to_depth(disp, depth, camera_intrinsic);
  return ret_code;
}

int StereonetProcess::postprocess_out_depth(const int idle_tensor_id, const double &uncertainty_th,
                                            const CameraIntrinsic &camera_intrinsic, cv::Mat &depth) {
  int ret_code = 0;
  cv::Mat disp, uncert;
  ret_code = postprocess(idle_tensor_id, uncertainty_th, disp, uncert);
  disp_to_depth(disp, depth, camera_intrinsic);
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
    auto disp = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

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
    auto disp = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx = reinterpret_cast<int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

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
    auto disp = reinterpret_cast<int32_t *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx = reinterpret_cast<int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

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
    auto disp = reinterpret_cast<int16_t *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx = reinterpret_cast<int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    // multiply element-wise and then add in the c channel
    for (int i = 0; i < c_dim; ++i) {
      Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim,
                                                                                     w_dim);
      Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim,
                                                                                    w_dim);
      result.noalias() += matrix_disp.cast<float>().cwiseProduct(matrix_spx.cast<float>());
    }
  } else {
    LOG_ERROR(logger_, "=> output tensor type unsupported! tensor[0]: "
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

int StereonetProcess::postprocess_convex_upsampling_with_interp(const std::vector<hbDNNTensor> &tensors,
                                                                cv::Mat &out_mat) {
  const int32_t *disp_shape = tensors[0].properties.validShape.dimensionSize;
  // int disp_c_dim = disp_shape[1];
  int disp_h_dim = disp_shape[2];
  int disp_w_dim = disp_shape[3];
  int total_disp_size = disp_h_dim * disp_w_dim;

  const int32_t *spx_shape = tensors[1].properties.validShape.dimensionSize;
  int spx_c_dim = spx_shape[1];
  int spx_h_dim = spx_shape[2];
  int spx_w_dim = spx_shape[3];
  int total_size = spx_h_dim * spx_w_dim;
  int32_t scale_h = spx_h_dim / disp_h_dim, scale_w = spx_w_dim / disp_w_dim;

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
  // calc disp
  out_mat = cv::Mat::zeros(spx_h_dim, spx_w_dim, CV_32FC1);
  float *result_ptr = reinterpret_cast<float *>(out_mat.data);
  if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S32 &&
      tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
    auto disp = reinterpret_cast<int32_t *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx = reinterpret_cast<int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    for (int32_t i = 0; i < spx_c_dim; ++i) {
      for (int32_t y = 0; y < spx_h_dim; ++y) {
        // compute y-index for low-res disparity (nearest-neighbor sampling)
        int32_t idx_y = y / scale_h;
        // offset of this output row in result_ptr
        int32_t output_offset = spx_w_dim * y;
        for (int32_t x = 0; x < spx_w_dim; x += 4) {
          // compute x-index for low-res disparity (nearest-neighbor sampling)
          int32_t idx_x = x / scale_w;

          // load spx
          int16x4_t spx_s16 = vld1_s16(&spx[y * spx_w_dim + x]);
          int32x4_t spx_s32 = vmovl_s16(spx_s16);

          // load disp
          int32_t disp_val_scalar = disp[idx_y * disp_w_dim + idx_x];
          int32x4_t disp_s32 = vdupq_n_s32(disp_val_scalar);

          // convert to float
          float32x4_t spx_f32 = vcvtq_f32_s32(spx_s32);
          float32x4_t disp_f32 = vcvtq_f32_s32(disp_s32);

          // disp * spx
          float32x4_t mul_result = vmulq_f32(disp_f32, spx_f32);

          // accumulate into output buffer
          float32x4_t current_output = vld1q_f32(&result_ptr[output_offset + x]);
          float32x4_t updated_output = vaddq_f32(current_output, mul_result);
          vst1q_f32(&result_ptr[output_offset + x], updated_output);
        }
      }
      // move to next disparity row
      disp += total_disp_size;
      // move to next spx row
      spx += total_size;
    }

    // result * scale_factor
    if (scale_factor != 1.0f) {
      for (int32_t j = 0; j < total_size; j += 4) {
        vst1q_f32(result_ptr + j, vmulq_n_f32(vld1q_f32(result_ptr + j), scale_factor));
      }
    }
  } else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    auto disp = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    for (int32_t i = 0; i < spx_c_dim; ++i) {
      for (int32_t y = 0; y < spx_h_dim; ++y) {
        // compute y-index for low-res disparity (nearest-neighbor sampling)
        int32_t idx_y = y / scale_h;
        // offset of this output row in result_ptr
        int32_t output_offset = spx_w_dim * y;
        for (int32_t x = 0; x < spx_w_dim; x += 4) {
          // compute x-index for low-res disparity (nearest-neighbor sampling)
          int32_t idx_x = x / scale_w;

          // load spx
          float32x4_t spx_f32 = vld1q_f32(&spx[y * spx_w_dim + x]);

          // load disp
          float disp_val_scalar = disp[idx_y * disp_w_dim + idx_x];
          float32x4_t disp_f32 = vdupq_n_f32(disp_val_scalar);

          // disp * spx
          float32x4_t mul_result = vmulq_f32(disp_f32, spx_f32);

          // accumulate into output buffer
          float32x4_t current_output = vld1q_f32(&result_ptr[output_offset + x]);
          float32x4_t updated_output = vaddq_f32(current_output, mul_result);
          vst1q_f32(&result_ptr[output_offset + x], updated_output);
        }
      }
      // move to next disparity row
      disp += total_disp_size;
      // move to next spx row
      spx += total_size;
    }

    // result * scale_factor
    if (scale_factor != 1.0f) {
      for (int32_t j = 0; j < total_size; j += 4) {
        float32x4_t cur = vld1q_f32(result_ptr + j);
        float32x4_t scaled = vmulq_n_f32(cur, scale_factor);
        vst1q_f32(result_ptr + j, scaled);
      }
    }
  } else {
    LOG_ERROR(logger_, "=> output tensor type unsupported! tensor[0]: "
                           << magic_enum::enum_name(static_cast<hbDNNDataType>(tensors[0].properties.tensorType))
                           << ", tensor[1]: "
                           << magic_enum::enum_name(static_cast<hbDNNDataType>(tensors[1].properties.tensorType)));
    return -1;
  }
  return 0;
}

int StereonetProcess::prepare_input_tensor(std::vector<hbDNNTensor> &input_tensors) {
  int ret_code = 0;
  LOG_WARN_ONCE(logger_, "=> ----- prepare_input_tensor -----");

  // allocate memory for input tensor
  input_tensors.resize(input_count_);
  for (int i = 0; i < input_count_; i++) {
    auto &tensor = input_tensors[i];
    // get input tensor properties
    hbDNNTensorProperties properties;
    ret_code = hbDNNGetInputTensorProperties(&properties, dnn_handle_, i);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetInputTensorProperties failed");
    LOG_WARN_ONCE(logger_, "=> input tensor type is "
                               << magic_enum::enum_name(static_cast<hbDNNDataType>(properties.tensorType)));
    input_tensor_type_ = properties.tensorType;

#ifdef PLATFORM_X5
    if ((properties.tensorType != HB_DNN_IMG_TYPE_NV12) && (properties.tensorType != HB_DNN_IMG_TYPE_NV12_SEPARATE)) {
      LOG_ERROR(logger_, "=> input tensor type is not in [HB_DNN_IMG_TYPE_NV12, HB_DNN_IMG_TYPE_NV12_SEPARATE]");
      return -1;
    }
#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)
    if ((properties.tensorType != HB_DNN_TENSOR_TYPE_U8)) {
      LOG_ERROR(logger_, "=> input tensor type is not in [HB_DNN_TENSOR_TYPE_U8]");
      return -1;
    }
#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)
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
      LOG_WARN_ONCE(logger_, "=> input[" << i << "].memsize: " << tensor.sysMem[0].memSize);
    } else if (properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
      ret_code = hbSysAllocCachedMem(&tensor.sysMem[0], model_input_h_ * model_input_w_);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      tensor.sysMem[0].memSize = model_input_h_ * model_input_w_;

      ret_code = hbSysAllocCachedMem(&tensor.sysMem[1], model_input_h_ * model_input_w_ / 2);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      tensor.sysMem[1].memSize = model_input_h_ * model_input_w_ / 2;
      LOG_WARN_ONCE(logger_, "=> input[" << i << "].memsize[0]: " << tensor.sysMem[0].memSize);
      LOG_WARN_ONCE(logger_, "=> input[" << i << "].memsize[1]: " << tensor.sysMem[1].memSize);
    } else {
      return -1;
    }
#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)
    if (properties.tensorType == HB_DNN_TENSOR_TYPE_U8) {
      ret_code = hbSysAllocCachedMem(&tensor.sysMem, properties.alignedByteSize);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      LOG_WARN_ONCE(logger_, "=> input tensor size: " << tensor.sysMem.memSize);
    } else {
      return -1;
    }
#endif
  }
  return ret_code;
}

int StereonetProcess::prepare_output_tensor(std::vector<hbDNNTensor> &output_tensors) {
  int ret_code = 0;
  LOG_WARN_ONCE(logger_, "=> ----- prepare_output_tensor -----");
  output_tensors.resize(output_count_);
  for (int i = 0; i < output_count_; ++i) {
    ret_code = hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle_, i);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetOutputTensorProperties failed");
    LOG_WARN_ONCE(logger_, "=> output tensor type is " << magic_enum::enum_name(
                               static_cast<hbDNNDataType>(output_tensors[i].properties.tensorType)));
    ret_code = hbSysAllocCachedMem(&TENSOR_SYSMEM(output_tensors[i], 0), output_tensors[i].properties.alignedByteSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
    LOG_WARN_ONCE(logger_, "=> output[" << i << "].memsize: " << output_tensors[i].properties.alignedByteSize);
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

int StereonetProcess::set_tensor_idle(const int &tensor_id) {
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
    LOG_ERROR(logger_, "=> input_tensor_type is not in [HB_DNN_IMG_TYPE_NV12, HB_DNN_IMG_TYPE_NV12_SEPARATE]");
    return -1;
  }
#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)
  hbDNNTensor &left_input_y_tensor = input_tensors[0];
  hbDNNTensor &left_input_uv_tensor = input_tensors[1];
  hbDNNTensor &right_input_y_tensor = input_tensors[2];
  hbDNNTensor &right_input_uv_tensor = input_tensors[3];

  if (input_tensor_type_ == HB_DNN_TENSOR_TYPE_U8) {
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
    LOG_ERROR(logger_, "=> input_tensor_type is not in [HB_DNN_TENSOR_TYPE_U8]");
    return -1;
  }
#endif

  return ret_code;
}

void StereonetProcess::get_model_input_size(int &w, int &h) const {
  w = model_input_w_;
  h = model_input_h_;
}

/*
void StereonetProcess::disp_to_depth(const cv::Mat &disp, cv::Mat &depth, const CameraIntrinsic &camera_intrinsic) {
  depth = cv::Mat::zeros(disp.size(), CV_16UC1);
  for (int i = 0; i < disp.rows; ++i) {
    for (int j = 0; j < disp.cols; ++j) {
      float d = disp.at<float>(i, j);
      if (d <= 0.0) {
        depth.at<uint16_t>(i, j) = 0;
      } else {
        float z = (camera_intrinsic.baseline * camera_intrinsic.fx * 1000.0) / (d + camera_intrinsic.doffs); // in mm
        if (z > 65535.0) {
          depth.at<uint16_t>(i, j) = 65535;
        } else {
          depth.at<uint16_t>(i, j) = static_cast<uint16_t>(z);
        }
      }
    }
  }
}
*/

void StereonetProcess::disp_to_depth(const cv::Mat &disp, cv::Mat &depth, const CameraIntrinsic &camera_intrinsic) {
  depth.create(disp.size(), CV_16UC1);
  const int rows = disp.rows;
  const int cols = disp.cols;

  float fb = camera_intrinsic.baseline * camera_intrinsic.fx * 1000.0f; // in mm
  float doffs = camera_intrinsic.doffs;

  for (int i = 0; i < rows; ++i) {
    const float *disp_ptr = disp.ptr<float>(i);
    uint16_t *depth_ptr = depth.ptr<uint16_t>(i);

    int j = 0;
    // use NEON to process 4 pixels at a time
    for (; j <= cols - 4; j += 4) {
      float32x4_t d = vld1q_f32(disp_ptr + j);

      // mask for d > 0
      uint32x4_t mask = vcgtq_f32(d, vdupq_n_f32(0.0f));

      // z = scale / (d+doffs)
      float32x4_t z = vdivq_f32(vdupq_n_f32(fb), vaddq_f32(d, vdupq_n_f32(doffs)));

      // clamp to 65535
      float32x4_t z_clamped = vminq_f32(z, vdupq_n_f32(65535.0f));

      // set 0 if d <= 0
      z_clamped = vbslq_f32(mask, z_clamped, vdupq_n_f32(0.0f));

      // convert to uint16_t
      uint16x4_t z_u16 = vmovn_u32(vcvtq_u32_f32(z_clamped));

      vst1_u16(depth_ptr + j, z_u16);
    }

    // process remaining pixels
    for (; j < cols; ++j) {
      float d = disp_ptr[j];
      if (d <= 0.0f)
        depth_ptr[j] = 0;
      else {
        float z = fb / (d + doffs);
        depth_ptr[j] = (z > 65535.0f) ? 65535 : static_cast<uint16_t>(z);
      }
    }
  }
}

void StereonetProcess::depth_to_pointcloud(const cv::Mat &depth, const CameraIntrinsic &camera_intrinsic,
                                           std::vector<PointXYZ> &pointcloud, const float &max_depth) {
  CV_Assert(depth.type() == CV_16UC1);

  const int rows = depth.rows;
  const int cols = depth.cols;
  pointcloud.clear();
  pointcloud.resize(static_cast<size_t>(rows) * cols); // allocate once

  const float inv_fx = 1.0f / camera_intrinsic.fx;
  const float inv_fy = 1.0f / camera_intrinsic.fy;
  const float cx = camera_intrinsic.cx;
  const float cy = camera_intrinsic.cy;

  size_t out_idx = 0;
  for (int i = 0; i < rows; ++i) {
    const uint16_t *dptr = depth.ptr<uint16_t>(i);
    const float y_factor = (i - cy) * inv_fy; // reuse per row
    for (int j = 0; j < cols; ++j) {
      const uint16_t d = dptr[j];
      if (d == 0) continue;        // invalid depth
      const float Z = d * 0.001f;  // mm -> m
      if (Z > max_depth) continue; // invalid depth
      const float X = (j - cx) * Z * inv_fx;
      const float Y = y_factor * Z;
      pointcloud[out_idx++] = PointXYZ(X, Y, Z);
    }
  }
  pointcloud.resize(out_idx);
}

void StereonetProcess::depth_to_pointcloud_rgb(const cv::Mat &depth, const cv::Mat &rgb,
                                               const CameraIntrinsic &camera_intrinsic,
                                               std::vector<PointXYZRGB> &pointcloud, const float &max_depth) {
  CV_Assert(depth.type() == CV_16UC1);
  CV_Assert(rgb.type() == CV_8UC3);

  const int rows = depth.rows;
  const int cols = depth.cols;
  pointcloud.clear();
  pointcloud.resize(static_cast<size_t>(rows) * cols); // allocate once

  const float inv_fx = 1.0f / camera_intrinsic.fx;
  const float inv_fy = 1.0f / camera_intrinsic.fy;
  const float cx = camera_intrinsic.cx;
  const float cy = camera_intrinsic.cy;

  size_t out_idx = 0;
  for (int i = 0; i < rows; ++i) {
    const uint16_t *dptr = depth.ptr<uint16_t>(i);
    const cv::Vec3b *rgb_ptr = rgb.ptr<cv::Vec3b>(i);
    const float y_factor = (i - cy) * inv_fy; // reuse per row
    for (int j = 0; j < cols; ++j) {
      const uint16_t d = dptr[j];
      if (d == 0) continue;        // invalid depth
      const float Z = d * 0.001f;  // mm -> m
      if (Z > max_depth) continue; // invalid depth
      const float X = (j - cx) * Z * inv_fx;
      const float Y = y_factor * Z;
      pointcloud[out_idx++] = PointXYZRGB(X, Y, Z, rgb_ptr[j][2], rgb_ptr[j][1], rgb_ptr[j][0]);
    }
  }
  pointcloud.resize(out_idx);
}

void StereonetProcess::dump_pcd_file(const std::string &filename, const std::vector<PointXYZ> &pointcloud,
                                     const std::string &format) {
  bool is_ascii = (format == "ascii");
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) return;

  const size_t n = pointcloud.size();

  // header
  std::ostringstream header;
  header << "# .PCD v0.7 - Point Cloud Data file format\n";
  header << "VERSION 0.7\n";
  header << "FIELDS x y z\n";
  header << "SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n";
  header << "WIDTH " << n << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << n << "\n";
  header << "DATA " << (is_ascii ? "ascii" : "binary") << "\n";
  ofs.write(header.str().c_str(), header.str().size());

  // write data
  if (is_ascii) {
    for (const auto &point : pointcloud) {
      ofs << point.X << " " << point.Y << " " << point.Z << "\n";
    }
  } else {
    for (const auto &point : pointcloud) {
      float data[3] = {point.X, point.Y, point.Z};
      ofs.write(reinterpret_cast<const char *>(data), sizeof(data));
    }
  }

  ofs.close();
}

void StereonetProcess::dump_pcd_file_rgb(const std::string &filename, const std::vector<PointXYZRGB> &pointcloud,
                                         const std::string &format) {
  bool is_ascii = (format == "ascii");
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs.is_open()) return;

  size_t n = pointcloud.size();

  // header
  std::ostringstream header;
  header << "# .PCD v0.7 - Point Cloud Data file format\n";
  header << "VERSION 0.7\n";
  header << "FIELDS x y z rgb\n";
  header << "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n";
  header << "WIDTH " << n << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << n << "\n";
  header << "DATA " << (is_ascii ? "ascii" : "binary") << "\n";
  ofs.write(header.str().c_str(), header.str().size());

  // helper to pack RGB to float
  auto packRGB = [](uint8_t r, uint8_t g, uint8_t b) -> float {
    uint32_t rgb = (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
    float f;
    std::memcpy(&f, &rgb, sizeof(float));
    return f;
  };

  // write data
  if (is_ascii) {
    for (const auto &p : pointcloud) {
      float rgb_f = packRGB(p.R, p.G, p.B);
      ofs << p.X << " " << p.Y << " " << p.Z << " " << rgb_f << "\n";
    }
  } else {
    for (const auto &p : pointcloud) {
      float data[4] = {p.X, p.Y, p.Z, packRGB(p.R, p.G, p.B)};
      ofs.write(reinterpret_cast<const char *>(data), sizeof(data));
    }
  }

  ofs.close();
}

static double compute_near_depth_percentile(const cv::Mat &depth, double percentile = 0.02) {
  std::vector<double> valid;
  valid.reserve(depth.total());

  for (int y = 0; y < depth.rows; ++y) {
    const uint16_t *ptr = depth.ptr<uint16_t>(y);
    for (int x = 0; x < depth.cols; ++x)
      if (ptr[x] > 0) valid.push_back(ptr[x]);
  }
  if (valid.empty()) return -1;
  size_t k = static_cast<size_t>(percentile * valid.size());
  std::nth_element(valid.begin(), valid.begin() + k, valid.end());
  return valid[k];
}

void StereonetProcess::convert_visual_img(const cv::Mat &rgb, const cv::Mat &disp, const cv::Mat &depth,
                                          const CameraIntrinsic &camera_intrinsic, cv::Mat &visual_img,
                                          int depth_decimal_num) {
  if (depth_decimal_num < 2) depth_decimal_num = 2; // cm
  if (depth_decimal_num > 3) depth_decimal_num = 3; // mm
  CV_Assert(rgb.type() == CV_8UC3);
  CV_Assert(disp.type() == CV_32FC1);
  CV_Assert(depth.type() == CV_16UC1);

  double fb = camera_intrinsic.baseline * camera_intrinsic.fx;
  double z_near = compute_near_depth_percentile(depth, 0.02);
  double z_far = z_near + 3000.0;
  int d_max = static_cast<int>(fb / (z_near / 1000.0) - camera_intrinsic.doffs);
  int d_min = static_cast<int>(fb / (z_far / 1000.0) - camera_intrinsic.doffs);
  disp.convertTo(visual_img, CV_8UC1, 255.0 / (d_max - d_min), -d_min * 255.0 / (d_max - d_min));
  visual_img.setTo(0, disp < d_min);
  visual_img.setTo(255, disp > d_max);
  cv::cvtColor(visual_img, visual_img, cv::COLOR_GRAY2BGR);

  static cv::Mat lut;
  if (lut.empty()) {
    cv::Mat tmp(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++) tmp.at<uchar>(i) = i;
    cv::applyColorMap(tmp, lut, cv::COLORMAP_JET);
  }
  cv::LUT(visual_img, lut, visual_img);
  cv::Mat mask = (disp == 0);
  visual_img.setTo(cv::Vec3b(0, 0, 0), mask);
  cv::vconcat(rgb, visual_img, visual_img);

  double font_scale = std::min(rgb.cols, rgb.rows) / 700.0;
  int set_num = 6;
  int x_step = rgb.cols / set_num;
  int y_step = rgb.rows / set_num;

  // draw lines
  for (int i = 1; i < set_num; ++i) {
    // vertical line
    cv::line(visual_img, cv::Point(i * x_step, 0), cv::Point(i * x_step, visual_img.rows), cv::Scalar(255, 255, 255),
             1);
    // horizontal line
    cv::line(visual_img, cv::Point(0, i * y_step), cv::Point(rgb.cols, i * y_step), cv::Scalar(255, 255, 255), 1);
    cv::line(visual_img, cv::Point(0, rgb.rows + i * y_step), cv::Point(rgb.cols, rgb.rows + i * y_step),
             cv::Scalar(255, 255, 255), 1);
  }

  // draw depth values
  for (int i = 1; i < set_num; ++i) {
    for (int j = 1; j < set_num; ++j) {
      int x = i * x_step;
      int y = j * y_step;
      float depth_value = depth.at<uint16_t>(y, x) * 0.001f; // convert mm to m
      std::stringstream depth_text;
      depth_text << std::fixed << std::setprecision(depth_decimal_num) << depth_value << "m";
      cv::putText(visual_img, depth_text.str(), cv::Point(x + 5, y - 5), cv::FONT_HERSHEY_SIMPLEX, font_scale,
                  CV_RGB(255, 255, 255), 2);
      cv::putText(visual_img, depth_text.str(), cv::Point(x + 5, rgb.rows + y - 5), cv::FONT_HERSHEY_SIMPLEX,
                  font_scale, CV_RGB(255, 255, 255), 2);
    }
  }
}

// calc percentile, percent should be [0, 100]
static float calc_percentile(const std::vector<float> &data, float percent) {
  if (data.empty()) {
    return 0.0f;
  }
  if (data.size() == 1) {
    return data[0];
  }

  float pos = (percent / 100.0f) * (static_cast<float>(data.size() - 1));
  int idx_low = static_cast<int>(std::floor(pos));
  int idx_high = static_cast<int>(std::ceil(pos));
  float frac = pos - static_cast<float>(idx_low);

  float vlow = data[idx_low];
  float vhigh = data[idx_high];
  return vlow + (vhigh - vlow) * frac;
}

static inline float clamp01(float x) {
  return std::max(0.0f, std::min(1.0f, x));
}

// easy linear interpolation
static inline uchar lerpU8(float a, float b, float t) {
  t = clamp01(t);
  return static_cast<uchar>(a + (b - a) * t + 0.5f);
}

// jet color mapping, input [0,1]
static cv::Vec3b jet_color(float v) {
  v = clamp01(v);

  // 0.00 - 0.25: Deep blue -> Light blue
  // 0.25 - 0.50: Light blue -> Green
  // 0.50 - 0.75: Green -> Yellow
  // 0.75 - 1.00: Yellow -> Red
  if (v <= 0.25f) {
    float t = v / 0.25f;
    return cv::Vec3b(lerpU8(128, 255, t), // B
                     lerpU8(0, 128, t),   // G
                     lerpU8(0, 0, t)      // R
    );
  } else if (v <= 0.5f) {
    float t = (v - 0.25f) / 0.25f;
    return cv::Vec3b(lerpU8(255, 255, t), // B
                     lerpU8(128, 255, t), // G
                     lerpU8(0, 0, t)      // R
    );
  } else if (v <= 0.75f) {
    float t = (v - 0.5f) / 0.25f;
    return cv::Vec3b(lerpU8(255, 0, t),   // B
                     lerpU8(255, 255, t), // G
                     lerpU8(0, 255, t)    // R
    );
  } else {
    float t = (v - 0.75f) / 0.25f;
    return cv::Vec3b(lerpU8(0, 0, t),    // B
                     lerpU8(255, 0, t),  // G
                     lerpU8(255, 255, t) // R
    );
  }
}

// normalize piecewise
static float piecewise_normalize(float v, float min_v, float p10, float p50, float p90, float max_v) {
  const float eps = 1e-6f;

  if (v <= 0.0f) {
    return 0.0f;
  }

  if (v <= p10) {
    float denom = std::max(p10 - min_v, eps);
    return (v - min_v) / denom * 0.25f;
  } else if (v <= p50) {
    float denom = std::max(p50 - p10, eps);
    return 0.25f + (v - p10) / denom * 0.25f;
  } else if (v <= p90) {
    float denom = std::max(p90 - p50, eps);
    return 0.50f + (v - p50) / denom * 0.25f;
  } else {
    float denom = std::max(max_v - p90, eps);
    return 0.75f + (v - p90) / denom * 0.25f;
  }
}

cv::Mat StereonetProcess::render_disp_or_depth(const cv::Mat &input, float min_disp, float max_disp, float min_depth,
                                               float max_depth) {
  if (input.empty()) {
    throw std::runtime_error("=> input image is empty");
  }

  bool is_disp = (input.type() == CV_32FC1);
  bool is_depth = (input.type() == CV_16UC1);

  if (!is_disp && !is_depth) {
    throw std::runtime_error("=> unsupported input type, only CV_32FC1 or CV_16UC1");
  }

  // convert to float
  cv::Mat img_f;
  if (is_disp) {
    input.convertTo(img_f, CV_32F);
  } else {
    input.convertTo(img_f, CV_32F);
  }

  // remove nan / inf
  for (int y = 0; y < img_f.rows; ++y) {
    float *row = img_f.ptr<float>(y);
    for (int x = 0; x < img_f.cols; ++x) {
      float &v = row[x];
      if (!std::isfinite(v)) {
        v = 0.0f;
      }
    }
  }

  // normalize
  for (int y = 0; y < img_f.rows; ++y) {
    float *row = img_f.ptr<float>(y);
    for (int x = 0; x < img_f.cols; ++x) {
      float &v = row[x];
      if (is_disp) {
        if (v < min_disp || v > max_disp) {
          v = 0.0f;
        }
      } else if (is_depth) {
        if (v < min_depth || v > max_depth) {
          v = 0.0f;
        }
      }
    }
  }

  // collect valid values
  std::vector<float> valid_values;
  valid_values.reserve(img_f.rows * img_f.cols);

  for (int y = 0; y < img_f.rows; ++y) {
    const float *row = img_f.ptr<float>(y);
    for (int x = 0; x < img_f.cols; ++x) {
      float v = row[x];
      if (v > 0.0f) {
        valid_values.push_back(v);
      }
    }
  }

  cv::Mat color(img_f.size(), CV_8UC3, cv::Scalar(0, 0, 0));

  if (valid_values.empty()) {
    return color;
  }

  std::sort(valid_values.begin(), valid_values.end());

  float min_v = valid_values.front();
  float max_v = valid_values.back();
  float p10 = calc_percentile(valid_values, 10.0f);
  float p50 = calc_percentile(valid_values, 50.0f);
  float p90 = calc_percentile(valid_values, 90.0f);

  // render
  for (int y = 0; y < img_f.rows; ++y) {
    const float *row = img_f.ptr<float>(y);
    cv::Vec3b *out_row = color.ptr<cv::Vec3b>(y);

    for (int x = 0; x < img_f.cols; ++x) {
      float v = row[x];
      if (v <= 0.0f) {
        out_row[x] = cv::Vec3b(0, 0, 0);
        continue;
      }

      float norm_v = piecewise_normalize(v, min_v, p10, p50, p90, max_v);
      out_row[x] = jet_color(norm_v);
    }
  }

  return color;
}

} // namespace stereonet
