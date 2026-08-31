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

#ifdef PLATFORM_S600
  if (model_input_h_ == 308 && model_input_w_ == 560) {
    // special case for S600 model
    model_input_h_ = 352;
    model_input_w_ = 640;
    inner_model_input_h_ = 308;
    inner_model_input_w_ = 560;
    LOG_WARN(logger_,
             "=> inner_model_input_h: " << inner_model_input_h_ << ", inner_model_input_w: " << inner_model_input_w_);
  }
#endif
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
#ifdef PLATFORM_S600
    if (inner_model_input_h_ > 0 && inner_model_input_w_ > 0) {
      cv::Mat left_img_bgr(model_input_h_, model_input_w_, CV_8UC3);
      cv::Mat right_img_bgr(model_input_h_, model_input_w_, CV_8UC3);
      ImgConvertUtils::nv12_to_bgr24_neon(left_img_data, left_img_bgr.data, model_input_w_, model_input_h_);
      ImgConvertUtils::nv12_to_bgr24_neon(right_img_data, right_img_bgr.data, model_input_w_, model_input_h_);
      cv::resize(left_img_bgr, left_img_bgr, cv::Size(inner_model_input_w_, inner_model_input_h_));
      cv::resize(right_img_bgr, right_img_bgr, cv::Size(inner_model_input_w_, inner_model_input_h_));
      const int align_width = ((left_img_bgr.cols + 63) / 64) * 64;
      if (align_width != left_img_bgr.cols) {
        const int pad_right = align_width - left_img_bgr.cols;
        cv::copyMakeBorder(left_img_bgr, left_img_bgr, 0, 0, 0, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::copyMakeBorder(right_img_bgr, right_img_bgr, 0, 0, 0, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      }
      std::vector<uint8_t> new_left_img_data, new_right_img_data;
      new_left_img_data.resize(left_img_bgr.cols * left_img_bgr.rows * 3 / 2);
      new_right_img_data.resize(right_img_bgr.cols * right_img_bgr.rows * 3 / 2);
      ImgConvertUtils::bgr_mat_to_nv12(left_img_bgr, new_left_img_data.data());
      ImgConvertUtils::bgr_mat_to_nv12(right_img_bgr, new_right_img_data.data());
      ret_code = fill_img_to_input_tensor(batch_input_tensors_[idle_tensor_id], new_left_img_data.data(),
                                          new_right_img_data.data());
    } else {
#endif
      ret_code = fill_img_to_input_tensor(batch_input_tensors_[idle_tensor_id], left_img_data, right_img_data);
#ifdef PLATFORM_S600
    }
#endif
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
    disparity_to_depth(disp, depth, *camera_intrinsic);

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
  // S600
  if (outputs.size() == 1) {
    ret_code = postprocess_only_disp(outputs, disp);
    set_tensor_idle(idle_tensor_id);
    return ret_code;
  }

  // X5 | S100
  if (outputs.size() < 2) {
    LOG_ERROR(logger_, "=> not enough output tensors for postprocess, size=" << outputs.size());
    set_tensor_idle(idle_tensor_id);
    return -1;
  }

  auto disp_tensor = outputs[0];
  auto spx_tensor = outputs[1];
  const int32_t *disp_shape = disp_tensor.properties.validShape.dimensionSize;
  int disp_c_dim = disp_shape[1];
  int disp_h_dim = disp_shape[2];
  int disp_w_dim = disp_shape[3];
  const int32_t *spx_shape = spx_tensor.properties.validShape.dimensionSize;
  int spx_c_dim = spx_shape[1];
  int spx_h_dim = spx_shape[2];
  int spx_w_dim = spx_shape[3];

  // postprocess
  if (output_count_ == 2 && disp_c_dim == 9 && spx_c_dim == 36 && disp_h_dim == spx_h_dim && disp_w_dim == spx_w_dim) {
    // V3.2
    ret_code = postprocess_convex_upsampling_2x_logits(outputs, disp);
  } else if (post_version_ == "v2.0" || (output_count_ == 2 && disp_h_dim == spx_h_dim && disp_w_dim == spx_w_dim)) {
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
  disparity_to_depth(disp, depth, camera_intrinsic);
  return ret_code;
}

int StereonetProcess::postprocess_out_depth(const int idle_tensor_id, const double &uncertainty_th,
                                            const CameraIntrinsic &camera_intrinsic, cv::Mat &depth) {
  int ret_code = 0;
  cv::Mat disp, uncert;
  ret_code = postprocess(idle_tensor_id, uncertainty_th, disp, uncert);
  disparity_to_depth(disp, depth, camera_intrinsic);
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

/*
int StereonetProcess::postprocess_convex_upsampling(const std::vector<hbDNNTensor> &tensors, cv::Mat &out_mat) {
  // ----------------------------
  // 1. Check tensor count
  // ----------------------------
  if (tensors.size() < 2) {
    LOG_ERROR(logger_, "=> tensors size < 2");
    return -1;
  }

  // ----------------------------
  // 2. Use valid shape as logical shape
  // ----------------------------
  const int32_t *disp_shape = tensors[0].properties.validShape.dimensionSize;
  const int32_t *spx_shape = tensors[1].properties.validShape.dimensionSize;

  const int32_t c_dim = disp_shape[1];
  const int32_t h_dim = disp_shape[2];
  const int32_t w_dim = disp_shape[3];

  const int32_t spx_c_dim = spx_shape[1];
  const int32_t spx_h_dim = spx_shape[2];
  const int32_t spx_w_dim = spx_shape[3];

  if (c_dim != spx_c_dim || h_dim != spx_h_dim || w_dim != spx_w_dim) {
    LOG_ERROR(logger_, "=> disp/spx shape mismatch, " << "disp=(" << c_dim << "," << h_dim << "," << w_dim << "), "
                                                      << "spx=(" << spx_c_dim << "," << spx_h_dim << "," << spx_w_dim
                                                      << ")");
    return -1;
  }

  // ----------------------------
  // 3. Get element size
  // ----------------------------
  auto get_elem_size = [](hbDNNDataType type) -> int32_t {
    switch (type) {
    case HB_DNN_TENSOR_TYPE_F32: return 4;
    case HB_DNN_TENSOR_TYPE_S32: return 4;
    case HB_DNN_TENSOR_TYPE_S16: return 2;
    default: return 0;
    }
  };

  const int32_t disp_elem_size = get_elem_size(static_cast<hbDNNDataType>(tensors[0].properties.tensorType));
  const int32_t spx_elem_size = get_elem_size(static_cast<hbDNNDataType>(tensors[1].properties.tensorType));

  if (disp_elem_size == 0 || spx_elem_size == 0) {
    LOG_ERROR(logger_, "=> unsupported tensor element size");
    return -1;
  }

  // ----------------------------
  // 4. Get stride in element unit
  //    stride[] is in bytes
  // ----------------------------
  const int32_t *disp_stride = tensors[0].properties.stride;
  const int32_t *spx_stride = tensors[1].properties.stride;

  if (disp_stride == nullptr || spx_stride == nullptr) {
    LOG_ERROR(logger_, "=> tensor stride is null");
    return -1;
  }

  const int32_t disp_c_stride = disp_stride[1] / disp_elem_size;
  const int32_t disp_h_stride = disp_stride[2] / disp_elem_size;

  const int32_t spx_c_stride = spx_stride[1] / spx_elem_size;
  const int32_t spx_h_stride = spx_stride[2] / spx_elem_size;

  // ----------------------------
  // 5. Get quantization scales
  // ----------------------------
  float scale_constant = 1.0f;
  float *disp_scale = &scale_constant;
  float *spx_scale = &scale_constant;

  if (tensors[0].properties.quantiType == SCALE) {
    disp_scale = tensors[0].properties.scale.scaleData;
  }
  if (tensors[1].properties.quantiType == SCALE) {
    spx_scale = tensors[1].properties.scale.scaleData;
  }

  const bool disp_per_channel_scale =
      (tensors[0].properties.quantiType == SCALE && tensors[0].properties.quantizeAxis == 1 &&
       tensors[0].properties.scale.scaleLen >= c_dim);

  const bool spx_per_channel_scale =
      (tensors[1].properties.quantiType == SCALE && tensors[1].properties.quantizeAxis == 1 &&
       tensors[1].properties.scale.scaleLen >= c_dim);

  // ----------------------------
  // 6. Allocate output
  // ----------------------------
  out_mat = cv::Mat::zeros(h_dim, w_dim, CV_32FC1);
  float *out_ptr = reinterpret_cast<float *>(out_mat.data);

  // ----------------------------
  // 7. Helper lambda for per-channel scale
  // ----------------------------
  auto get_disp_scale = [&](int32_t c) -> float {
    if (tensors[0].properties.quantiType != SCALE) return 1.0f;
    return disp_per_channel_scale ? disp_scale[c] : disp_scale[0];
  };

  auto get_spx_scale = [&](int32_t c) -> float {
    if (tensors[1].properties.quantiType != SCALE) return 1.0f;
    return spx_per_channel_scale ? spx_scale[c] : spx_scale[0];
  };

  // ----------------------------
  // 8. Accumulate with stride-aware access
  // ----------------------------
  if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
      tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    auto disp_base = reinterpret_cast<const float *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx_base = reinterpret_cast<const float *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    for (int32_t c = 0; c < c_dim; ++c) {
      const float cur_scale = get_disp_scale(c) * get_spx_scale(c);
      const float *disp_c_ptr = disp_base + c * disp_c_stride;
      const float *spx_c_ptr = spx_base + c * spx_c_stride;

      for (int32_t y = 0; y < h_dim; ++y) {
        const float *disp_row = disp_c_ptr + y * disp_h_stride;
        const float *spx_row = spx_c_ptr + y * spx_h_stride;
        float *out_row = out_ptr + y * w_dim;

        for (int32_t x = 0; x < w_dim; ++x) {
          out_row[x] += disp_row[x] * spx_row[x] * cur_scale;
        }
      }
    }
  } else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
    auto disp_base = reinterpret_cast<const float *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx_base = reinterpret_cast<const int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    for (int32_t c = 0; c < c_dim; ++c) {
      const float cur_scale = get_disp_scale(c) * get_spx_scale(c);
      const float *disp_c_ptr = disp_base + c * disp_c_stride;
      const int16_t *spx_c_ptr = spx_base + c * spx_c_stride;

      for (int32_t y = 0; y < h_dim; ++y) {
        const float *disp_row = disp_c_ptr + y * disp_h_stride;
        const int16_t *spx_row = spx_c_ptr + y * spx_h_stride;
        float *out_row = out_ptr + y * w_dim;

        for (int32_t x = 0; x < w_dim; ++x) {
          out_row[x] += disp_row[x] * static_cast<float>(spx_row[x]) * cur_scale;
        }
      }
    }
  } else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S32 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
    auto disp_base = reinterpret_cast<const int32_t *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx_base = reinterpret_cast<const int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    for (int32_t c = 0; c < c_dim; ++c) {
      const float cur_scale = get_disp_scale(c) * get_spx_scale(c);
      const int32_t *disp_c_ptr = disp_base + c * disp_c_stride;
      const int16_t *spx_c_ptr = spx_base + c * spx_c_stride;

      for (int32_t y = 0; y < h_dim; ++y) {
        const int32_t *disp_row = disp_c_ptr + y * disp_h_stride;
        const int16_t *spx_row = spx_c_ptr + y * spx_h_stride;
        float *out_row = out_ptr + y * w_dim;

        for (int32_t x = 0; x < w_dim; ++x) {
          out_row[x] += static_cast<float>(disp_row[x]) * static_cast<float>(spx_row[x]) * cur_scale;
        }
      }
    }
  } else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S16 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
    auto disp_base = reinterpret_cast<const int16_t *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx_base = reinterpret_cast<const int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    for (int32_t c = 0; c < c_dim; ++c) {
      const float cur_scale = get_disp_scale(c) * get_spx_scale(c);
      const int16_t *disp_c_ptr = disp_base + c * disp_c_stride;
      const int16_t *spx_c_ptr = spx_base + c * spx_c_stride;

      for (int32_t y = 0; y < h_dim; ++y) {
        const int16_t *disp_row = disp_c_ptr + y * disp_h_stride;
        const int16_t *spx_row = spx_c_ptr + y * spx_h_stride;
        float *out_row = out_ptr + y * w_dim;

        for (int32_t x = 0; x < w_dim; ++x) {
          out_row[x] += static_cast<float>(disp_row[x]) * static_cast<float>(spx_row[x]) * cur_scale;
        }
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
*/

int StereonetProcess::postprocess_only_disp(const std::vector<hbDNNTensor> &tensors, cv::Mat &out_mat) {
  if (tensors.empty()) {
    LOG_ERROR(logger_, "=> postprocess_only_disp: tensors is empty.");
    return -1;
  }

  const hbDNNTensor &disp_tensor = tensors[0];
  const auto &properties = disp_tensor.properties;
  const auto &valid_shape = properties.validShape;

  // Support:
  // 3D: [N, H, W]
  // 4D: [N, C, H, W]
  if (valid_shape.numDimensions != 3 && valid_shape.numDimensions != 4) {
    LOG_ERROR(logger_, "=> invalid disp tensor dimension, expected 3 or 4 dims, got " << valid_shape.numDimensions);
    return -1;
  }

  const int32_t *shape = valid_shape.dimensionSize;

  int32_t n_dim = 0;
  int32_t c_dim = 1;
  int32_t h_dim = 0;
  int32_t w_dim = 0;

  int32_t h_stride_index = 0;
  int32_t w_stride_index = 0;

  if (valid_shape.numDimensions == 3) {
    // Layout: N H W
    n_dim = shape[0];
    h_dim = shape[1];
    w_dim = shape[2];

    h_stride_index = 1;
    w_stride_index = 2;
  } else {
    // Layout: N C H W
    n_dim = shape[0];
    c_dim = shape[1];
    h_dim = shape[2];
    w_dim = shape[3];

    h_stride_index = 2;
    w_stride_index = 3;
  }

  if (n_dim != 1 || c_dim != 1 || h_dim <= 0 || w_dim <= 0) {
    LOG_ERROR(logger_,
              "=> invalid disp tensor shape, n=" << n_dim << ", c=" << c_dim << ", h=" << h_dim << ", w=" << w_dim);
    return -1;
  }

  // Get stride information.
  const auto *stride = properties.alignedByteSize > 0 ? properties.stride : nullptr;

  if (stride == nullptr) {
    LOG_ERROR(logger_, "=> disp tensor stride is null.");
    return -1;
  }

  const void *vir_addr = TENSOR_SYSMEM(disp_tensor, 0).virAddr;
  if (vir_addr == nullptr) {
    LOG_ERROR(logger_, "=> disp tensor virAddr is null.");
    return -1;
  }

  // Read the quantization scale.
  float scale = 1.0f;

  if (properties.quantiType == SCALE) {
    if (properties.scale.scaleData == nullptr || properties.scale.scaleLen <= 0) {
      LOG_ERROR(logger_, "=> disp tensor quantiType is SCALE, but scale data is invalid.");
      return -1;
    }

    // Single-channel output uses the first scale value.
    scale = properties.scale.scaleData[0];
  }

  out_mat = cv::Mat::zeros(h_dim, w_dim, CV_32FC1);

  switch (properties.tensorType) {
  case HB_DNN_TENSOR_TYPE_F32: {
    constexpr int32_t elem_size = sizeof(float);

    if (stride[h_stride_index] % elem_size != 0 || stride[w_stride_index] % elem_size != 0) {
      LOG_ERROR(logger_, "=> invalid F32 stride, h_stride_bytes=" << stride[h_stride_index]
                                                                  << ", w_stride_bytes=" << stride[w_stride_index]);
      return -1;
    }

    const int64_t h_stride = static_cast<int64_t>(stride[h_stride_index]) / elem_size;
    const int64_t w_stride = static_cast<int64_t>(stride[w_stride_index]) / elem_size;

    const auto *src = reinterpret_cast<const float *>(vir_addr);

    for (int32_t y = 0; y < h_dim; ++y) {
      const float *src_row = src + y * h_stride;
      float *dst_row = out_mat.ptr<float>(y);

      if (w_stride == 1 && std::abs(scale - 1.0f) < 1e-8f) {
        std::memcpy(dst_row, src_row, static_cast<size_t>(w_dim) * sizeof(float));
      } else {
        for (int32_t x = 0; x < w_dim; ++x) {
          dst_row[x] = src_row[x * w_stride] * scale;
        }
      }
    }

    break;
  }

  case HB_DNN_TENSOR_TYPE_S32: {
    constexpr int32_t elem_size = sizeof(int32_t);

    if (stride[h_stride_index] % elem_size != 0 || stride[w_stride_index] % elem_size != 0) {
      LOG_ERROR(logger_, "=> invalid S32 stride, h_stride_bytes=" << stride[h_stride_index]
                                                                  << ", w_stride_bytes=" << stride[w_stride_index]);
      return -1;
    }

    const int64_t h_stride = static_cast<int64_t>(stride[h_stride_index]) / elem_size;
    const int64_t w_stride = static_cast<int64_t>(stride[w_stride_index]) / elem_size;

    const auto *src = reinterpret_cast<const int32_t *>(vir_addr);

    for (int32_t y = 0; y < h_dim; ++y) {
      const int32_t *src_row = src + y * h_stride;
      float *dst_row = out_mat.ptr<float>(y);

      for (int32_t x = 0; x < w_dim; ++x) {
        dst_row[x] = static_cast<float>(src_row[x * w_stride]) * scale;
      }
    }

    break;
  }

  case HB_DNN_TENSOR_TYPE_S16: {
    constexpr int32_t elem_size = sizeof(int16_t);

    if (stride[h_stride_index] % elem_size != 0 || stride[w_stride_index] % elem_size != 0) {
      LOG_ERROR(logger_, "=> invalid S16 stride, h_stride_bytes=" << stride[h_stride_index]
                                                                  << ", w_stride_bytes=" << stride[w_stride_index]);
      return -1;
    }

    const int64_t h_stride = static_cast<int64_t>(stride[h_stride_index]) / elem_size;
    const int64_t w_stride = static_cast<int64_t>(stride[w_stride_index]) / elem_size;

    const auto *src = reinterpret_cast<const int16_t *>(vir_addr);

    for (int32_t y = 0; y < h_dim; ++y) {
      const int16_t *src_row = src + y * h_stride;
      float *dst_row = out_mat.ptr<float>(y);

      for (int32_t x = 0; x < w_dim; ++x) {
        dst_row[x] = static_cast<float>(src_row[x * w_stride]) * scale;
      }
    }

    break;
  }

  default:
    LOG_ERROR(logger_, "=> unsupported disp tensor type: "
                           << magic_enum::enum_name(static_cast<hbDNNDataType>(properties.tensorType))
                           << ", only support F32, S32 and S16.");
    return -1;
  }

  return 0;
}

int StereonetProcess::postprocess_convex_upsampling_with_interp(const std::vector<hbDNNTensor> &tensors,
                                                                cv::Mat &out_mat) {
  // ----------------------------
  // 1. Use valid shape for logical output size
  // ----------------------------
  const int32_t *disp_valid_shape = tensors[0].properties.validShape.dimensionSize;
  const int32_t *spx_valid_shape = tensors[1].properties.validShape.dimensionSize;

  const int32_t disp_c_dim = disp_valid_shape[1];
  const int32_t disp_h_dim = disp_valid_shape[2];
  const int32_t disp_w_dim = disp_valid_shape[3];

  const int32_t spx_c_dim = spx_valid_shape[1];
  const int32_t spx_h_dim = spx_valid_shape[2];
  const int32_t spx_w_dim = spx_valid_shape[3];

  if (disp_c_dim != spx_c_dim) {
    LOG_ERROR(logger_, "=> disp/spx channel mismatch, disp_c=" << disp_c_dim << ", spx_c=" << spx_c_dim);
    return -1;
  }

  if (disp_h_dim <= 0 || disp_w_dim <= 0 || spx_h_dim <= 0 || spx_w_dim <= 0) {
    LOG_ERROR(logger_, "=> invalid tensor shape.");
    return -1;
  }

  const int32_t scale_h = spx_h_dim / disp_h_dim;
  const int32_t scale_w = spx_w_dim / disp_w_dim;
  if (scale_h <= 0 || scale_w <= 0) {
    LOG_ERROR(logger_, "=> invalid upsample scale, scale_h=" << scale_h << ", scale_w=" << scale_w);
    return -1;
  }

  // ----------------------------
  // 2. Use stride for real memory layout
  //    stride is in bytes.
  //    Use auto to support both int32_t* and int64_t* SDK definitions.
  // ----------------------------
  auto disp_stride = tensors[0].properties.alignedByteSize ? tensors[0].properties.stride : nullptr;
  auto spx_stride = tensors[1].properties.alignedByteSize ? tensors[1].properties.stride : nullptr;

  if (disp_stride == nullptr || spx_stride == nullptr) {
    LOG_ERROR(logger_, "=> tensor stride is null.");
    return -1;
  }

  // element size in bytes
  int disp_elem_size = 0;
  int spx_elem_size = 0;

  switch (tensors[0].properties.tensorType) {
  case HB_DNN_TENSOR_TYPE_S32: disp_elem_size = 4; break;
  case HB_DNN_TENSOR_TYPE_F32: disp_elem_size = 4; break;
  default: LOG_ERROR(logger_, "=> unsupported disp tensor type."); return -1;
  }

  switch (tensors[1].properties.tensorType) {
  case HB_DNN_TENSOR_TYPE_S16: spx_elem_size = 2; break;
  case HB_DNN_TENSOR_TYPE_F32: spx_elem_size = 4; break;
  default: LOG_ERROR(logger_, "=> unsupported spx tensor type."); return -1;
  }

  // stride[1]: bytes per channel
  // stride[2]: bytes per row
  const int64_t disp_c_stride = static_cast<int64_t>(disp_stride[1]) / disp_elem_size;
  const int64_t disp_h_stride = static_cast<int64_t>(disp_stride[2]) / disp_elem_size;
  const int64_t spx_c_stride = static_cast<int64_t>(spx_stride[1]) / spx_elem_size;
  const int64_t spx_h_stride = static_cast<int64_t>(spx_stride[2]) / spx_elem_size;

  // ----------------------------
  // 3. Get quant scales
  // ----------------------------
  float scale_constant = 1.0f;
  float *disp_scale = &scale_constant;
  float *spx_scale = &scale_constant;

  if (tensors[0].properties.quantiType == SCALE) {
    disp_scale = tensors[0].properties.scale.scaleData;
  }
  if (tensors[1].properties.quantiType == SCALE) {
    spx_scale = tensors[1].properties.scale.scaleData;
  }

  // For disp output, quantizeAxis = 1, so each channel may have its own scale.
  // For spx, usually one shared scale is enough in your model.
  const bool disp_per_channel_scale =
      (tensors[0].properties.quantiType == SCALE && tensors[0].properties.scale.scaleLen >= disp_c_dim &&
       tensors[0].properties.quantizeAxis == 1);

  const float spx_scale_val = *spx_scale;

  // ----------------------------
  // 4. Allocate output with valid size
  // ----------------------------
  out_mat = cv::Mat::zeros(spx_h_dim, spx_w_dim, CV_32FC1);
  float *result_ptr = reinterpret_cast<float *>(out_mat.data);

  // ----------------------------
  // 5. Read memory using stride, not valid width
  // ----------------------------
  if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S32 &&
      tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
    auto disp_base = reinterpret_cast<int32_t *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx_base = reinterpret_cast<int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    for (int32_t c = 0; c < spx_c_dim; ++c) {
      const float cur_disp_scale = disp_per_channel_scale ? disp_scale[c] : (*disp_scale);
      const float cur_scale = cur_disp_scale * spx_scale_val;

      const int32_t *disp_c_ptr = disp_base + c * disp_c_stride;
      const int16_t *spx_c_ptr = spx_base + c * spx_c_stride;

      for (int32_t y = 0; y < spx_h_dim; ++y) {
        const int32_t idx_y = std::min(y / scale_h, disp_h_dim - 1);
        float *out_row = result_ptr + y * spx_w_dim;
        const int16_t *spx_row = spx_c_ptr + y * spx_h_stride;
        const int32_t *disp_row = disp_c_ptr + idx_y * disp_h_stride;

        int32_t x = 0;
        for (; x <= spx_w_dim - 4; x += 4) {
          const int32_t idx_x0 = std::min((x + 0) / scale_w, disp_w_dim - 1);
          const int32_t idx_x1 = std::min((x + 1) / scale_w, disp_w_dim - 1);
          const int32_t idx_x2 = std::min((x + 2) / scale_w, disp_w_dim - 1);
          const int32_t idx_x3 = std::min((x + 3) / scale_w, disp_w_dim - 1);

          int32x4_t disp_s32 = {disp_row[idx_x0], disp_row[idx_x1], disp_row[idx_x2], disp_row[idx_x3]};
          int16x4_t spx_s16 = vld1_s16(spx_row + x);
          int32x4_t spx_s32 = vmovl_s16(spx_s16);

          float32x4_t disp_f32 = vcvtq_f32_s32(disp_s32);
          float32x4_t spx_f32 = vcvtq_f32_s32(spx_s32);
          float32x4_t mul_f32 = vmulq_n_f32(vmulq_f32(disp_f32, spx_f32), cur_scale);

          float32x4_t out_f32 = vld1q_f32(out_row + x);
          out_f32 = vaddq_f32(out_f32, mul_f32);
          vst1q_f32(out_row + x, out_f32);
        }
        for (; x < spx_w_dim; ++x) {
          const int32_t idx_x = std::min(x / scale_w, disp_w_dim - 1);
          out_row[x] += static_cast<float>(disp_row[idx_x]) * static_cast<float>(spx_row[x]) * cur_scale;
        }
      }
    }
  } else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    auto disp_base = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    auto spx_base = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    for (int32_t c = 0; c < spx_c_dim; ++c) {
      const float cur_disp_scale = disp_per_channel_scale ? disp_scale[c] : (*disp_scale);
      const float cur_scale = cur_disp_scale * spx_scale_val;

      const float *disp_c_ptr = disp_base + c * disp_c_stride;
      const float *spx_c_ptr = spx_base + c * spx_c_stride;

      for (int32_t y = 0; y < spx_h_dim; ++y) {
        const int32_t idx_y = std::min(y / scale_h, disp_h_dim - 1);
        float *out_row = result_ptr + y * spx_w_dim;
        const float *spx_row = spx_c_ptr + y * spx_h_stride;
        const float *disp_row = disp_c_ptr + idx_y * disp_h_stride;

        int32_t x = 0;
        for (; x <= spx_w_dim - 4; x += 4) {
          const int32_t idx_x0 = std::min((x + 0) / scale_w, disp_w_dim - 1);
          const int32_t idx_x1 = std::min((x + 1) / scale_w, disp_w_dim - 1);
          const int32_t idx_x2 = std::min((x + 2) / scale_w, disp_w_dim - 1);
          const int32_t idx_x3 = std::min((x + 3) / scale_w, disp_w_dim - 1);

          float32x4_t disp_f32 = {disp_row[idx_x0], disp_row[idx_x1], disp_row[idx_x2], disp_row[idx_x3]};
          float32x4_t spx_f32 = vld1q_f32(spx_row + x);
          float32x4_t mul_f32 = vmulq_n_f32(vmulq_f32(disp_f32, spx_f32), cur_scale);

          float32x4_t out_f32 = vld1q_f32(out_row + x);
          out_f32 = vaddq_f32(out_f32, mul_f32);
          vst1q_f32(out_row + x, out_f32);
        }
        for (; x < spx_w_dim; ++x) {
          const int32_t idx_x = std::min(x / scale_w, disp_w_dim - 1);
          out_row[x] += disp_row[idx_x] * spx_row[x] * cur_scale;
        }
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

int StereonetProcess::postprocess_convex_upsampling_2x_logits(const std::vector<hbDNNTensor> &tensors,
                                                              cv::Mat &out_mat) {
  if (tensors.size() < 2) {
    LOG_ERROR(logger_, "=> postprocess_convex_upsampling_2x_logits: tensors.size() < 2");
    return -1;
  }

  const hbDNNTensor &unfold_tensor = tensors[0];
  const hbDNNTensor &mask_tensor = tensors[1];

  const auto &unfold_prop = unfold_tensor.properties;
  const auto &mask_prop = mask_tensor.properties;

  // ------------------------------------------------------------
  // 1. Check tensor shape
  //
  // unfold_info : [1, 9,  H, W]
  // mask_logits : [1, 36, H, W]
  //
  // 36 = 9 * 2 * 2
  // ------------------------------------------------------------
  if (unfold_prop.validShape.numDimensions != 4 || mask_prop.validShape.numDimensions != 4) {
    LOG_ERROR(logger_, "=> convex upsample expects 4D NCHW tensors");
    return -1;
  }

  const int32_t *unfold_shape = unfold_prop.validShape.dimensionSize;
  const int32_t *mask_shape = mask_prop.validShape.dimensionSize;

  const int32_t n = unfold_shape[0];
  const int32_t unfold_c = unfold_shape[1];
  const int32_t h = unfold_shape[2];
  const int32_t w = unfold_shape[3];

  const int32_t mask_n = mask_shape[0];
  const int32_t mask_c = mask_shape[1];
  const int32_t mask_h = mask_shape[2];
  const int32_t mask_w = mask_shape[3];

  constexpr int32_t kKernel = 9;
  constexpr int32_t kUpsample = 2;
  constexpr int32_t kSubPixels = kUpsample * kUpsample; // 4

  if (n != 1 || mask_n != 1 || unfold_c != kKernel || mask_c != kKernel * kSubPixels || h != mask_h || w != mask_w) {
    LOG_ERROR(logger_, "=> invalid convex upsample shape, "
                       "unfold=["
                           << n << "," << unfold_c << "," << h << "," << w << "], mask=[" << mask_n << "," << mask_c
                           << "," << mask_h << "," << mask_w << "]");
    return -1;
  }

  // ------------------------------------------------------------
  // 2. Current model outputs F32
  // ------------------------------------------------------------
  if (unfold_prop.tensorType != HB_DNN_TENSOR_TYPE_F32 || mask_prop.tensorType != HB_DNN_TENSOR_TYPE_F32) {
    LOG_ERROR(logger_, "=> convex upsample currently only supports F32, "
                       "unfold="
                           << magic_enum::enum_name(static_cast<hbDNNDataType>(unfold_prop.tensorType))
                           << ", mask=" << magic_enum::enum_name(static_cast<hbDNNDataType>(mask_prop.tensorType)));
    return -1;
  }

  // ------------------------------------------------------------
  // 3. Get tensor memory
  // ------------------------------------------------------------
  const auto *unfold_base = reinterpret_cast<const float *>(TENSOR_SYSMEM(unfold_tensor, 0).virAddr);

  const auto *mask_base = reinterpret_cast<const float *>(TENSOR_SYSMEM(mask_tensor, 0).virAddr);

  if (unfold_base == nullptr || mask_base == nullptr) {
    LOG_ERROR(logger_, "=> convex upsample tensor virAddr is null");
    return -1;
  }

  // ------------------------------------------------------------
  // 4. Read stride
  //
  // stride[] unit is byte.
  //
  // Example:
  //
  // unfold:
  //   stride[1] = 184320 bytes/channel
  //   stride[2] = 1152   bytes/row
  //   stride[3] = 4      bytes/pixel
  //
  // ------------------------------------------------------------
  const auto *unfold_stride = unfold_prop.stride;
  const auto *mask_stride = mask_prop.stride;

  if (unfold_stride == nullptr || mask_stride == nullptr) {
    LOG_ERROR(logger_, "=> convex upsample tensor stride is null");
    return -1;
  }

  constexpr int32_t elem_size = sizeof(float);

  if (unfold_stride[1] % elem_size != 0 || unfold_stride[2] % elem_size != 0 || unfold_stride[3] % elem_size != 0 ||
      mask_stride[1] % elem_size != 0 || mask_stride[2] % elem_size != 0 || mask_stride[3] % elem_size != 0) {
    LOG_ERROR(logger_, "=> invalid F32 tensor stride");
    return -1;
  }

  const int64_t unfold_c_stride = static_cast<int64_t>(unfold_stride[1]) / elem_size;
  const int64_t unfold_h_stride = static_cast<int64_t>(unfold_stride[2]) / elem_size;
  const int64_t unfold_w_stride = static_cast<int64_t>(unfold_stride[3]) / elem_size;

  const int64_t mask_c_stride = static_cast<int64_t>(mask_stride[1]) / elem_size;
  const int64_t mask_h_stride = static_cast<int64_t>(mask_stride[2]) / elem_size;
  const int64_t mask_w_stride = static_cast<int64_t>(mask_stride[3]) / elem_size;

  // ------------------------------------------------------------
  // 5. Output:
  //
  // [H, W] -> [2H, 2W]
  // 160x288 -> 320x576
  // ------------------------------------------------------------
  const int32_t out_h = h * kUpsample;
  const int32_t out_w = w * kUpsample;

  out_mat = cv::Mat::zeros(out_h, out_w, CV_32FC1);

  // ------------------------------------------------------------
  // 6. Convex upsampling
  //
  // Python:
  //
  // mask =
  //   mask_logits.reshape(N, 1, 9, 2, 2, H, W)
  //
  // Therefore original mask channel mapping:
  //
  // channel = k * 4 + dy * 2 + dx
  //
  // k  : 0~8
  // dy : 0~1
  // dx : 0~1
  //
  // For every low-resolution pixel:
  //
  //   generate:
  //
  //   (2y,   2x)
  //   (2y,   2x+1)
  //   (2y+1, 2x)
  //   (2y+1, 2x+1)
  //
  // Each output pixel:
  //
  // disp =
  //   sum_k softmax(mask_logits[k]) * unfold_info[k]
  //
  // ------------------------------------------------------------
  for (int32_t y = 0; y < h; ++y) {
    for (int32_t x = 0; x < w; ++x) {

      // One low-resolution pixel generates 2x2 output pixels.
      for (int32_t dy = 0; dy < kUpsample; ++dy) {
        float *out_row = out_mat.ptr<float>(y * kUpsample + dy);

        for (int32_t dx = 0; dx < kUpsample; ++dx) {

          // ----------------------------------------------------
          // Step 1:
          // Find max logit for numerically stable softmax.
          //
          // Python:
          //
          // x = x - x.max(axis=2)
          // ----------------------------------------------------
          float max_logit = -std::numeric_limits<float>::infinity();

          for (int32_t k = 0; k < kKernel; ++k) {
            const int32_t mask_channel = k * kSubPixels + dy * kUpsample + dx;

            const float *mask_c_ptr = mask_base + static_cast<int64_t>(mask_channel) * mask_c_stride;

            const float logit =
                mask_c_ptr[static_cast<int64_t>(y) * mask_h_stride + static_cast<int64_t>(x) * mask_w_stride];

            max_logit = std::max(max_logit, logit);
          }

          // ----------------------------------------------------
          // Step 2:
          // Calculate softmax denominator.
          // ----------------------------------------------------
          float softmax_sum = 0.0f;

          float exp_logits[kKernel];

          for (int32_t k = 0; k < kKernel; ++k) {
            const int32_t mask_channel = k * kSubPixels + dy * kUpsample + dx;

            const float *mask_c_ptr = mask_base + static_cast<int64_t>(mask_channel) * mask_c_stride;

            const float logit =
                mask_c_ptr[static_cast<int64_t>(y) * mask_h_stride + static_cast<int64_t>(x) * mask_w_stride];

            const float e = std::exp(logit - max_logit);

            exp_logits[k] = e;
            softmax_sum += e;
          }

          if (softmax_sum <= 0.0f) {
            out_row[x * kUpsample + dx] = 0.0f;
            continue;
          }

          // ----------------------------------------------------
          // Step 3:
          //
          // weighted sum:
          //
          // disp = sum(weight[k] * unfold[k])
          // ----------------------------------------------------
          float value = 0.0f;

          const float inv_softmax_sum = 1.0f / softmax_sum;

          for (int32_t k = 0; k < kKernel; ++k) {
            const float weight = exp_logits[k] * inv_softmax_sum;

            const float *unfold_c_ptr = unfold_base + static_cast<int64_t>(k) * unfold_c_stride;

            const float disp_value =
                unfold_c_ptr[static_cast<int64_t>(y) * unfold_h_stride + static_cast<int64_t>(x) * unfold_w_stride];

            value += weight * disp_value;
          }

          // ----------------------------------------------------
          // Equivalent to Python:
          //
          // transpose + reshape
          // ----------------------------------------------------
          out_row[x * kUpsample + dx] = value;
        }
      }
    }
  }

  return 0;
}

int StereonetProcess::prepare_input_tensor(std::vector<hbDNNTensor> &input_tensors) {
  static bool prt_flag = true;
  int ret_code = 0;
  if (prt_flag) LOG_WARN(logger_, "=> ----- prepare_input_tensor -----");

  // allocate memory for input tensor
  input_tensors.resize(input_count_);
  for (int i = 0; i < input_count_; i++) {
    auto &tensor = input_tensors[i];
    // get input tensor properties
    hbDNNTensorProperties properties;
    ret_code = hbDNNGetInputTensorProperties(&properties, dnn_handle_, i);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetInputTensorProperties failed");
    if (prt_flag)
      LOG_WARN(logger_,
               "=> input tensor type is " << magic_enum::enum_name(static_cast<hbDNNDataType>(properties.tensorType)));
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

#ifdef PLATFORM_S100
    // properties.quantizeAxis = 3;
    // MARCH: "nash-e" (J5/J6E) need to set stride to align with 32-byte for nv12 input
    auto dim_len = properties.validShape.numDimensions;
    for (int32_t dim_i = dim_len - 1; dim_i >= 0; --dim_i) {
      if (properties.stride[dim_i] == -1) {
        auto cur_stride = properties.stride[dim_i + 1] * properties.validShape.dimensionSize[dim_i + 1];
        properties.stride[dim_i] = ALIGN_32(cur_stride);
      }
    }
    properties.alignedByteSize = properties.stride[0];
#endif

#ifdef PLATFORM_S600
    // properties.quantizeAxis = 3;
    // "nash-p" (J6P/H) need to set stride to align with 64-byte for nv12 input
    auto dim_len = properties.validShape.numDimensions;
    for (int32_t dim_i = dim_len - 1; dim_i >= 0; --dim_i) {
      if (properties.stride[dim_i] == -1) {
        auto cur_stride = properties.stride[dim_i + 1] * properties.validShape.dimensionSize[dim_i + 1];
        properties.stride[dim_i] = ALIGN_64(cur_stride);
      }
    }
    properties.alignedByteSize = properties.stride[0];
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
      if (prt_flag) LOG_WARN(logger_, "=> input[" << i << "].memsize: " << tensor.sysMem[0].memSize);
    } else if (properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
      ret_code = hbSysAllocCachedMem(&tensor.sysMem[0], model_input_h_ * model_input_w_);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      tensor.sysMem[0].memSize = model_input_h_ * model_input_w_;

      ret_code = hbSysAllocCachedMem(&tensor.sysMem[1], model_input_h_ * model_input_w_ / 2);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      tensor.sysMem[1].memSize = model_input_h_ * model_input_w_ / 2;
      if (prt_flag) LOG_WARN(logger_, "=> input[" << i << "].memsize[0]: " << tensor.sysMem[0].memSize);
      if (prt_flag) LOG_WARN(logger_, "=> input[" << i << "].memsize[1]: " << tensor.sysMem[1].memSize);
    } else {
      return -1;
    }
#endif

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)
    if (properties.tensorType == HB_DNN_TENSOR_TYPE_U8) {
      ret_code = hbSysAllocCachedMem(&tensor.sysMem, properties.alignedByteSize);
      HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
      if (prt_flag) LOG_WARN(logger_, "=> input tensor size: " << tensor.sysMem.memSize);
    } else {
      return -1;
    }
#endif
    if (prt_flag) LOG_WARN(logger_, "=> ----------------");
  }
  prt_flag = false;
  return ret_code;
}

int StereonetProcess::prepare_output_tensor(std::vector<hbDNNTensor> &output_tensors) {
  static bool prt_flag = true;
  int ret_code = 0;
  LOG_WARN_ONCE(logger_, "=> ----- prepare_output_tensor -----");
  output_tensors.resize(output_count_);
  for (int i = 0; i < output_count_; ++i) {
    ret_code = hbDNNGetOutputTensorProperties(&output_tensors[i].properties, dnn_handle_, i);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbDNNGetOutputTensorProperties failed");
    if (prt_flag)
      LOG_WARN(logger_, "=> output tensor type is " << magic_enum::enum_name(
                            static_cast<hbDNNDataType>(output_tensors[i].properties.tensorType)));
    ret_code = hbSysAllocCachedMem(&TENSOR_SYSMEM(output_tensors[i], 0), output_tensors[i].properties.alignedByteSize);
    HB_CHECK_SUCCESS(logger_, ret_code, "hbSysAllocCachedMem failed");
    if (prt_flag) LOG_WARN(logger_, "=> output[" << i << "].memsize: " << output_tensors[i].properties.alignedByteSize);
  }
  prt_flag = false;
  return ret_code;
}

int StereonetProcess::get_idle_tensor() {
  for (int i = 0; i < max_memory_count_; ++i) {
    bool expected = true;
    if (idle_tensor_[i].compare_exchange_strong(expected, false, std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
      return i;
    }
  }

  return -1;
}

void StereonetProcess::set_tensor_idle(const InferenceHandle &tensor_id) {
  if (tensor_id >= 0 && tensor_id < max_memory_count_) {
    idle_tensor_[tensor_id].store(true, std::memory_order_release);
  }
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

void StereonetProcess::perspective_disparity_to_depth(const cv::Mat &disp, cv::Mat &depth,
                                                      const CameraIntrinsic &camera_intrinsic) {
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

void StereonetProcess::longlati_disparity_to_depth(const cv::Mat &disp, cv::Mat &depth,
                                                   const CameraIntrinsic &camera_intrinsic) {
  depth.create(disp.size(), CV_16UC1);
  const int rows = disp.rows;
  const int cols = disp.cols;

  // Spherical (longitude-latitude) stereo triangulation, matching the Python
  // longlati reconstruction:
  //   pi_w = PI / (W-1)               // radians per pixel of longitude (= 1/fx)
  //   diff = pi_w * disparity         // angular disparity (rad)
  //   tt   = (col/(W-1) - 0.5) * PI   // longitude of the pixel (rad)
  //   mgnt = baseline * sin(col*pi_w - diff) / sin(diff)   // radial distance (m)
  // The scalar depth stored here is mgnt converted to millimeters.
  const double bl = camera_intrinsic.baseline; // meters
  const double pi_w = (cols > 1) ? (M_PI / (cols - 1)) : 0.0;

  for (int i = 0; i < rows; ++i) {
    const float *disp_ptr = disp.ptr<float>(i);
    uint16_t *depth_ptr = depth.ptr<uint16_t>(i);
    for (int j = 0; j < cols; ++j) {
      const float d = disp_ptr[j];
      if (d <= 0.0f) {
        depth_ptr[j] = 0;
        continue;
      }
      const double diff = pi_w * d;
      const double sin_diff = std::sin(diff);
      if (sin_diff <= 1e-12) {
        // angular disparity too small (very far) or invalid -> saturate
        depth_ptr[j] = 65535;
        continue;
      }
      const double col_angle = static_cast<double>(j) * pi_w;         // = j*PI/(cols-1)
      const double mgnt = bl * std::sin(col_angle - diff) / sin_diff; // meters
      if (!std::isfinite(mgnt) || mgnt <= 0.0) {
        depth_ptr[j] = 0;
        continue;
      }
      double mm = mgnt * 1000.0;
      if (mm > 65535.0) mm = 65535.0;
      depth_ptr[j] = static_cast<uint16_t>(mm);
    }
  }
}

void StereonetProcess::disparity_to_depth(const cv::Mat &disp, cv::Mat &depth,
                                          const CameraIntrinsic &camera_intrinsic) {
  if (camera_intrinsic.rectify_model == "RECTIFY_LONGLATI") {
    longlati_disparity_to_depth(disp, depth, camera_intrinsic);
  } else {
    perspective_disparity_to_depth(disp, depth, camera_intrinsic);
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

// speckle filter
template <typename T>
static void apply_speckle_filter_impl(cv::Mat &img, int speckle_size, double speckle_diff, int connectivity) {
  const int rows = img.rows;
  const int cols = img.cols;

  cv::Mat labels(rows, cols, CV_32S, cv::Scalar(-1));
  int current_label = 0;

  const int dx4[4] = {1, -1, 0, 0};
  const int dy4[4] = {0, 0, 1, -1};

  const int dx8[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  const int dy8[8] = {0, 0, 1, -1, 1, -1, 1, -1};

  std::vector<cv::Point> component_pixels;
  std::queue<cv::Point> q;

  auto is_valid_value = [](T v) -> bool {
    if constexpr (std::is_floating_point<T>::value) {
      return std::isfinite(v) && v > static_cast<T>(0);
    } else {
      return v > static_cast<T>(0);
    }
  };

  auto value_diff_ok = [speckle_diff](T a, T b) -> bool {
    double da = static_cast<double>(a);
    double db = static_cast<double>(b);
    return std::fabs(da - db) <= speckle_diff;
  };

  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      if (labels.at<int>(y, x) != -1) {
        continue;
      }

      T seed_val = img.at<T>(y, x);
      if (!is_valid_value(seed_val)) {
        labels.at<int>(y, x) = -2; // invalid
        continue;
      }

      component_pixels.clear();
      labels.at<int>(y, x) = current_label;
      q.push(cv::Point(x, y));
      component_pixels.push_back(cv::Point(x, y));

      while (!q.empty()) {
        cv::Point p = q.front();
        q.pop();

        const T cur_val = img.at<T>(p.y, p.x);

        const int *dx = (connectivity == 4) ? dx4 : dx8;
        const int *dy = (connectivity == 4) ? dy4 : dy8;
        const int neighbor_count = (connectivity == 4) ? 4 : 8;

        for (int k = 0; k < neighbor_count; ++k) {
          int nx = p.x + dx[k];
          int ny = p.y + dy[k];

          if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) {
            continue;
          }

          int &nlabel = labels.at<int>(ny, nx);
          if (nlabel != -1) {
            continue;
          }

          T nval = img.at<T>(ny, nx);
          if (!is_valid_value(nval)) {
            nlabel = -2;
            continue;
          }

          if (value_diff_ok(nval, cur_val)) {
            nlabel = current_label;
            q.push(cv::Point(nx, ny));
            component_pixels.push_back(cv::Point(nx, ny));
          }
        }
      }

      if (static_cast<int>(component_pixels.size()) <= speckle_size) {
        for (const auto &pt : component_pixels) {
          img.at<T>(pt.y, pt.x) = static_cast<T>(0);
        }
      }

      ++current_label;
    }
  }
}

/**
 * @brief speckle filter for disparity map or depth map
 * @param img
 * @param speckle_size
 * @param speckle_diff
 * @param connectivity
 */
static void apply_speckle_filter(cv::Mat &img, int speckle_size = 100, double speckle_diff = 2.0,
                                 int connectivity = 8) {
  if (img.empty()) {
    return;
  }

  if (img.channels() != 1) {
    throw std::runtime_error("=> apply_speckle_filter only supports single-channel images");
  }

  if (connectivity != 4 && connectivity != 8) {
    throw std::runtime_error("=> connectivity must be 4 or 8");
  }

  if (img.type() == CV_32FC1) {
    apply_speckle_filter_impl<float>(img, speckle_size, speckle_diff, connectivity);
  } else if (img.type() == CV_16UC1) {
    apply_speckle_filter_impl<uint16_t>(img, speckle_size, speckle_diff, connectivity);
  } else {
    throw std::runtime_error("=> apply_speckle_filter only supports CV_32FC1 and CV_16UC1");
  }
}

cv::Mat StereonetProcess::render_disp_or_depth(const cv::Mat &input, float min_disp, float max_disp, float min_depth,
                                               float max_depth, bool enable_speckle_filter, int speckle_size,
                                               double speckle_diff, int speckle_connectivity) {
  if (input.empty()) {
    throw std::runtime_error("=> input image is empty");
  }

  bool is_disp = (input.type() == CV_32FC1);
  bool is_depth = (input.type() == CV_16UC1);

  if (!is_disp && !is_depth) {
    throw std::runtime_error("=> unsupported input type, only CV_32FC1 or CV_16UC1");
  }

  cv::Mat filtered = input.clone();
  if (enable_speckle_filter) {
    apply_speckle_filter(filtered, speckle_size, speckle_diff, speckle_connectivity);
  }

  // convert to float
  cv::Mat img_f;
  filtered.convertTo(img_f, CV_32F);

  // normalize
  for (int y = 0; y < img_f.rows; ++y) {
    float *row = img_f.ptr<float>(y);
    for (int x = 0; x < img_f.cols; ++x) {
      float &v = row[x];
      if (is_disp) {
        if (v < min_disp) v = 0.0f;
        if (v > max_disp) v = max_disp;
      } else if (is_depth) {
        if (v < min_depth) v = 0.0f;
        if (v > max_depth) v = max_depth;
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
