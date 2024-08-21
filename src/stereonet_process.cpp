//
// Created by zhy on 7/1/24.
//

#include <arm_neon.h>
#include <fstream>
#include <cassert>
#include <vector>

#include "stereonet_process.h"
#include "image_conversion.h"

static void Dequantize(float *output,
                       int32_t *input,
                       float *input_scale,
                       int32_t *input_shape,
                       int32_t *input_aligned_shape) {
  int32_t channel = input_shape[1];
  int32_t height = input_shape[2];
  // Here width is a multiple of 4, and neon can be used to accelerate
  // calculations
  int32_t width = input_shape[3];

  for (int32_t c = 0; c < channel; c++) {
    float32x4_t scale = vdupq_n_f32(input_scale[c]);
    for (int32_t h = 0; h < height; h++) {
      for (int32_t w = 0; w < width; w += 4) {
        int32x4_t input_data_tmp =
            vld1q_s32(&input[h * input_aligned_shape[3] + w]);
        float32x4_t input_data = vcvtq_f32_s32(input_data_tmp);
        float32x4_t ouput_data = vmulq_f32(input_data, scale);
        vst1q_f32(&output[h * width + w], ouput_data);
      }
    }
    input += input_aligned_shape[2] * input_aligned_shape[3];
    output += height * width;
  }
}

static void Dequantize16(float *output,
                         int16_t *input,
                         float *input_scale,
                         int32_t *input_shape,
                         int32_t *input_aligned_shape) {
  int32_t channel = input_shape[1];
  int32_t height = input_shape[2];
  // Here width is a multiple of 4, and neon can be used to accelerate
  // calculations
  int32_t width = input_shape[3];

  for (int32_t c = 0; c < channel; c++) {
    float scale = input_scale[c];
    for (int32_t h = 0; h < height; h++) {
      for (int32_t w = 0; w < width; w ++) {
        int16_t input_data_tmp = input[h * input_aligned_shape[3] + w];
        output[h * width + w] = (float)input_data_tmp * scale;
      }
    }
    input += input_aligned_shape[2] * input_aligned_shape[3];
    output += height * width;
  }
}

static void Dequantize16_neon(float *output,
                         int16_t *input,
                         float *input_scale,
                         int32_t *input_shape,
                         int32_t *input_aligned_shape) {
  int32_t channel = input_shape[1];
  int32_t height = input_shape[2];
  int32_t width = input_shape[3];
  for (int32_t c = 0; c < channel; c++) {
    float scale = input_scale[c];
    float32x4_t scale_neon = vdupq_n_f32(scale);
    for (int32_t h = 0; h < height; h++) {
      for (int32_t w = 0; w < width; w += 4) {
        int16x4_t input_data_tmp = vld1_s16(input + h * input_aligned_shape[3] + w);
        float32x4_t input_data_f32 = vcvtq_f32_s32(vmovl_s16(input_data_tmp));
        float32x4_t result = vmulq_f32(input_data_f32, scale_neon);
        vst1q_f32(output + h * width + w, result);
      }
    }
    input += input_aligned_shape[2] * input_aligned_shape[3];
    output += height * width;
  }
}

static int32_t nearest_interpolate(float *output_data,
                                   float *input_data,
                                   int16_t *weight,
                                   float *weight_scale,
                                   int32_t stride,
                                   int32_t channels,
                                   int32_t output_height,
                                   int32_t output_width,
                                   int32_t input_height,
                                   int32_t input_width,
                                   float scale_h,
                                   float scale_w) {
  float32x4_t weight_scale_val = vmulq_n_f32(vld1q_f32(weight_scale), stride);

  for (int32_t c{0}; c < channels; ++c) {
    for (int32_t y{0}; y < output_height; ++y) {
      int32_t idx_y = y / scale_h;
      int32_t output_offset = output_width * y;

      for (int32_t x{0}; x < output_width; x += 4) {
        int32_t idx_x = x / scale_w;
        int16x4_t weight_val = vld1_s16(&weight[y * output_width + x]);
        float32x4_t weight_val_vector = vcvtq_f32_s32(vmovl_s16(weight_val));
        float32x4_t weight_data =
            vmulq_f32(weight_val_vector, weight_scale_val);

        float32x4_t x_11 = vdupq_n_f32(input_data[idx_y * input_width + idx_x]);
        float32x4_t result = vmulq_f32(x_11, weight_data);

        float32x4_t current_output = vld1q_f32(&output_data[output_offset + x]);
        current_output = vaddq_f32(current_output, result);
        vst1q_f32(&output_data[output_offset + x], current_output);
      }
    }
    input_data += input_height * input_width;
    weight += output_height * output_width;
  }
  return 0;
}

static int32_t feature_add(float *spg,
                           float *feature,
                           int32_t input_height,
                           int32_t input_width,
                           int32_t maxdisp) {
  for (int i = 0; i < input_height; ++i) {
    for (int j = 0; j < input_width; ++j) {
      int index = i * input_width + j;
      feature[index] += spg[index];
      if (feature[index] < 0) feature[index] = 0;
      feature[index] *= maxdisp;
    }
  }
  return 0;
}

static int32_t feature_add_neon(float *spg,
                           float *feature,
                           int32_t input_height,
                           int32_t input_width,
                           int32_t maxdisp) {
  int total_elements = input_height * input_width;

  float32x4_t maxdisp_vec = vdupq_n_f32(maxdisp);
  float32x4_t zero_vec = vdupq_n_f32(0.0f);

  for (int i = 0; i < total_elements; i += 4) {
    float32x4_t spg_vec = vld1q_f32(spg + i);
    float32x4_t feature_vec = vld1q_f32(feature + i);
    feature_vec = vaddq_f32(feature_vec, spg_vec);
    feature_vec = vmaxq_f32(feature_vec, zero_vec);
    feature_vec = vmulq_f32(feature_vec, maxdisp_vec);
    vst1q_f32(feature + i, feature_vec);
  }

  return 0;
}

static int32_t dump_to_color(
    int32_t feat_h, int32_t feat_w,
    std::vector<float> &points,
    cv::Mat &feat_visual) {
  cv::Mat feat_mat(feat_h, feat_w, CV_32F, points.data());
  feat_mat.convertTo(feat_visual, CV_8U, 1, 0);
  cv::convertScaleAbs(feat_visual, feat_visual, 2);
  cv::applyColorMap(feat_visual, feat_visual, cv::COLORMAP_JET);
  return 0;
}

int postprocess3(std::vector<hbDNNTensor> &tensors,
                 std::vector<float> &points) {
  int low_max_stride_ = 4;
  int maxdisp_ = 192 * 2;
  int tensors_last_index = tensors.size() - 1;
  for (int32_t i = 0; i < 3; i++) {
    hbSysFlushMem(&(tensors[tensors_last_index - i].sysMem[0]),
        HB_SYS_MEM_CACHE_INVALIDATE);
  }
  // get tensor info
  int32_t *cost_data =
      reinterpret_cast<int32_t *>(tensors[tensors_last_index - 1].sysMem[0].virAddr);
  int32_t *cost_valid_shape = tensors[tensors_last_index - 1].properties.validShape.dimensionSize;
  int32_t *cost_aligned_shape =
      tensors[tensors_last_index- 1].properties.alignedShape.dimensionSize;
  float *cost_scale = tensors[tensors_last_index - 1].properties.scale.scaleData;
  int32_t unflod_c = cost_valid_shape[1];
  int32_t unflod_h = cost_valid_shape[2];
  int32_t unflod_w = cost_valid_shape[3];

  int16_t *spg_data = reinterpret_cast<int16_t *>(tensors[tensors_last_index].sysMem[0].virAddr);
  int32_t *spg_valid_shape = tensors[tensors_last_index].properties.validShape.dimensionSize;
  float *spg_scale = tensors[tensors_last_index].properties.scale.scaleData;

  int32_t spg_unflod_c = spg_valid_shape[1];
  int32_t spg_unflod_h = spg_valid_shape[2];
  int32_t spg_unflod_w = spg_valid_shape[3];
  int32_t *spg_aligned_shape =
      tensors[tensors_last_index].properties.alignedShape.dimensionSize;
  std::vector<float> feat(unflod_c * unflod_h * unflod_w);
  {
    ScopeProcessTime t("Dequantize feat");
    Dequantize(
        feat.data(), cost_data, cost_scale, cost_valid_shape, cost_aligned_shape);
  }
  // interpolate
  int32_t feat_h = unflod_h * low_max_stride_;
  int32_t feat_w = unflod_w * low_max_stride_;
  points.resize(feat_h * feat_w, 0.f);

  nearest_interpolate(points.data(),
                      feat.data(),
                      spg_data,
                      spg_scale,
                      maxdisp_,
                      unflod_c,
                      feat_h,
                      feat_w,
                      unflod_h,
                      unflod_w,
                      low_max_stride_,
                      low_max_stride_);
  return 0;
}

int postprocess2(std::vector<hbDNNTensor> &tensors,
    std::vector<float> &points) {
  int low_max_stride_ = 2;
  int maxdisp_ = 192;
  for (int32_t i = 0; i < 2; i++) {
    hbSysFlushMem(&(tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  }
  // get tensor info
  int16_t *cost_data =
      reinterpret_cast<int16_t *>(tensors[0].sysMem[0].virAddr);
  int32_t *cost_valid_shape = tensors[0].properties.validShape.dimensionSize;
  int32_t *cost_aligned_shape =
      tensors[0].properties.alignedShape.dimensionSize;
  float *cost_scale = tensors[0].properties.scale.scaleData;
  int32_t unflod_c = cost_valid_shape[1];
  int32_t unflod_h = cost_valid_shape[2];
  int32_t unflod_w = cost_valid_shape[3];

  int32_t *spg_data = reinterpret_cast<int32_t *>(tensors[1].sysMem[0].virAddr);
  int32_t *spg_valid_shape = tensors[1].properties.validShape.dimensionSize;
  float *spg_scale = tensors[1].properties.scale.scaleData;

  int32_t spg_unflod_c = spg_valid_shape[1];
  int32_t spg_unflod_h = spg_valid_shape[2];
  int32_t spg_unflod_w = spg_valid_shape[3];
  int32_t *spg_aligned_shape =
      tensors[0].properties.alignedShape.dimensionSize;

  std::vector<float> feat(unflod_c * unflod_h * unflod_w);
  Dequantize16_neon(
      feat.data(), cost_data, cost_scale, cost_valid_shape, cost_aligned_shape);

  std::vector<float> spg(spg_unflod_c * spg_unflod_h * spg_unflod_w);
  Dequantize(
      spg.data(), spg_data, spg_scale, spg_valid_shape, spg_aligned_shape);

  int32_t feat_h = unflod_h * low_max_stride_;
  int32_t feat_w = unflod_w * low_max_stride_;
  points.resize(feat_h * feat_w, 0.f);
  cv::Mat feat_mat(unflod_c * unflod_h, unflod_w, CV_32FC1, feat.data());
  cv::Mat points_mat(feat_h, feat_w, CV_32FC1, points.data());
  cv::resize(feat_mat, points_mat, cv::Size(feat_w, feat_h));
  feature_add_neon(spg.data(), points.data(), feat_h, feat_w, maxdisp_);
  return 0;
}

int postprocess(std::vector<hbDNNTensor> &tensors,
                std::vector<float> &points) {
  int low_max_stride_ = 2;
  int maxdisp_ = 192;
  for (int32_t i = 0; i < 2; i++) {
    hbSysFlushMem(&(tensors[i].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  }
  // get tensor info
  int32_t *cost_data =
      reinterpret_cast<int32_t *>(tensors[0].sysMem[0].virAddr);
  int32_t *cost_valid_shape = tensors[0].properties.validShape.dimensionSize;
  int32_t *cost_aligned_shape =
      tensors[0].properties.alignedShape.dimensionSize;
  float *cost_scale = tensors[0].properties.scale.scaleData;
  int32_t unflod_c = cost_valid_shape[1];
  int32_t unflod_h = cost_valid_shape[2];
  int32_t unflod_w = cost_valid_shape[3];

  int16_t *spg_data = reinterpret_cast<int16_t *>(tensors[1].sysMem[0].virAddr);
  int32_t *spg_valid_shape = tensors[1].properties.validShape.dimensionSize;
  float *spg_scale = tensors[1].properties.scale.scaleData;

  int32_t spg_unflod_c = spg_valid_shape[1];
  int32_t spg_unflod_h = spg_valid_shape[2];
  int32_t spg_unflod_w = spg_valid_shape[3];

  std::vector<float> feat(unflod_c * unflod_h * unflod_w);
  {
    ScopeProcessTime t("Dequantize feat");
    Dequantize(
        feat.data(), cost_data, cost_scale, cost_valid_shape, cost_aligned_shape);
  }
  low_max_stride_ = spg_unflod_h / unflod_h;
  // interpolate
  int32_t feat_h = unflod_h * low_max_stride_;
  int32_t feat_w = unflod_w * low_max_stride_;
  points.resize(feat_h * feat_w, 0.f);

  nearest_interpolate(points.data(),
                      feat.data(),
                      spg_data,
                      spg_scale,
                      maxdisp_,
                      unflod_c,
                      feat_h,
                      feat_w,
                      unflod_h,
                      unflod_w,
                      low_max_stride_,
                      low_max_stride_);
  return 0;
}

static int32_t print_model_info(hbPackedDNNHandle_t *packed_dnn_handle)
{
  int32_t i = 0, j = 0;
  hbDNNHandle_t dnn_handle;
  const char **model_name_list;
  int32_t model_count = 0;
  hbDNNTensorProperties properties;

  HB_CHECK_SUCCESS(hbDNNGetModelNameList(
      &model_name_list, &model_count, *packed_dnn_handle),
                   "hbDNNGetModelNameList failed");
  if (model_count <= 0) {
    std::cout << "Modle count <= 0" << std::endl;
    return -1;
  }
  HB_CHECK_SUCCESS(
      hbDNNGetModelHandle(&dnn_handle, *packed_dnn_handle, model_name_list[0]),
      "hbDNNGetModelHandle failed");

  std::cout << "Model info:\nmodel_name: \n" << model_name_list[0] << std::endl;

  int32_t input_count = 0;
  int32_t output_count = 0;
  HB_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle),
                   "hbDNNGetInputCount failed");
  HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle),
                   "hbDNNGetInputCount failed");

  std::cout << "Input count: " << input_count << std::endl;
  for (i = 0; i < input_count; i++) {
    HB_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&properties, dnn_handle, i),
        "hbDNNGetInputTensorProperties failed");

    std::cout << "input[" << i << "]: tensorLayout: " << properties.tensorLayout
              << " tensorType: " << properties.tensorType << " validShape:(";
    for (j = 0; j < properties.validShape.numDimensions; j++)
      std::cout << properties.validShape.dimensionSize[j] << ", ";
    std::cout << "), alignedShape:(";
    for (j = 0; j < properties.alignedShape.numDimensions; j++)
      std::cout << ", " << properties.alignedShape.dimensionSize[j];
    std::cout << ")" << std::endl;
  }

  std::cout << "Output count: " << output_count << std::endl;
  for (i = 0; i < output_count; i++) {
    HB_CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&properties, dnn_handle, i),
        "hbDNNGetOutputTensorProperties failed");
    std::cout << "Output[" << i << "]: tensorLayout: " << properties.tensorLayout
              << " tensorType: " << properties.tensorType << " validShape:(";
    for (j = 0; j < properties.validShape.numDimensions; j++)
      std::cout << properties.validShape.dimensionSize[j] << ", ";
    std::cout << "), alignedShape:(";
    for (j = 0; j < properties.alignedShape.numDimensions; j++)
      std::cout << ", " << properties.alignedShape.dimensionSize[j];
    std::cout << ")" << std::endl;
  }
  return 0;
}

static int32_t prepare_input_tensor(std::vector<hbDNNTensor> &input_tensor,
                                    hbDNNHandle_t dnn_handle) {
  int model_h, model_w;
  input_tensor.resize(2);

  hbDNNTensorProperties properties = {0};
  for (auto &tensor : input_tensor) {
    HB_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");
    tensor.properties = properties;
    tensor.properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE;
    switch (properties.tensorLayout) {
      case HB_DNN_LAYOUT_NHWC:
        model_h = properties.validShape.dimensionSize[1];
        model_w = properties.validShape.dimensionSize[2];
        break;
      case HB_DNN_LAYOUT_NCHW:
        model_h = properties.validShape.dimensionSize[2];
        model_w = properties.validShape.dimensionSize[3];
        break;
      default:
        return -1;
    }
    tensor.properties.validShape.numDimensions = 4;
    tensor.properties.validShape.dimensionSize[0] = 1;
    tensor.properties.validShape.dimensionSize[1] = 3;
    tensor.properties.validShape.dimensionSize[2] = model_h; 
    tensor.properties.validShape.dimensionSize[3] = model_w;
    tensor.properties.alignedShape = tensor.properties.validShape;
    
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&tensor.sysMem[0],
                                         model_h * model_w),
                     "hbSysAllocCachedMem failed");
    tensor.sysMem[0].memSize = model_h * model_w;

    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&tensor.sysMem[1],
                                         model_h * model_w / 2),
                     "hbSysAllocCachedMem failed");
    tensor.sysMem[1].memSize = model_h * model_w / 2;
  }
  return 0;
}

static int32_t prepare_output_tensor(std::vector<hbDNNTensor> &output_tensor,
                                     hbDNNHandle_t dnn_handle) {
  int32_t ret = 0;
  int32_t i = 0;
  int32_t output_count = 0;
  // hbDNNTensorProperties properties;
  hbDNNGetOutputCount(&output_count, dnn_handle);
  output_tensor.resize(output_count);
  for (i = 0; i < output_count; ++i) {
    HB_CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&output_tensor[i].properties, dnn_handle, i),
        "hbDNNGetOutputTensorProperties failed");
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&output_tensor[i].sysMem[0],
        output_tensor[i].properties.alignedByteSize),
                     "hbSysAllocCachedMem failed");
  }
  return ret;
}

static int32_t get_model_input_size(hbDNNHandle_t dnn_handle,
    int32_t &width, int32_t &height) {
  hbDNNTensorProperties properties = {0};
  HB_CHECK_SUCCESS(
      hbDNNGetInputTensorProperties(&properties, dnn_handle, 0),
      "hbDNNGetInputTensorProperties failed");
  switch (properties.tensorLayout) {
    case HB_DNN_LAYOUT_NHWC:
      height = properties.validShape.dimensionSize[1];
      width = properties.validShape.dimensionSize[2];
      break;
    case HB_DNN_LAYOUT_NCHW:
      height = properties.validShape.dimensionSize[2];
      width = properties.validShape.dimensionSize[3];
      break;
    default:
      return -1;
  }
  return 0;
}

static int32_t get_model_output_size(hbDNNHandle_t dnn_handle,
                                     int32_t &width, int32_t &height) {
  hbDNNTensorProperties properties = {0};
  HB_CHECK_SUCCESS(
      hbDNNGetOutputTensorProperties(&properties, dnn_handle, 1),
      "hbDNNGetInputTensorProperties failed");
  switch (properties.tensorLayout) {
    case HB_DNN_LAYOUT_NHWC:
      height = properties.validShape.dimensionSize[1];
      width = properties.validShape.dimensionSize[2];
      break;
    case HB_DNN_LAYOUT_NCHW:
      height = properties.validShape.dimensionSize[2];
      width = properties.validShape.dimensionSize[3];
      break;
    default:
      return -1;
  }
  return 0;
}

static int32_t release_tensor(std::vector<hbDNNTensor> &output_tensor, int mem_len)
{
  for (auto & i : output_tensor) {
    for (int j = 0; j < mem_len; ++j) {
      HB_CHECK_SUCCESS(hbSysFreeMem(&(i.sysMem[j])),
                       "hbSysFreeMem failed");
    }
  }
  return 0;
}

static void bgr_to_nv12(const cv::Mat &bgr, cv::Mat &nv12) {
  int width = bgr.cols;
  int height = bgr.rows;
  nv12 = cv::Mat(height * 3 / 2, width, CV_8UC1);
  image_conversion::bgr24_to_nv12_neon(bgr.data, nv12.data, width, height);
}

int StereonetProcess::stereonet_init(const std::string &model_file_name) {
  int32_t model_count = 0;
  hbDNNTensorProperties properties;
  hbPackedDNNHandle_t packed_dnn_handle;
  hbDNNHandle_t dnn_handle;
  const char **model_name_list;
  const char *model_file = model_file_name.c_str();
//  hbDNNInitializeFromFiles(&packed_dnn_handle, (char const **)&model_file, 1);
  // 加载模型
  HB_CHECK_SUCCESS(
      hbDNNInitializeFromFiles(&packed_dnn_handle, (char const **)&model_file, 1),
      "hbDNNInitializeFromFiles failed"); // 从本地文件加载模型

  // 打印模型信息
  print_model_info(&packed_dnn_handle);

  HB_CHECK_SUCCESS(hbDNNGetModelNameList(
      &model_name_list, &model_count, packed_dnn_handle),
                   "hbDNNGetModelNameList failed");
  if (model_count <= 0) {
    printf("Modle count <= 0\n");
    return -1;
  }

  HB_CHECK_SUCCESS(
      hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]),
      "hbDNNGetModelHandle failed");

  packed_dnn_handle_ = packed_dnn_handle;
  dnn_handle_ = dnn_handle;

  get_model_input_size(dnn_handle_, model_input_w_, model_input_h_);
  get_model_output_size(dnn_handle_, model_output_w_, model_output_h_);


  for (int i = 0; i < MAX_PROCESS_COUNT; i++) {
    idle_tensor_.emplace_back(true);
  }

  input_tensors_.resize(MAX_PROCESS_COUNT);
  for (int i = 0; i < MAX_PROCESS_COUNT; i++) {
    prepare_input_tensor(input_tensors_[i], dnn_handle_);
  }

  output_tensors_.resize(MAX_PROCESS_COUNT);
  for (int i = 0; i < MAX_PROCESS_COUNT; ++i) {
    prepare_output_tensor(output_tensors_[i], dnn_handle_);
  }
  return 0;
}

int StereonetProcess::stereonet_deinit() {
  for (auto & input_tensor : input_tensors_) {
    release_tensor(input_tensor, 2);
  }
  for (auto & output_tensor : output_tensors_) {
    release_tensor(output_tensor, 1);
  }
  HB_CHECK_SUCCESS(hbDNNRelease(packed_dnn_handle_),
      "hbDNNRelease failed");
  return 0;
}

int StereonetProcess::get_idle_tensor() {
  for (int i = 0; i < MAX_PROCESS_COUNT; ++i) {
    if (idle_tensor_[i]) {
      idle_tensor_[i] = false;
      return i;
    }
  }
  return -1;
}

int StereonetProcess::set_tensor_idle(int tensor_id) {
  if (tensor_id >=0 || tensor_id < MAX_PROCESS_COUNT) {
    idle_tensor_[tensor_id] = true;
    return 0;
  }
  return -1;
}

int StereonetProcess::stereonet_inference(
    const cv::Mat &left_img,
    const cv::Mat &right_img,
    bool is_nv12,
    std::vector<float> &points) {
  int ret = 0;
  int idle_tensor_id = -1;
  hbDNNTaskHandle_t task_handle = nullptr;
  cv::Mat left_img_nv12, right_img_nv12;

  if ((idle_tensor_id = get_idle_tensor()) == -1) {
    std::cout << "get_idle_tensor failed" << std::endl;
    return StereonetErrorCode::TENSOR_BUSY;
  }
  if (is_nv12) {
    left_img_nv12 = left_img;
    right_img_nv12 = right_img;
  } else {
    ScopeProcessTime t("bgr_to_nv12");
    bgr_to_nv12(left_img, left_img_nv12);
    bgr_to_nv12(right_img, right_img_nv12);
  }

  hbDNNTensor &left_input_tensor  = input_tensors_[idle_tensor_id][0],
              &right_input_tensor = input_tensors_[idle_tensor_id][1];
  /*
  assert(left_img_nv12.rows * left_img_nv12.cols == model_input_h_ * model_input_w_ * 3 / 2);
  assert((left_input_tensor.sysMem[0].memSize +
      left_input_tensor.sysMem[1].memSize) == model_input_h_ * model_input_w_ * 3 / 2);

  assert(right_img_nv12.rows * right_img_nv12.cols == model_input_h_ * model_input_w_ * 3 / 2);
  assert((right_input_tensor.sysMem[0].memSize +
      right_input_tensor.sysMem[1].memSize) == model_input_h_ * model_input_w_ * 3 / 2);


  static int iii = 0;
  std::stringstream iss;
  iss << std::setw(6) << std::setfill('0') << iii++;
  auto image_seq = iss.str();
  std::ofstream bin( "./230ai_data/" + image_seq + ".yuv", std::ios::out | std::ios::binary);
  bin.write((const char*)(left_img_nv12.data), left_input_tensor.sysMem[0].memSize + left_input_tensor.sysMem[1].memSize);
  bin.write((const char*)right_img_nv12.data, right_input_tensor.sysMem[0].memSize + right_input_tensor.sysMem[1].memSize);
   */

  hbSysWriteMem(&left_input_tensor.sysMem[0],
                (char *)left_img_nv12.data,
                left_input_tensor.sysMem[0].memSize);
  hbSysWriteMem(&left_input_tensor.sysMem[1],
                (char *) left_img_nv12.data + left_input_tensor.sysMem[0].memSize,
                left_input_tensor.sysMem[1].memSize);

  hbSysWriteMem(&right_input_tensor.sysMem[0],
                (char *)right_img_nv12.data,
                right_input_tensor.sysMem[0].memSize);
  hbSysWriteMem(&right_input_tensor.sysMem[1],
                (char *) right_img_nv12.data + right_input_tensor.sysMem[0].memSize,
                right_input_tensor.sysMem[1].memSize);

  hbSysFlushMem(&left_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(&left_input_tensor.sysMem[1], HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(&right_input_tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
  hbSysFlushMem(&right_input_tensor.sysMem[1], HB_SYS_MEM_CACHE_CLEAN);

  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  hbDNNTensor *output = &output_tensors_[idle_tensor_id][0];

  ret = hbDNNInfer(&task_handle,
                   &output,
                   &input_tensors_[idle_tensor_id][0],
                   dnn_handle_,
                   &infer_ctrl_param);
  if (ret) {
    set_tensor_idle(idle_tensor_id);
    printf("hbDNNInfer failed\n");
    return StereonetErrorCode::DNN_ERROR;
  }
  // wait task done
  {
    ScopeProcessTime t("hbDNNWaitTaskDone");
    ret = hbDNNWaitTaskDone(task_handle, 0);
    if (ret) {
      set_tensor_idle(idle_tensor_id);
      printf("hbDNNWaitTaskDone failed\n");
      return StereonetErrorCode::DNN_ERROR;
    }
  }

  for (int32_t i = 0; i < output_count_; i++) {
    hbSysFlushMem(&(output_tensors_[idle_tensor_id][i].sysMem[0]),
        HB_SYS_MEM_CACHE_INVALIDATE);
  }

  // release task handle
  ret = hbDNNReleaseTask(task_handle);
  set_tensor_idle(idle_tensor_id);
  if (ret) {
    printf("hbDNNReleaseTask failed\n");
    return StereonetErrorCode::DNN_ERROR;
  }

  ScopeProcessTime t("postprocess");
  postprocess(output_tensors_[idle_tensor_id], points);
  return StereonetErrorCode::OK;
}

StereonetProcess::StereonetProcess() {}
