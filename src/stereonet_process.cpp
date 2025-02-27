//
// Created by zhy on 7/1/24.
//

#include <arm_neon.h>
#include <fstream>
#include <cassert>
#include <vector>

#include "stereonet_process.h"
#include "image_conversion.h"

static std::string tensor_type_to_str(int32_t tensor_type)
{
    switch (tensor_type)
    {
#ifdef PLATFORM_X5
    case HB_DNN_IMG_TYPE_Y:
        return "HB_DNN_IMG_TYPE_Y";
    case HB_DNN_IMG_TYPE_NV12:
        return "HB_DNN_IMG_TYPE_NV12";
    case HB_DNN_IMG_TYPE_NV12_SEPARATE:
        return "HB_DNN_IMG_TYPE_NV12_SEPARATE";
    case HB_DNN_IMG_TYPE_YUV444:
        return "HB_DNN_IMG_TYPE_YUV444";
    case HB_DNN_IMG_TYPE_RGB:
        return "HB_DNN_IMG_TYPE_RGB";
    case HB_DNN_IMG_TYPE_BGR:
        return "HB_DNN_IMG_TYPE_BGR";
#endif
    case HB_DNN_TENSOR_TYPE_S4:
        return "HB_DNN_TENSOR_TYPE_S4";
    case HB_DNN_TENSOR_TYPE_U4:
        return "HB_DNN_TENSOR_TYPE_U4";
    case HB_DNN_TENSOR_TYPE_S8:
        return "HB_DNN_TENSOR_TYPE_S8";
    case HB_DNN_TENSOR_TYPE_U8:
        return "HB_DNN_TENSOR_TYPE_U8";
    case HB_DNN_TENSOR_TYPE_F16:
        return "HB_DNN_TENSOR_TYPE_F16";
    case HB_DNN_TENSOR_TYPE_S16:
        return "HB_DNN_TENSOR_TYPE_S16";
    case HB_DNN_TENSOR_TYPE_U16:
        return "HB_DNN_TENSOR_TYPE_U16";
    case HB_DNN_TENSOR_TYPE_F32:
        return "HB_DNN_TENSOR_TYPE_F32";
    case HB_DNN_TENSOR_TYPE_S32:
        return "HB_DNN_TENSOR_TYPE_S32";
    case HB_DNN_TENSOR_TYPE_U32:
        return "HB_DNN_TENSOR_TYPE_U32";
    case HB_DNN_TENSOR_TYPE_F64:
        return "HB_DNN_TENSOR_TYPE_F64";
    case HB_DNN_TENSOR_TYPE_S64:
        return "HB_DNN_TENSOR_TYPE_S64";
    case HB_DNN_TENSOR_TYPE_U64:
        return "HB_DNN_TENSOR_TYPE_U64";
    case HB_DNN_TENSOR_TYPE_MAX:
        return "HB_DNN_TENSOR_TYPE_MAX";
    default:
        return "Unknown";
    }
}

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

int postprocess_v1(std::vector<hbDNNTensor> &tensors,
                std::vector<float> &points,
                int max_disp) {
  int low_max_stride_ = 2;
  for (int32_t i = 0; i < 2; i++) {
    hbSysFlushMem(&(TENSOR_SYSMEM(tensors[i], 0)), HB_SYS_MEM_CACHE_INVALIDATE);
  }
  // get tensor info
  int32_t *cost_data =
      reinterpret_cast<int32_t *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
  int32_t *cost_valid_shape = tensors[0].properties.validShape.dimensionSize;
  int32_t cost_aligned_shape[4];
  cost_aligned_shape[0] = ALIGN_32(cost_valid_shape[0]);
  cost_aligned_shape[1] = ALIGN_32(cost_valid_shape[1]);
  cost_aligned_shape[2] = ALIGN_32(cost_valid_shape[2]);
  cost_aligned_shape[3] = ALIGN_32(cost_valid_shape[3]);
  float *cost_scale = tensors[0].properties.scale.scaleData;
  int32_t unflod_c = cost_valid_shape[1];
  int32_t unflod_h = cost_valid_shape[2];
  int32_t unflod_w = cost_valid_shape[3];

  int16_t *spg_data = reinterpret_cast<int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);
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
                      max_disp,
                      unflod_c,
                      feat_h,
                      feat_w,
                      unflod_h,
                      unflod_w,
                      low_max_stride_,
                      low_max_stride_);
  return 0;
}

int postprocess_v2(std::vector<hbDNNTensor> &tensors,
                std::vector<float> &points,
                int max_disp) {
  for (int32_t i = 0; i < 2; i++) {
    hbSysFlushMem(&(TENSOR_SYSMEM(tensors[i], 0)),
        HB_SYS_MEM_CACHE_INVALIDATE);
  }

  // get shape info
  int32_t *disp_shape = tensors[0].properties.validShape.dimensionSize;
  int c_dim = disp_shape[1];
  int h_dim = disp_shape[2];
  int w_dim = disp_shape[3];

  // calc disp
  Eigen::MatrixXf result = Eigen::MatrixXf::Zero(h_dim, w_dim);
  if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32
  && tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_F32) {
    // get tensor info
    float *disp = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    float *spx = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    // multiply element-wise and then add in the c channel

    for (int i = 0; i < c_dim; ++i) {
      Eigen::Map<Eigen::Matrix<float,
          Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim, w_dim);
      Eigen::Map<Eigen::Matrix<float,
          Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim, w_dim);
      result.noalias() += matrix_disp.cwiseProduct(matrix_spx);
    }

    // write the result to the points
    points.resize(h_dim * w_dim, 0.f);
    Eigen::Map<Eigen::MatrixXf>(points.data(), h_dim, w_dim).noalias() = result;
  } else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32
  && tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16) {
    // get tensor info
    float *disp = reinterpret_cast<float *>(TENSOR_SYSMEM(tensors[0], 0).virAddr);
    int16_t *spx = reinterpret_cast<int16_t *>(TENSOR_SYSMEM(tensors[1], 0).virAddr);

    // multiply element-wise and then add in the c channel
    for (int i = 0; i < c_dim; ++i) {
      Eigen::Map<Eigen::Matrix<float,
          Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim, w_dim);
      Eigen::Map<Eigen::Matrix<int16_t,
          Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim, w_dim);
      result.noalias() += matrix_disp.cwiseProduct(matrix_spx.cast<float>());
    }
  } else {
    RCLCPP_INFO_STREAM(rclcpp::get_logger(""),
        "=> output tensor type unsupported! tensor[0]: "
        << tensor_type_to_str(tensors[0].properties.tensorType)
        << ", tensor[1]: " << tensor_type_to_str(tensors[1].properties.tensorType));
    return -1;
  }

  // get scale info
  float scale_constant = 1.0;
  float *disp_scale = &scale_constant;
  float *spx_scale = &scale_constant;
  if (tensors[0].properties.quantiType == SCALE) {
    disp_scale = tensors[0].properties.scale.scaleData;
  }
  if (tensors[1].properties.quantiType == SCALE) {
    spx_scale = tensors[1].properties.scale.scaleData;
  }

  // write the result to the points
  points.resize(h_dim * w_dim, 0.f);
  Eigen::Map<Eigen::MatrixXf>(points.data(), h_dim, w_dim).noalias() = result * (*disp_scale) * (*spx_scale);

  return 0;
}

int postprocess_v3(std::vector<hbDNNTensor> &tensors,
                std::vector<float> &points,
                int max_disp,
                float uncertainty_th) {
  cv::Mat mask, uncert, infer_disp, init_disp;
  int32_t *disp_shape = tensors[0].properties.validShape.dimensionSize;
  int32_t c_dim = disp_shape[1];
  int32_t h_dim = disp_shape[2];
  int32_t w_dim = disp_shape[3];              
  std::vector<float> infer_points, init_points;           
  std::vector<hbDNNTensor> infer_disp_tensor(tensors.begin(), tensors.begin() + 2);
  std::vector<hbDNNTensor> init_disp_tensor(tensors.begin() + 2, tensors.begin() + 4);
  postprocess_v2(infer_disp_tensor, infer_points, max_disp);
  if (uncertainty_th > 0.0f) {
    postprocess_v2(init_disp_tensor, init_points, max_disp);
    infer_disp = cv::Mat(h_dim, w_dim, CV_32FC1, infer_points.data());
    init_disp = cv::Mat(h_dim, w_dim, CV_32FC1, init_points.data());
    uncert = cv::abs(init_disp - infer_disp) / init_disp;
    cv::threshold(uncert, mask, uncertainty_th, 1, cv::THRESH_BINARY_INV);  
    infer_disp = infer_disp.mul(mask);
  }
  points = std::move(infer_points);
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
                   "hbDNNGetOutputCount failed");

  std::cout << "Input count: " << input_count << std::endl;
  for (i = 0; i < input_count; i++) {
    HB_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&properties, dnn_handle, i),
        "hbDNNGetInputTensorProperties failed");
#ifdef PLATFORM_X5
    std::cout << "input[" << i << "]: tensorLayout: " << properties.tensorLayout
              << " tensorType: " << properties.tensorType << " validShape:(";
#endif
#ifdef PLATFORM_S100
    std::cout << "input[" << i << "]: " << " tensorType: "
              << properties.tensorType << " validShape:(";
#endif
    for (j = 0; j < properties.validShape.numDimensions; j++)
      std::cout << properties.validShape.dimensionSize[j] << ", ";
    std::cout << "), alignedShape:(";
    for (j = 0; j < properties.validShape.numDimensions; j++)
      std::cout << ", " << ALIGN_32(properties.validShape.dimensionSize[j]);
    std::cout << ")" << std::endl;
  }

  std::cout << "Output count: " << output_count << std::endl;
  for (i = 0; i < output_count; i++) {
    HB_CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&properties, dnn_handle, i),
        "hbDNNGetOutputTensorProperties failed");
#ifdef PLATFORM_X5
    std::cout << "input[" << i << "]: tensorLayout: " << properties.tensorLayout
              << " tensorType: " << properties.tensorType << " validShape:(";
#endif
#ifdef PLATFORM_S100
    std::cout << "input[" << i << "]: " << " tensorType: "
              << properties.tensorType << " validShape:(";
#endif
    for (j = 0; j < properties.validShape.numDimensions; j++)
      std::cout << properties.validShape.dimensionSize[j] << ", ";
    std::cout << "), alignedShape:(";
    for (j = 0; j < properties.validShape.numDimensions; j++)
      std::cout << ", " << ALIGN_32(properties.validShape.dimensionSize[j]);
    std::cout << ")" << std::endl;
  }
  return 0;
}

int32_t StereonetProcess::prepare_input_tensor(std::vector<hbDNNTensor> &input_tensor,
                                    hbDNNHandle_t dnn_handle) {
  int model_h, model_w;
  input_tensor.resize(2);

  hbDNNTensorProperties properties = {0};
  for (auto &tensor : input_tensor) {
    HB_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");
    tensor.properties = properties;
    input_tensor_type_ = properties.tensorType;
    RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "=> input tensor type: " <<
                        tensor_type_to_str(tensor.properties.tensorType));
    // tensor.properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE;
    get_hw(properties, model_h, model_w);
    tensor.properties.validShape.numDimensions = 4;
    tensor.properties.validShape.dimensionSize[0] = 1;
    tensor.properties.validShape.dimensionSize[1] = 3;
    tensor.properties.validShape.dimensionSize[2] = model_h; 
    tensor.properties.validShape.dimensionSize[3] = model_w;
#ifdef PLATFORM_X5
    tensor.properties.alignedShape = tensor.properties.validShape;
#endif
    // check input tensor type
    if (properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
      HB_CHECK_SUCCESS(hbSysAllocCachedMem(&TENSOR_SYSMEM(tensor, 0), model_h * model_w),
          "hbSysAllocCachedMem failed");
      TENSOR_SYSMEM(tensor, 0).memSize = model_h * model_w;

      HB_CHECK_SUCCESS(hbSysAllocCachedMem(&TENSOR_SYSMEM(tensor, 1), model_h * model_w / 2),
          "hbSysAllocCachedMem failed");
      TENSOR_SYSMEM(tensor, 1).memSize = model_h * model_w / 2;
    } else if (properties.tensorType == HB_DNN_IMG_TYPE_NV12) {
      HB_CHECK_SUCCESS(hbSysAllocCachedMem(&TENSOR_SYSMEM(tensor, 0), (3 * model_h * model_w) / 2),
          "hbSysAllocCachedMem failed");
      TENSOR_SYSMEM(tensor, 0).memSize = (3 * model_h * model_w) / 2;
    } else {
      return -1;
    }
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
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&TENSOR_SYSMEM(output_tensor[i], 0),
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
  get_hw(properties, height, width);
  return 0;
}

static int32_t get_model_output_size(hbDNNHandle_t dnn_handle,
                                     int32_t &width, int32_t &height) {
  hbDNNTensorProperties properties = {0};
  HB_CHECK_SUCCESS(
      hbDNNGetOutputTensorProperties(&properties, dnn_handle, 1),
      "hbDNNGetInputTensorProperties failed");
  get_hw(properties, height, width);
  return 0;
}

static int32_t release_tensor(std::vector<hbDNNTensor> &output_tensor, int mem_len)
{
  for (auto & i : output_tensor) {
    for (int j = 0; j < mem_len; ++j) {
      HB_CHECK_SUCCESS(hbSysFreeMem(&(TENSOR_SYSMEM(i, j))),
                       "hbSysFreeMem failed");
    }
  }
  return 0;
}

int StereonetProcess::stereonet_init(const std::string &model_file_name,
    int max_disp, const std::string &postprocess, float uncertainty_th) {
  postprocess_ = postprocess;
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

  max_disp_ = max_disp;
  HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count_, dnn_handle_),
      "hbDNNGetOutputCount failed");  
      
  uncertainty_th_ = uncertainty_th;
      
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
    image_conversion::bgr_to_nv12(left_img, left_img_nv12);
    image_conversion::bgr_to_nv12(right_img, right_img_nv12);
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

  if (input_tensor_type_ == HB_DNN_IMG_TYPE_NV12_SEPARATE) {
    hbSysWriteMem(&TENSOR_SYSMEM(left_input_tensor, 0),
        (char *)left_img_nv12.data, TENSOR_SYSMEM(left_input_tensor, 0).memSize);
    hbSysWriteMem(&TENSOR_SYSMEM(left_input_tensor, 1),
        (char *) left_img_nv12.data + TENSOR_SYSMEM(left_input_tensor, 0).memSize,
        TENSOR_SYSMEM(left_input_tensor, 1).memSize);

    hbSysWriteMem(&TENSOR_SYSMEM(right_input_tensor, 0),
        (char *)right_img_nv12.data, TENSOR_SYSMEM(right_input_tensor, 0).memSize);
    hbSysWriteMem(&TENSOR_SYSMEM(right_input_tensor, 1),
        (char *) right_img_nv12.data + TENSOR_SYSMEM(right_input_tensor, 0).memSize,
        TENSOR_SYSMEM(right_input_tensor, 1).memSize);

    hbSysFlushMem(&TENSOR_SYSMEM(left_input_tensor, 0), HB_SYS_MEM_CACHE_CLEAN);
    hbSysFlushMem(&TENSOR_SYSMEM(left_input_tensor, 1), HB_SYS_MEM_CACHE_CLEAN);
    hbSysFlushMem(&TENSOR_SYSMEM(right_input_tensor, 0), HB_SYS_MEM_CACHE_CLEAN);
    hbSysFlushMem(&TENSOR_SYSMEM(right_input_tensor, 1), HB_SYS_MEM_CACHE_CLEAN);
  } else if (input_tensor_type_ == HB_DNN_IMG_TYPE_NV12) {
    hbSysWriteMem(&TENSOR_SYSMEM(left_input_tensor, 0), (char *)left_img_nv12.data,
                  TENSOR_SYSMEM(left_input_tensor, 0).memSize);
    hbSysWriteMem(&TENSOR_SYSMEM(right_input_tensor, 0), (char *)right_img_nv12.data,
                  TENSOR_SYSMEM(right_input_tensor, 0).memSize);

    hbSysFlushMem(&TENSOR_SYSMEM(left_input_tensor, 0), HB_SYS_MEM_CACHE_CLEAN);
    hbSysFlushMem(&TENSOR_SYSMEM(right_input_tensor, 0), HB_SYS_MEM_CACHE_CLEAN);
  } else {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger(""),
        "\033[31m=> input tensor flush mem errror:" << tensor_type_to_str(input_tensor_type_) << "!\033[0m");
    return StereonetErrorCode::INPUT_ERROR;
  }

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
    hbSysFlushMem(&(TENSOR_SYSMEM(output_tensors_[idle_tensor_id][i], 0)),
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
  if (postprocess_ == "v1")
  {
    postprocess_v1(output_tensors_[idle_tensor_id], points, max_disp_);
  }
  else if (postprocess_ == "v2")
  {
    postprocess_v2(output_tensors_[idle_tensor_id], points, max_disp_);
  }
  else if (postprocess_ == "v3") {
    postprocess_v3(output_tensors_[idle_tensor_id], points, max_disp_, uncertainty_th_);
  }
  return StereonetErrorCode::OK;
}

StereonetProcess::StereonetProcess() {}
