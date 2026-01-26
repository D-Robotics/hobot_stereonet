// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_HB_DNN_H_
#define DNN_HB_DNN_H_

#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define HB_DNN_TENSOR_MAX_DIMENSIONS (10)

typedef void *hbDNNPackedHandle_t;
typedef void *hbDNNHandle_t;

typedef enum {
  HB_DNN_TENSOR_TYPE_S4,
  HB_DNN_TENSOR_TYPE_U4,
  HB_DNN_TENSOR_TYPE_S8,
  HB_DNN_TENSOR_TYPE_U8,
  HB_DNN_TENSOR_TYPE_F16,
  HB_DNN_TENSOR_TYPE_S16,
  HB_DNN_TENSOR_TYPE_U16,
  HB_DNN_TENSOR_TYPE_F32,
  HB_DNN_TENSOR_TYPE_S32,
  HB_DNN_TENSOR_TYPE_U32,
  HB_DNN_TENSOR_TYPE_F64,
  HB_DNN_TENSOR_TYPE_S64,
  HB_DNN_TENSOR_TYPE_U64,
  HB_DNN_TENSOR_TYPE_BOOL8,
  HB_DNN_TENSOR_TYPE_MAX
} hbDNNDataType;

typedef struct hbDNNTensorShape {
  int32_t dimensionSize[HB_DNN_TENSOR_MAX_DIMENSIONS];
  int32_t numDimensions;
} hbDNNTensorShape;

/**
 * Quantize/Dequantize by scale
 * For Dequantize:
 * if zeroPointLen = 0 f(x_i) = x_i * scaleData[i]
 * if zeroPointLen > 0 f(x_i) = (x_i - zeroPointData[i]) * scaleData[i]
 * 
 * For Quantize:
 * if zeroPointLen = 0 f(x_i) = g(x_i / scaleData[i])
 * if zeroPointLen > 0 f(x_i) = g(x_i / scaleData[i] + zeroPointData[i])
 * which g(x) = clip(nearbyint(x)), use fesetround(FE_TONEAREST), U8: 0 <= g(x) <= 255, S8: -128 <= g(x) <= 127
 */
typedef struct hbDNNQuantiScale {
  int32_t scaleLen;
  float *scaleData;
  int32_t zeroPointLen;
  int32_t *zeroPointData;
} hbDNNQuantiScale;

typedef enum {
  NONE,  // no quantization
  SCALE
} hbDNNQuantiType;

typedef struct hbDNNTensorProperties {
  hbDNNTensorShape validShape;
  int32_t tensorType;
  hbDNNQuantiScale scale;
  hbDNNQuantiType quantiType;
  int32_t quantizeAxis;
  int64_t alignedByteSize;
  int64_t stride[HB_DNN_TENSOR_MAX_DIMENSIONS];
} hbDNNTensorProperties;

typedef struct hbDNNTensor {
  hbUCPSysMem sysMem;
  hbDNNTensorProperties properties;
} hbDNNTensor;

typedef enum {
  HB_DNN_DESC_TYPE_UNKNOWN = 0,
  HB_DNN_DESC_TYPE_STRING
} hbDNNDescType;

/**
 * @brief Get DNN version
 * 
 * @return DNN version info
 */
char const *hbDNNGetVersion();

/**
 * @brief Creates and initializes Horizon DNN Networks from file list
 * 
 * @param[out] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @param[in] modelFileNames Path of the model files.
 * @param[in] modelFileCount Number of the model files.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNInitializeFromFiles(hbDNNPackedHandle_t *dnnPackedHandle,
                                 char const **modelFileNames,
                                 int32_t modelFileCount);

/**
 * @brief Creates and initializes Horizon DNN Networks from memory
 * 
 * @param[out] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @param[in] modelData Pointer to the model file
 * @param[in] modelDataLengths Length of the model data.
 * @param[in] modelDataCount Length of the model data.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNInitializeFromDDR(hbDNNPackedHandle_t *dnnPackedHandle,
                               const void **modelData,
                               int32_t *modelDataLengths,
                               int32_t modelDataCount);

/**
 * @brief Release DNN Networks in a given packed handle
 * 
 * @param[in] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNRelease(hbDNNPackedHandle_t dnnPackedHandle);

/**
 * @brief Get model names from given packed handle
 * 
 * @param[out] modelNameList List of model names.
 * @param[out] modelNameCount Number of model names.
 * @param[in] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetModelNameList(char const ***modelNameList,
                              int32_t *modelNameCount,
                              hbDNNPackedHandle_t dnnPackedHandle);

/**
 * @brief Get DNN Network handle from packed Handle with given model name
 * 
 * @param[out] dnnHandle DNN handle, pointing to one model.
 * @param[in] dnnPackedHandle DNN handle, pointing to multiple models.
 * @param[in] modelName Model name.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetModelHandle(hbDNNHandle_t *dnnHandle,
                            hbDNNPackedHandle_t dnnPackedHandle,
                            char const *modelName);

/**
 * @brief Get input count
 * 
 * @param[out] inputCount Number of input tensors of the model.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputCount(int32_t *inputCount, hbDNNHandle_t dnnHandle);

/**
 * @brief Get model input name
 * 
 * @param[out] name Name of the input tensor of the model.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] inputIndex Index of the input tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputName(char const **name, hbDNNHandle_t dnnHandle,
                          int32_t inputIndex);

/**
 * @brief Get input tensor properties
 * 
 * @param[out] properties Info of the input tensor.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] inputIndex Index of the input tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputTensorProperties(hbDNNTensorProperties *properties,
                                      hbDNNHandle_t dnnHandle,
                                      int32_t inputIndex);

/**
 * @brief Get output count
 * 
 * @param[out] outputCount Number of the output tensors of the model.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputCount(int32_t *outputCount, hbDNNHandle_t dnnHandle);

/**
 * @brief Get model output name
 * 
 * @param[out] name Name of the output tensor of the model.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] outputIndex Index of the output tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputName(char const **name, hbDNNHandle_t dnnHandle,
                           int32_t outputIndex);

/**
 * @brief Get output tensor properties
 * 
 * @param[out] properties Info of the output tensor.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] outputIndex Index of the output tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputTensorProperties(hbDNNTensorProperties *properties,
                                       hbDNNHandle_t dnnHandle,
                                       int32_t outputIndex);

/**
 * @brief Get model input description
 * 
 * @param[out] desc Address of the description information.
 * @param[out] size Size of the description information.
 * @param[out] type Type of the description information, please refer to hbDNNDescType.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] inputIndex Index of the input tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetInputDesc(char const **desc, uint32_t *size, int32_t *type,
                          hbDNNHandle_t dnnHandle, int32_t inputIndex);

/**
 * @brief Get model output description
 * 
 * @param[out] desc Address of the description information.
 * @param[out] size Size of the description information.
 * @param[out] type Type of the description information, please refer to hbDNNDescType.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @param[in] outputIndex Index of the output tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetOutputDesc(char const **desc, uint32_t *size, int32_t *type,
                           hbDNNHandle_t dnnHandle, int32_t outputIndex);

/**
 * @brief Get model description
 * 
 * @param[out] desc Address of the description information.
 * @param[out] size Size of the description information.
 * @param[out] type Type of the description information, please refer to hbDNNDescType.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetModelDesc(char const **desc, uint32_t *size, int32_t *type,
                          hbDNNHandle_t dnnHandle);

/**
 * @brief Get model compile BPU core number
 * 
 * @param[out] bpuCoreNum compile BPU core number of the model.
 * @param[in] dnnHandle DNN handle, pointing to one model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetCompileBpuCoreNum(int32_t *bpuCoreNum, hbDNNHandle_t dnnHandle);

/**
 * @brief Get hbm description
 * 
 * @param[out] desc Address of the description information.
 * @param[out] size Size of the description information.
 * @param[out] type Type of the description information, please refer to hbDNNDescType.
 * @param[in] dnnPackedHandle Horizon DNN handle, pointing to multiple models.
 * @param[in] index Index of multiple hbm models that are loaded through hbDNNInitializeFromFiles or hbDNNInitializeFromDDR, the index should be in the range of [0, modelFileCount) or [0, modelDataCount).
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetHBMDesc(char const **desc, uint32_t *size, int32_t *type,
                        hbDNNPackedHandle_t dnnPackedHandle, int32_t index);

/**
 * @brief DNN inference
 * 
 * @param[in/out] taskHandle:
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given *taskHandle is not nullptr, attach task to task handle, which represents multi model task. The given *taskHandle must be obtained through case1 and not already committed or released.
 * case3: given taskHandle is nullptr to run in sync mode with default ctrl param
 * @param[out] output Pointer to the output tensor array, the size of array should be equal to $(`hbDNNGetOutputCount`)
 * @param[in] input Input tensor array, the size of array should be equal to  $(`hbDNNGetInputCount`)
 * @param[in] dnnHandle Pointer to the dnn handle which represents model handle
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNInferV2(hbUCPTaskHandle_t *taskHandle, hbDNNTensor *output,
                     hbDNNTensor const *input, hbDNNHandle_t dnnHandle);

/**
 * @brief Get task output tensor properties. 
 * 
 * @param[out] properties Info of the output tensor.
 * @param[in] taskHandle Pointer to the ucp task handle. Dynamic output acquisition of synchronous execution tasks is not supported because taskHandle cannot be obtained.
 * @param[in] subTaskIndex Index of the sub task of the task handle.
 * @param[in] outputIndex Index of the output tensor of the model.
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNGetTaskOutputTensorProperties(hbDNNTensorProperties *properties,
                                           hbUCPTaskHandle_t taskHandle,
                                           int32_t subTaskIndex,
                                           int32_t outputIndex);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // DNN_HB_DNN_H_
