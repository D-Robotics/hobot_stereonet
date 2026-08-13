// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_HB_DNN_STATUS_H_
#define DNN_HB_DNN_STATUS_H_

#include <stdint.h>

#include "hobot/hb_ucp_status.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  HB_DNN_SUCCESS = HB_UCP_SUCCESS,
  HB_DNN_INVALID_ARGUMENT = HB_UCP_INVALID_ARGUMENT,
  HB_DNN_INVALID_MODEL = HB_UCP_MODEL_INVALID,
  HB_DNN_MODEL_NUMBER_EXCEED_LIMIT = HB_UCP_MODEL_NUMBER_EXCEED_LIMIT,
  HB_DNN_INVALID_PACKED_DNN_HANDLE = HB_UCP_INVALID_ARGUMENT,
  HB_DNN_INVALID_DNN_HANDLE = HB_UCP_INVALID_ARGUMENT,
  HB_DNN_CAN_NOT_OPEN_FILE = HB_UCP_FILE_OPEN_FAILED,
  HB_DNN_OUT_OF_MEMORY = HB_UCP_MEM_ALLOC_FAIL,
  HB_DNN_TIMEOUT = HB_UCP_TASK_TIMEOUT,
  HB_DNN_TASK_NUM_EXCEED_LIMIT = HB_UCP_TASK_NUMBER_EXCEED_LIMIT,
  HB_DNN_TASK_BATCH_SIZE_EXCEED_LIMIT = HB_UCP_TASK_NUMBER_EXCEED_LIMIT,
  HB_DNN_INVALID_TASK_HANDLE = HB_UCP_TASK_HANDLE_INVALID,
  HB_DNN_RUN_TASK_FAILED = HB_UCP_TASK_RUN_FAILED,
  HB_DNN_MODEL_IS_RUNNING = HB_UCP_MODEL_IS_INUSE,
  HB_DNN_INCOMPATIBLE_MODEL = HB_UCP_MODEL_INCOMPATIBLE,
  HB_DNN_API_USE_ERROR = HB_UCP_API_USE_ERROR,
  HB_DNN_MULTI_PROGRESS_USE_ERROR = HB_UCP_API_USE_ERROR
} hbDNNStatus;

/**
 * @brief Get DNN error code description
 * 
 * @param[in] errorCode dnn error code
 * @return DNN error code description in nature language
 */
char const *hbDNNGetErrorDesc(int32_t errorCode);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // DNN_HB_DNN_STATUS_H_
