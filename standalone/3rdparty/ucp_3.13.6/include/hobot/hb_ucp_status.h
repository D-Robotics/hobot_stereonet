// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef HB_UCP_STATUS_H_
#define HB_UCP_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stdint.h>

// 6 position status code: XX-XXXXX
// first 2 position is the type of status, include task, memory, backend and so on
// other position is the detail description of status
typedef enum {
  // common status
  HB_UCP_SUCCESS = 0,
  HB_UCP_INVALID_ARGUMENT = -100001,
  HB_UCP_API_USE_ERROR = -100002,
  HB_UCP_INIT_FAILED = -100003,
  HB_UCP_UNKNOWN_ERROR = -100004,
  // task error
  HB_UCP_TASK_NUMBER_EXCEED_LIMIT = -200001,
  HB_UCP_TASK_TIMEOUT = -200002,
  HB_UCP_TASK_RUN_FAILED = -200003,
  HB_UCP_TASK_HANDLE_INVALID = -200004,
  // op error
  HB_UCP_OP_NUMBER_EXCEED_LIMIT = -300001,
  HB_UCP_OP_NOT_REGISTER = -300002,
  HB_UCP_OP_CMD_UNAVAILABLE = -300003,
  // memory error
  HB_UCP_MEM_ALLOC_FAIL = -400001,
  HB_UCP_MEM_FREE_FAIL = -400002,
  HB_UCP_MEM_FLUSH_FAIL = -400003,
  HB_UCP_MEM_INVALIDATE_FAIL = -400004,
  HB_UCP_MEM_IS_INVALID = -400005,
  HB_UCP_MEM_MAP_FAIL = -400006,
  HB_UCP_MEM_UNMAP_FAIL = -400007,
  // file
  HB_UCP_FILE_OPEN_FAILED = -500001,
  // model
  HB_UCP_MODEL_NUMBER_EXCEED_LIMIT = -600001,
  HB_UCP_MODEL_INVALID = -600002,
  HB_UCP_MODEL_IS_INUSE = -600003,
  HB_UCP_MODEL_INCOMPATIBLE = -600004,
  // dsp
  HB_UCP_DSP_UNAVAILABLE = -700001,
  HB_UCP_DSP_XV_ALLOC_FAIL = -700002,
  HB_UCP_DSP_XV_FREE_FAIL = -700003,
  HB_UCP_DSP_IDMA_COPY_FAIL = -700004,
  HB_UCP_DSP_IDMA_BAD_INIT = -700005,
  HB_UCP_DSP_MMAP_FAIL = -700006,
  HB_UCP_DSP_INVALID_SCALE = -700007,
  HB_UCP_DSP_UNMAP_FAIL = -700008,
  HB_UCP_DSP_OFFLINE = -700009,
  // codec
  HB_UCP_CODEC_OPERATION_NOT_ALLOWED = -800001,
  HB_UCP_CODEC_INSUFFICIENT_RES = -800002,
  HB_UCP_CODEC_NO_FREE_INSTANCE = -800003,
  HB_UCP_CODEC_INVALID_INSTANCE = -800004,
  HB_UCP_CODEC_BUFFER_WAIT_TIMEOUT = -800005,
  HB_UCP_CODEC_IS_INUSE = -800006,
  // isp
  HB_UCP_ISP_NO_AVAILABLE_SLOT = -900001,
  HB_UCP_ISP_CTX_INUSE = -900002,
} hbUCPStatus;

/**
 * @brief Get UCP error code description
 * 
 * @param[in] errorCode error code
 * @return UCP error code description in nature language
 */
char const *hbUCPGetErrorDesc(int32_t errorCode);

}  // __cplusplus

#endif  // HB_UCP_STATUS_H_
