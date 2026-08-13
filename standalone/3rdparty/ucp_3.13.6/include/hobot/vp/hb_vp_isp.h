// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_ISP_H_
#define VP_HB_VP_ISP_H_

#include "hb_vp.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef void *hbVPISPContext;

/**
 * bufNum: Number of ISP output buffers
 * bufCached: Whether the module output buffer is cache enabled. It is only effective when isDirCon=0
 * backend: Backend ID that performs the ISP task
 * width: Raw data width
 * height: Raw data height
 */
typedef struct {
  int8_t bufNum;
  int8_t bufCached;
  uint64_t backend;
  int32_t width;
  int32_t height;
} hbVPISPCtxParam;

/**
 * @brief Create ISP Context
 * 
 * @param[out] context of ISP task
 * @param[in] ISP context param
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPCreateISPContext(hbVPISPContext *context,
                             hbVPISPCtxParam const *ispCtxParam);

/**
 * @brief Call isp hardware to process raw image and get yuv image
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[in] srcImg the source input image
 * @param[in] context of ISP
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPISP(hbUCPTaskHandle_t *taskHandle, hbVPImage const *srcImg,
                hbVPISPContext context);

/**
 * @brief Get ISP output buffer, The life cycle of the image is consistent with the task handle. 
 *  If the task handle is released, the image buffer will also become invalid.
 * @param[out] outImg the destination output image
 * @param[in] context handle of context
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPGetISPOutputBuffer(hbVPImage *outImg, hbUCPTaskHandle_t handle);

/**
 * @brief Release ISP context
 * @param[in] context of ISP
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPReleaseISPContext(hbVPISPContext context);

#ifdef __cplusplus
};
#endif  // __cplusplus

#endif  // VP_HB_VP_ISP_H_
