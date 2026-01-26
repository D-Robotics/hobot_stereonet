// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_THRESHOLD_H_
#define VP_HB_VP_THRESHOLD_H_

#include "hb_vp.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * HB_VP_THRESH_TOZERO: changes values less than the threshold to 0
 */
typedef enum {
  HB_VP_THRESH_TOZERO = 3,
} hbVPThresholdType;

/**
 * thresh: threshold value, value <= 255
 * maxVal: reserved field
 * type: reserved field
 */
typedef struct {
  double thresh;
  double maxVal;
  int8_t type;
} hbVPThresholdParam;

/**
 * @brief Threshold srcImg by threshold param
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg the destination output image
 * @param[in] srcImg the source input image
 * @param[in] hbVPThresholdParam param for Threshold
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPThreshold(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                      hbVPImage const *srcImg,
                      hbVPThresholdParam const *thresholdParam);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_THRESHOLD_H_
