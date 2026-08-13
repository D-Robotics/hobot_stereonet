// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_ERODE_H_
#define VP_HB_VP_ERODE_H_

#include "hb_vp.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * pointLocX: reserved field
 * pointLocY: reserved field
 * iterations: reserved field
 * borderType: supports enum hbVPBorderType only
 * borderValue: reserved field
 */
typedef struct {
  int8_t pointLocX;
  int8_t pointLocY;
  int8_t iterations;
  int8_t borderType;
  uint8_t borderValue;
} hbVPErodeParam;

/**
 * @brief Erodes an image by using a specific structuring element
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg the destination output image
 * @param[in] srcImg the source input image
 * @param[in] erodeKernel kernel for Erode
 * @param[in] erodeParam param for Erode
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPErode(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                  hbVPImage const *srcImg, hbVPFilterKernel const *erodeKernel,
                  hbVPErodeParam const *erodeParam);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_ERODE_H_
