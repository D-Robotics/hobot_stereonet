// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_CANNY_H_
#define VP_HB_VP_CANNY_H_

#include "hb_vp.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  HB_VP_NORM_L1 = 1,
} hbVPCannyNorm;

/**
 * threshold1: first threshold for hysteresis procedure
 * threshold2: second threshold for hysteresis procedure
 * kernelSize: supports 3, 5 and 7, means kernel size is 3x3, 5x5 and 7x7
 * norm: supports enum norm1
 * overlap: reserved field
 */
typedef struct {
  uint32_t threshold1;
  uint32_t threshold2;
  int8_t kernelSize;
  int8_t norm;
  int8_t overlap;
} hbVPCannyParam;

/**
 * @brief Finds edges in an image using the Canny algorithm
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg the destination output image
 * @param[in] srcImg the source input image
 * @param[in] cannyParam param for Canny
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPCanny(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                  hbVPImage const *srcImg, hbVPCannyParam const *cannyParam);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_CANNY_H_
