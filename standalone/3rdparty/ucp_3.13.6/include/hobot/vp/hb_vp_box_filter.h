// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_BOX_FILTER_H_
#define VP_HB_VP_BOX_FILTER_H_

#include "hb_vp.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * kernelHeight: blurring kernel height, height is odd values in the range of [3-31) and equal to width
 * kernelWidth: blurring kernel width, width is odd values in the range of [3-31) and equal to height
 * pointLocX: reserved field
 * pointLocY: reserved field
 * normalize: reserved field
 * borderType: reserved field
 */
typedef struct {
  int8_t kernelHeight;
  int8_t kernelWidth;
  int8_t pointLocX;
  int8_t pointLocY;
  int8_t normalize;
  int8_t borderType;
} hbVPBoxFilterParam;

/**
 * @brief Applies the box filter to an image
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg the destination output image
 * @param[in] srcImg the source input image
 * @param[in] boxFilterParam param for BoxFilter
 * @return 0 if success, return defined error code otherwise
*/
int32_t hbVPBoxFilter(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                      hbVPImage const *srcImg,
                      hbVPBoxFilterParam const *boxFilterParam);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_BOX_FILTER_H_
