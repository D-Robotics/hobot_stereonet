// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_SOBEL_H_
#define VP_HB_VP_SOBEL_H_

#include "hb_vp.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * scale: reserved field
 * delta: reserved field
 * dx: order of the derivative x
 * dy: order of the derivative y
 * kernelSize: size of the extended Sobel kernel, supports 3 and 5
 * borderType: supports enum hbVPBorderType
 */
typedef struct {
  double scale;
  double delta;
  int8_t dx;
  int8_t dy;
  int8_t kernelSize;
  int8_t borderType;
} hbVPSobelParam;

/**
 * @brief Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg the destination output image
 * @param[in] srcImg the source input image
 * @param[in] sobelParam param for Sobel
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPSobel(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                  hbVPImage const *srcImg, hbVPSobelParam const *sobelParam);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_SOBEL_H_
