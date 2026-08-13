// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_ROTATE_H_
#define VP_HB_VP_ROTATE_H_

#include "hb_vp.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * HB_VP_ROTATE_90_CLOCKWISE: rotate 90 degrees clockwise
 * HB_VP_ROTATE_180_CLOCKWISE: rotate 180 degrees clockwise
 * HB_VP_ROTATE_90_COUNTERCLOCKWISE: rotate 90 degrees counterclockwise
 */
typedef enum {
  HB_VP_ROTATE_90_CLOCKWISE = 0,
  HB_VP_ROTATE_180_CLOCKWISE,
  HB_VP_ROTATE_90_COUNTERCLOCKWISE,
} hbVPRotateDegree;

/**
 * @brief Rotates a 2D image in multiples of 90 degrees
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg the destination output image, which has the same type and format with src, its size is calculated as follows
 *                      rotate 90 clockwise: dstImg.width = srcImg.height
 *                                           dstImg.height = srcImg.width
 *                      rotate 180 clockwise: dstImg.width = srcImg.width
 *                                            dstImg.height = srcImg.height
 *                      rotate 90 counterclockwise: dstImg.width = srcImg.height
 *                                                  dstImg.height = srcImg.width
 * @param[in] srcImg the source input image
 * @param[in] rotateCode rotate code, it should be odd value of hbVPRotateDegree
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPRotate(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                   hbVPImage const *srcImg, hbVPRotateDegree rotateCode);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_ROTATE_H_
