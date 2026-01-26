// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_ROI_RESIZE_H_
#define VP_HB_VP_ROI_RESIZE_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "hb_vp.h"

/**
 * interpolation: supports hbVPInterpolationFlags
 * paddingValue: padding value, order by channel sequence, default is 0
 */
typedef struct {
  int8_t interpolation;
  uint32_t paddingValue[4];
} hbVPRoiResizeParam;

/**
 * @brief Crop roi area and then resize and padding it with same width and height scale
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg the destination output image
 * @param[in] srcImg the source input image
 * @param[in] roi roi area
 * @param[in] roiResizeParam param for RoiResize
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPRoiResize(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                      hbVPImage const *srcImg, hbVPRoi const *roi,
                      hbVPRoiResizeParam const *roiResizeParam);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_ROI_RESIZE_H_
