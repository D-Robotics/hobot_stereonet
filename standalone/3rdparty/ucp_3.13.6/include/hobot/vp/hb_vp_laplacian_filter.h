// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_LAPLACIAN_FILTER_H_
#define VP_HB_VP_LAPLACIAN_FILTER_H_

#include "hb_vp.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * kernelSize: only supports 1, inner kernel { 0, 1, 0, 1, -4, 1, 0, 1, 0 }
 * borderType: supports enum hbVPBorderType
 * normalize: whether normalize, non-zero represent true
 */
typedef struct {
  int8_t kernelSize;
  int8_t borderType;
  int8_t normalize;
} hbVPLaplacianFilterParam;

/**
 * @brief Applies the laplacian filter to an image
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg the destination output image, data type should be int16
 * @param[in] srcImg the source input image, data type should be uint8
 * @param[in] laplacianParam param for LaplacianFilter
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPLaplacianFilter(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                            hbVPImage const *srcImg,
                            hbVPLaplacianFilterParam const *laplacianParam);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_LAPLACIAN_FILTER_H_
