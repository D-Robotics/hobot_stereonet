// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_RESIZE_H_
#define VP_HB_VP_RESIZE_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "hb_vp.h"

/**
 * @brief Resizes an image
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg the destination output image
 * @param[in] srcImg the source input image
 * @param[in] interpType interpolation type
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPResize(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                   hbVPImage const *srcImg, hbVPInterpolationType interpType);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_RESIZE_H_
