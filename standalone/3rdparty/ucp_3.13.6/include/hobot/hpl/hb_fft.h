// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef HPL_HB_FFT_H_
#define HPL_HB_FFT_H_

#include "hb_hpl.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * fft parameter
 * pSize: select fft point
 * normalize: reserved field, normalize fft result or not, default is 0
 */
typedef struct {
  hbFFTPointSize pSize;
  int8_t normalize;
} hbFFTParam;

/**
 * @brief FFT operator on 1d complex data for type hbHPLImaginaryData
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dst destination output data
 * @param[in] src source input data
 * @param[in] param param for FFT1D
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbFFT1D(hbUCPTaskHandle_t *taskHandle, hbHPLImaginaryData *dst,
                hbHPLImaginaryData const *src, hbFFTParam const *param);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // HPL_HB_FFT_H_
