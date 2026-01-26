// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef HPL_HB_HPL_H_
#define HPL_HB_HPL_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stdint.h>

#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_status.h"
#include "hobot/hb_ucp_sys.h"

#define HB_HPL_MAX_DIMS (3)

typedef enum {
  HB_HPL_DATA_TYPE_S16 = 0,
  HB_HPL_DATA_TYPE_S32,
  HB_HPL_DATA_TYPE_F32
} hbHPLDataType;

/**
 * format of data of real part and imaginary part
 * HB_IM_FORMAT_INTERLEAVED: R I R I...
 *    real data imaginary data mixed, only one pointer to address is valid
 * HB_IM_FORMAT_SEPARATE:  R R...    I I...
 *    real data and imaginary data saved in different pointer, two pointer is valid
 */
typedef enum {
  HB_IM_FORMAT_INTERLEAVED = 0,
  HB_IM_FORMAT_SEPARATE
} hbHPLImFormat;

/**
 * struct of imaginary
 * realDataVirAddr: real data virtual address
 * realDataPhyAddr: real data physical address
 * imDataVirAddr: imaginary data virtual address
 * imDataPhyAddr: imaginary data physical address
 * dataType: data type of imaginary real part and imaginary part
 * imFormat: format of data of real part and imaginary part, only real data address valid when format is HB_IM_FORMAT_INTERLEAVED
 * dimensionSize: depth of per channel
 * numDimensionSize: the amount of data channel
*/
typedef struct {
  void* realDataVirAddr;
  uint64_t realDataPhyAddr;
  void* imDataVirAddr;
  uint64_t imDataPhyAddr;
  hbHPLDataType dataType;
  hbHPLImFormat imFormat;
  int32_t dimensionSize[HB_HPL_MAX_DIMS];
  int32_t numDimensionSize;
} hbHPLImaginaryData;

typedef enum {
  HB_FFT_POINT_SIZE_16 = 0,
  HB_FFT_POINT_SIZE_32,
  HB_FFT_POINT_SIZE_64,
  HB_FFT_POINT_SIZE_128,
  HB_FFT_POINT_SIZE_256,
  HB_FFT_POINT_SIZE_512,
  HB_FFT_POINT_SIZE_1024
} hbFFTPointSize;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // HPL_HB_HPL_H_
