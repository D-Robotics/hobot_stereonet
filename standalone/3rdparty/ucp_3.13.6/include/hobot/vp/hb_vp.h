// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_H_
#define VP_HB_VP_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stdint.h>

#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_status.h"
#include "hobot/hb_ucp_sys.h"

/**
 * HB_VP_IMAGE_FORMAT_Y       : YYYYYYYY...
 * HB_VP_IMAGE_FORMAT_NV12    : YYYYYYYY... UVUV...
 * HB_VP_IMAGE_FORMAT_RGB_P   : RRRR...GGGG...BBBB...
 * HB_VP_IMAGE_FORMAT_RGB     : RGBRGBRGB...(C3) or RGB_RGB_RGB_...(C4)
 * HB_VP_IMAGE_FORMAT_BGR_P   : BBBB...GGGG...RRRR
 * HB_VP_IMAGE_FORMAT_BGR     : BGRBGRBGR...(C3) or BGR_BGR_BGR_...(C4)
 * HB_VP_IMAGE_FORMAT_YUV444  : YUVYUVYUV... or YUV_YUV_YUV_...
 * HB_VP_IMAGE_FORMAT_YUV444_P: YYYY...UUUU...VVVV...
 * HB_VP_IMAGE_FORMAT_YUV420  : YYYY...U...V...
 * HB_VP_IMAGE_FORMAT_RAW_RGGB_COMP_16: RGRGRG... GBGBGB... 
 * HB_VP_IMAGE_FORMAT_NV12_Y10C8_MSB: YYYYYYYY...(10bit align to 16bit) UVUV...(8bit), msb
 * HB_VP_IMAGE_FORMAT_NV12_Y12C8_MSB: YYYYYYYY...(12bit align to 16bit) UVUV...(8bit), msb
*/
typedef enum {
  HB_VP_IMAGE_FORMAT_Y = 0,
  HB_VP_IMAGE_FORMAT_NV12,
  HB_VP_IMAGE_FORMAT_RGB_P,
  HB_VP_IMAGE_FORMAT_RGB,
  HB_VP_IMAGE_FORMAT_BGR_P,
  HB_VP_IMAGE_FORMAT_BGR,
  HB_VP_IMAGE_FORMAT_YUV444,
  HB_VP_IMAGE_FORMAT_YUV444_P,
  HB_VP_IMAGE_FORMAT_YUV420,
  HB_VP_IMAGE_FORMAT_RAW_RGGB_COMP_16,
  HB_VP_IMAGE_FORMAT_NV12_Y10C8_MSB,
  HB_VP_IMAGE_FORMAT_NV12_Y12C8_MSB
} hbVPImageFormat;

/**
 * HB_VP_IMAGE_TYPE_U8C1  : 1 channel with unsigned int8 per element
 * HB_VP_IMAGE_TYPE_U8C3  : 3 channel with unsigned int8 per element
 * HB_VP_IMAGE_TYPE_U8C4  : 4 channel with unsigned int8 per element
 * HB_VP_IMAGE_TYPE_S16C1 : 1 channel with signed int16 (short) per element
 * HB_VP_IMAGE_TYPE_S16C2 : 2 channel with signed int16 (short) per element
 * HB_VP_IMAGE_TYPE_S16C3 : 3 channel with signed int16 (short) per element
 * HB_VP_IMAGE_TYPE_S32C1 : 1 channel with signed int32 per element
 * HB_VP_IMAGE_TYPE_F32C1 : 1 channel with float32 per element
 * HB_VP_IMAGE_TYPE_F64C1 : 1 channel with float64 per element
 * HB_VP_IMAGE_TYPE_F64C2 : 2 channel with float64 per element
 * HB_VP_IMAGE_TYPE_U10C1 : 1 channel with unsigned int10 per element 
 * HB_VP_IMAGE_TYPE_U12C1 : 1 channel with unsigned int12 per element
 * HB_VP_IMAGE_TYPE_U16C1 : 1 channel with unsigned int16 per element
 * image depth and channel .eg: For HB_VP_IMAGE_FORMAT_RGB, channel = 3
 * which imageType should be HB_VP_IMAGE_TYPE_U8C3, channel may > 3 which
 * means there exist extra space for alignment requirements, imageType
 * should be HB_VP_IMAGE_TYPE_U8C4 (RRR_ GGG_ BBB_).
 * Need to sync to dsp data type (/backends/dsp/hobot/src/cv/core.h) while update
 */
typedef enum {
  HB_VP_IMAGE_TYPE_U8C1 = 0,
  HB_VP_IMAGE_TYPE_U8C3,
  HB_VP_IMAGE_TYPE_U8C4,
  HB_VP_IMAGE_TYPE_S16C1,
  HB_VP_IMAGE_TYPE_S16C2,
  HB_VP_IMAGE_TYPE_S16C3,
  HB_VP_IMAGE_TYPE_S32C1,
  HB_VP_IMAGE_TYPE_F32C1,
  HB_VP_IMAGE_TYPE_F64C1,
  HB_VP_IMAGE_TYPE_F64C2,
  HB_VP_IMAGE_TYPE_U10C1,
  HB_VP_IMAGE_TYPE_U12C1,
  HB_VP_IMAGE_TYPE_U16C1
} hbVPImageType;

/**
 * imageFormat: image format
 * imageType: image element type
 * width: image width
 * height: image height
 * stride: bit width of per column
 * dataVirAddr: image virtual address
 * dataPhyAddr: image physical address
 * uvVirAddr: image virtual address of UV data
 * uvPhyAddr: image physical address of UV data
 * uvStride: bit width of UV data per column
 */
typedef struct hbVPImage {
  uint8_t imageFormat;
  uint8_t imageType;
  int32_t width;
  int32_t height;
  int32_t stride;
  void *dataVirAddr;
  uint64_t dataPhyAddr;
  void *uvVirAddr;
  uint64_t uvPhyAddr;
  int32_t uvStride;
} hbVPImage;

/**
 * A contiguous block of memory
 * phyAddr: physical address of memory
 * virAddr: virtual address of memory
 * memSize: actual memory size alloced
 * size: numbers of element of array
 */
typedef struct hbVPArray {
  uint64_t phyAddr;
  void *virAddr;
  uint32_t memSize;
  uint32_t size;
} hbVPArray;

/**
 * left: left edge of roi area
 * top: top edge of roi area
 * right: right edge of roi area
 * bottom: bottom edge of roi area
 */
typedef struct hbVPRoi {
  int32_t left;
  int32_t top;
  int32_t right;
  int32_t bottom;
} hbVPRoi;

/**
 * 2D filter kernel of image process
 * dataType: element type of kernel, refer to hbVPImageType
 * width: width of filter kernel
 * height: height of filter kernel
 * dataPhyAddr: physical address of filter kernel
 * datavirAddr: virtual address of memory
 */
typedef struct hbVPFilterKernel {
  uint8_t dataType;
  int32_t width;
  int32_t height;
  uint64_t dataPhyAddr;
  void *dataVirAddr;
} hbVPFilterKernel;

/**
 * HB_VP_INTER_NEAREST: nearest neighbor interpolation
 * HB_VP_INTER_LINEAR: bilinear interpolation
 * HB_VP_INTER_GAUSSIAN: gaussian interpolation
 * */
typedef enum {
  HB_VP_INTER_NEAREST = 0,
  HB_VP_INTER_LINEAR = 1,
  HB_VP_INTER_GAUSSIAN = 2
} hbVPInterpolationType;

/**
 * HB_VP_BORDER_CONSTANT: `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
 * HB_VP_BORDER_REPLICATE: `aaaaaa|abcdefgh|hhhhhhh`
 * */
typedef enum {
  HB_VP_BORDER_CONSTANT = 0,
  HB_VP_BORDER_REPLICATE = 1,
} hbVPBorderType;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_H_
