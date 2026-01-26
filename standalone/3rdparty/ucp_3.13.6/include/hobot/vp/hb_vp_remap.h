// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_REMAP_H_
#define VP_HB_VP_REMAP_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "hb_vp.h"

typedef void *hbVPMapWrap_t;

/**
 * vp remap param
 * mapPhyAddr: physical address of map memory info, type is double
 * mapVirAddr: virtual address of map memory info
 * srcWidth: input width
 * srcHeight: input height
 * mapWidth: map width
 * mapHeight: map height
 * interpoType: nearest or bilinear only
 * padValue: reserved field
 * borderType: reserved field
 * dataType: data type and channel num in phyAddr
*/
typedef struct hbVPRemapParam {
  uint64_t mapPhyAddr;
  void *mapVirAddr;
  int32_t srcWidth;
  int32_t srcHeight;
  int32_t mapWidth;
  int32_t mapHeight;
  int8_t interpoType;
  int8_t borderType;
  uint8_t padValue;
  uint8_t dataType;
} hbVPRemapParam;

/**
 * @brief Generate map wrap
 * 
 * @param[out] mapWrap map wrap handle, dispatched on gdc or dsp
 * @param[in] param param for mapWrap
 * @param[in] backend vp mapWrap backend, HB_UCP_CORE_ANY will create all possible payload
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPCreateMapWrap(hbVPMapWrap_t *mapWrap, hbVPRemapParam const *param,
                          uint64_t backend);

/**
 * @brief Release map wrap
 * 
 * @param[in] mapWrap vp map wrap
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbVPReleaseMapWrap(hbVPMapWrap_t mapWrap);

/**
 * @brief Applies a generic geometrical transformation to an image
 * 
 * @param[out] taskHandle return a pointer represent the task if success
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] dstImg pointer to the output image array
 * @param[in] srcImg pointer to the input image array
 * @param[in] mapWrap map wrap handle
 * @return 0 if success, return defined error code otherwise
*/
int32_t hbVPRemap(hbUCPTaskHandle_t *taskHandle, hbVPImage *dstImg,
                  hbVPImage const *srcImg, hbVPMapWrap_t mapWrap);
#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_REMAP_H_
