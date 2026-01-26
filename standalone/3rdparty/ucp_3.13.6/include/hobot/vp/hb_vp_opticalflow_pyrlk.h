// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef VP_HB_VP_OPTICALFLOW_PYRLK_H_
#define VP_HB_VP_OPTICALFLOW_PYRLK_H_

#include "hb_vp.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define HB_VP_INITIALIZE_OPTICAL_FLOW_PARAM(param) \
  {                                                \
    (param)->pyrLevels = 5;                        \
    (param)->winSize = 7;                          \
    (param)->criteriaEpsilon = 0;                  \
    (param)->maxIteration = 10;                    \
    (param)->minEigThreshold = 1e-4;               \
  }

/**
 * Define the key points
 * x: x coordinate
 * y: y coordinate
 */
typedef struct hbVPKeyPoint {
  float x;
  float y;
} hbVPKeyPoint;

/**
 * Define OpticalFlowPyrLK parameter
 * pyrLevels: gaussian pyramid levels use.
 * winSize: window size
 * maxIteration: iteration times
 * criteriaEpsilon: termination threshold
 * minEigThreshold：minimum eigenvalue threshold
 * confEnable: enable optical flow confidence
*/
typedef struct hbVPLKOFParam {
  uint32_t pyrLevels;
  uint32_t winSize;
  uint32_t maxIteration;
  float criteriaEpsilon;
  float minEigThreshold;
  uint8_t confEnable;
} hbVPLKOFParam;

/**
 * @brief Optical flow calculation
 * For pyramid image input, the address of the i layer needs to be 
 * greater than the i-1 layer to ensure that the memory will not overlap. 
 * 
 * @param[out] taskHandle handle of task
 * case1: given *taskHandle is nullptr, create new task handle
 * case2: given taskHandle is nullptr to run in sync mode
 * @param[out] currPoints pointer to the output key points array
 * @param[out] currPointsStatus output status vector, data type should be unsigned char
 * @param[out] currPointsConf optical flow confidence
 * case1: j5 environment, data type should be float
 * case2: j6 environment, should be nullptr
 * @param[in] prevPoints pointer to the previous key points array
 * @param[in] currPym pointer to current gaussian pyramid image array
 * @param[in] prevPym pointer to previous gaussian pyramid image array 
 * @param[in] lkofParam param for OpticalFlowPyrLK
 * @return 0 if success, return defined error code otherwise
*/

int32_t hbVPOpticalFlowPyrLK(hbUCPTaskHandle_t *taskHandle,
                             hbVPArray *currPoints, hbVPArray *currPointsStatus,
                             hbVPArray *currPointsConf,
                             hbVPArray const *prevPoints,
                             hbVPImage const *currPym, hbVPImage const *prevPym,
                             hbVPLKOFParam const *lkofParam);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // VP_HB_VP_OPTICALFLOW_PYRLK_H_
