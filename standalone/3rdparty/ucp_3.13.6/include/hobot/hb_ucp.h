// Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef HB_UCP_H_
#define HB_UCP_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stdint.h>

#define HB_UCP_VERSION_MAJOR (3U)
#define HB_UCP_VERSION_MINOR (13U)
#define HB_UCP_VERSION_PATCH (6U)

typedef void *hbUCPTaskHandle_t;

typedef void (*hbUCPTaskDoneCb)(hbUCPTaskHandle_t taskHandle, int32_t status,
                                void *userdata);

#define HB_UCP_ALL_BACKENDS                                                \
  (HB_UCP_CORE_ANY | HB_UCP_BPU_CORE_0 | HB_UCP_BPU_CORE_1 |               \
   HB_UCP_BPU_CORE_2 | HB_UCP_BPU_CORE_3 | HB_UCP_BPU_CORE_ANY |           \
   HB_UCP_BPU_CORE_ANY | HB_UCP_DSP_CORE_0 | HB_UCP_DSP_CORE_1 |           \
   HB_UCP_DSP_CORE_ANY | HB_UCP_GDC_CORE_0 | HB_UCP_GDC_CORE_1 |           \
   HB_UCP_GDC_CORE_ANY | HB_UCP_STITCH_CORE_0 | HB_UCP_LKOF_CORE_0 |       \
   HB_UCP_JPU_CORE_0 | HB_UCP_JPU_CORE_1 | HB_UCP_JPU_CORE_2 |             \
   HB_UCP_JPU_CORE_ANY | HB_UCP_VPU_CORE_0 | HB_UCP_VPU_CORE_1 |           \
   HB_UCP_VPU_CORE_2 | HB_UCP_VPU_CORE_ANY | HB_UCP_PYRAMID_CORE_0 |       \
   HB_UCP_PYRAMID_CORE_1 | HB_UCP_PYRAMID_CORE_2 | HB_UCP_PYRAMID_CORE_3 | \
   HB_UCP_PYRAMID_CORE_4 | HB_UCP_PYRAMID_CORE_ANY | HB_UCP_ISP_CORE_0 |   \
   HB_UCP_ISP_CORE_1 | HB_UCP_ISP_CORE_2 | HB_UCP_ISP_CORE_3)

#define HB_UCP_CORE_ANY (0ULL << 0)
#define HB_UCP_BPU_CORE_0 (1ULL << 0)
#define HB_UCP_BPU_CORE_1 (1ULL << 1)
#define HB_UCP_BPU_CORE_2 (1ULL << 2)
#define HB_UCP_BPU_CORE_3 (1ULL << 3)
#define HB_UCP_BPU_CORE_ANY (1ULL << 7)
#define HB_UCP_DSP_CORE_0 (1ULL << 8)
#define HB_UCP_DSP_CORE_1 (1ULL << 9)
#define HB_UCP_DSP_CORE_ANY (1ULL << 15)
#define HB_UCP_GDC_CORE_0 (1ULL << 16)
#define HB_UCP_GDC_CORE_1 (1ULL << 17)
#define HB_UCP_GDC_CORE_ANY (1ULL << 19)
#define HB_UCP_STITCH_CORE_0 (1ULL << 20)
#define HB_UCP_LKOF_CORE_0 (1ULL << 24)
#define HB_UCP_JPU_CORE_0 (1ULL << 25)
#define HB_UCP_JPU_CORE_1 (1ULL << 26)
#define HB_UCP_JPU_CORE_2 (1ULL << 27)
#define HB_UCP_JPU_CORE_ANY (1ULL << 29)
#define HB_UCP_VPU_CORE_0 (1ULL << 30)
#define HB_UCP_VPU_CORE_1 (1ULL << 31)
#define HB_UCP_VPU_CORE_2 (1ULL << 32)
#define HB_UCP_VPU_CORE_ANY (1ULL << 33)
#define HB_UCP_PYRAMID_CORE_0 (1ULL << 35)
#define HB_UCP_PYRAMID_CORE_1 (1ULL << 36)
#define HB_UCP_PYRAMID_CORE_2 (1ULL << 37)
#define HB_UCP_PYRAMID_CORE_3 (1ULL << 38)
#define HB_UCP_PYRAMID_CORE_4 (1ULL << 39)
#define HB_UCP_PYRAMID_CORE_ANY (1ULL << 40)
#define HB_UCP_ISP_CORE_0 (1ULL << 42)
#define HB_UCP_ISP_CORE_1 (1ULL << 43)
#define HB_UCP_ISP_CORE_2 (1ULL << 44)
#define HB_UCP_ISP_CORE_3 (1ULL << 45)

typedef enum hbUCPTaskPriority {
  HB_UCP_PRIORITY_LOWEST = 0,
  HB_UCP_PRIORITY_HIGHEST = 255
} hbUCPTaskPriority;

#define HB_UCP_INITIALIZE_SCHED_PARAM(param)    \
  {                                             \
    (param)->priority = HB_UCP_PRIORITY_LOWEST; \
    (param)->deviceId = 0U;                     \
    (param)->customId = 0;                      \
    (param)->backend = HB_UCP_CORE_ANY;         \
  }

/**
 * priority: task priority
 * customId: custom priority
 * backend: conduction hardware
 * 00 ... 00 0000 0000 0000 0000 0000 0000 
 * |   ...  |   GDC   |   DSP   |  BPU   |
 * deviceId: device id
*/
typedef struct hbUCPSchedParam {
  int32_t priority;
  int64_t customId;
  uint64_t backend;
  uint32_t deviceId;
} hbUCPSchedParam;

/**
 * @brief Get api version
 * 
 * @return const char pointer, which represents the ucp version
 */
const char *hbUCPGetVersion();

/**
 * @brief Get current running soc name
 * 
 * @return const char pointer, which represents the soc name
 */
const char *hbUCPGetSocName();

/**
 * @brief Submit task to Unified Computing Platform with scheduling parameters
 * 
 * @param[in] taskHandle pointer to the task
 * @param[in] schedParam task schedule parameter
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbUCPSubmitTask(hbUCPTaskHandle_t taskHandle,
                        hbUCPSchedParam *schedParam);

/**
 * @brief Wait util task completed.
 * 
 * @param[in] taskHandle pointer to the task
 * @param[in] timeout timeout for waiting task, unit is milliseconds(ms)
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbUCPWaitTaskDone(hbUCPTaskHandle_t taskHandle, int32_t timeout);

/**
 * @brief Set task done callback
 * 
 * @param[in] taskHandle pointer to the task
 * @param[in] taskDoneCb callback function
 * @param[in] userdata userdata 
 * @return 0 if success, return defined error code otherwise
*/
int32_t hbUCPSetTaskDoneCb(hbUCPTaskHandle_t taskHandle,
                           hbUCPTaskDoneCb taskDoneCb, void *userdata);

/**
 * @brief Release a task and its related resources.
 * 
 * @param[in] taskHandle pointer to the task
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbUCPReleaseTask(hbUCPTaskHandle_t taskHandle);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // HB_UCP_H_
