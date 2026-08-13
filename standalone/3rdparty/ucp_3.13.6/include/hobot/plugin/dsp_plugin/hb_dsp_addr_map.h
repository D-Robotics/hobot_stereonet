// Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef PLUGIN_DSP_PLUGIN_HB_DSP_ADDR_MAP_H_
#define PLUGIN_DSP_PLUGIN_HB_DSP_ADDR_MAP_H_

#include "hobot/hb_ucp_sys.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
 * @brief smmu map
 * 
 * @param[out] out only phyAddr of out that can be accessed by dsp
 * @param[in] in  smmu map in into out
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDSPAddrMap(hbUCPSysMem *out, hbUCPSysMem const *in);

/**
 * @brief smmu unmap
 *
 * @param[in] mem mapped dsp hbUCPSysMem
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDSPAddrUnmap(hbUCPSysMem const *mem);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // PLUGIN_DSP_PLUGIN_HB_DSP_ADDR_MAP_H_
