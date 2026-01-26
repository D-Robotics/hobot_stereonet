// Copyright (c) 2024 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef PLUGIN_DSP_PLUGIN_HB_DSP_H_
#define PLUGIN_DSP_PLUGIN_HB_DSP_H_

#include <stdint.h>

#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  // Naming rules: B means begin, E means end

  // [0x0, 0x400) reserved for framework
  HB_DSP_RPC_CMD_R_B = 0x0,
  HB_DSP_RPC_CMD_PING = HB_DSP_RPC_CMD_R_B,
  HB_DSP_RPC_CMD_CONFIG = 0x01,
  HB_DSP_RPC_CMD_R_E = 0x3ff,

  // [0x400, 0x800) reserved for neural network operators
  HB_DSP_RPC_CMD_NN_B = 0x400,
  HB_DSP_RPC_CMD_NN_E = 0x7ff,

  // [0x800, 0xfff) reserved for cv operators
  HB_DSP_RPC_CMD_CV_B = 0x800,
  HB_DSP_RPC_CMD_CV_E = 0xfff,

  // [0x1000, 0x13ff) reserved for hpl operators
  HB_DSP_RPC_CMD_HPL_B = 0x1000,
  HB_DSP_RPC_CMD_HPL_E = 0x13ff,

  // [0x1400, 0xffff] for custom purpose
  HB_DSP_RPC_CMD_BUTT = 0xffff
} hbDSPRpcCmd;

/**
 * @brief create dsp rpc task
 * 
 * @param[out] taskHandle the handle of new dsp task
 * @param[in] input pointer to a custom structure for input
 * @param[in] output pointer to a custom structure for output
 * @param[in] rpcCmd custom cmd of dsp task
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDSPRpcV2(hbUCPTaskHandle_t *taskHandle, hbUCPSysMem *input,
                   hbUCPSysMem *output, int32_t rpcCmd);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // PLUGIN_DSP_PLUGIN_HB_DSP_H_
