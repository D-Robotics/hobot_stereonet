// Copyright (c) 2025，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HOBOT_STEREONET_INCLUDE_DNN_PLATFORM_H_
#define HOBOT_STEREONET_INCLUDE_DNN_PLATFORM_H_

#if defined(PLATFORM_S100) || defined(PLATFORM_S600)

#include "hobot/dnn/hb_dnn.h"
#include "hobot/dnn/hb_dnn_status.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#define ALIGN(value, alignment) (((value) + ((alignment) - 1)) & ~((alignment) - 1))
#define ALIGN_32(value) ALIGN(value, 32)

#define TENSOR_SYSMEM(x, y) x.sysMem

using hbPackedDNNHandle_t = hbDNNPackedHandle_t;
using hbDNNTaskHandle_t = hbUCPTaskHandle_t;
using hbDNNInferCtrlParam = hbUCPSchedParam;

using hbSysMem = hbUCPSysMem;

#define HB_DNN_INITIALIZE_INFER_CTRL_PARAM HB_UCP_INITIALIZE_SCHED_PARAM

static int hbSysAllocCachedMem(hbSysMem *mem, uint32_t size) {
  return hbUCPMallocCached(mem, size, 0);
}

static int hbSysWriteMem(hbSysMem *dest, char *src, uint64_t size) {
  memcpy(dest->virAddr, src, size);
  return HB_DNN_SUCCESS;
}

static int hbSysFlushMem(hbSysMem const *mem, int32_t flag) {
  return hbUCPMemFlush(mem, flag);
}

static int hbSysFreeMem(hbSysMem *mem) {
  return hbUCPFree(mem);
}

static int hbDNNReleaseTask(hbDNNTaskHandle_t taskHandle) {
  return hbUCPReleaseTask(taskHandle);
}

static int hbDNNInfer(hbDNNTaskHandle_t *taskHandle, hbDNNTensor **output, hbDNNTensor const *input, hbDNNHandle_t dnnHandle, hbDNNInferCtrlParam *inferCtrlParam) {
  int ret;
  ret = hbDNNInferV2(taskHandle, *output, input, dnnHandle);
  if (ret != HB_DNN_SUCCESS) {
    std::cout << "hbDNNInferV2 failed: " << ret << std::endl;
    return ret;
  }
  inferCtrlParam->backend = HB_UCP_BPU_CORE_ANY;
  ret = hbUCPSubmitTask(*taskHandle, inferCtrlParam);
  if (ret != HB_DNN_SUCCESS) {
    std::cout << "hbUCPSubmitTask failed: " << ret << std::endl;
    return ret;
  }
  return ret;
}

static int hbDNNWaitTaskDone(hbDNNTaskHandle_t taskHandle, int32_t timeout) {
  return hbUCPWaitTaskDone(taskHandle, timeout);
}

static void hbGetInputTensorHW(const hbDNNTensorProperties &properties, int32_t &height, int32_t &width) {
  switch (properties.quantizeAxis) {
  case 3: // NHWC
    height = properties.validShape.dimensionSize[1];
    width = properties.validShape.dimensionSize[2];
    break;
  case 1: // NCHW
    height = properties.validShape.dimensionSize[2];
    width = properties.validShape.dimensionSize[3];
    break;
  case 0: std::cout << "unknow properties.quantizeAxis for 0!" << std::endl; return;
  }
}

#endif

#ifdef PLATFORM_X5

#include <dnn/hb_dnn.h>

#define TENSOR_SYSMEM(x, y) x.sysMem[y]

static void hbGetInputTensorHW(const hbDNNTensorProperties &properties, int32_t &height, int32_t &width) {
  switch (properties.tensorLayout) {
  case HB_DNN_LAYOUT_NHWC:
    height = properties.validShape.dimensionSize[1];
    width = properties.validShape.dimensionSize[2];
    break;
  case HB_DNN_LAYOUT_NCHW:
    height = properties.validShape.dimensionSize[2];
    width = properties.validShape.dimensionSize[3];
    break;
  }
}

#endif

#endif // HOBOT_STEREONET_INCLUDE_DNN_PLATFORM_H_
