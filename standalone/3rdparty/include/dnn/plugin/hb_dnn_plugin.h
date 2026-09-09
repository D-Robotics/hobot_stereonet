// Copyright (c) 2024，D-Robotics.
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

// Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
//
// The material in this file is confidential and contains trade secrets
// of Horizon Robotics Inc. This is proprietary information owned by
// Horizon Robotics Inc. No part of this work may be disclosed,
// reproduced, copied, transmitted, or used in any way for any purpose,
// without the express written permission of Horizon Robotics Inc.

#ifndef DNN_PLUGIN_HB_DNN_PLUGIN_H_
#define DNN_PLUGIN_HB_DNN_PLUGIN_H_

#include <string>

#include "dnn/plugin/hb_dnn_layer.h"

typedef hobot::dnn::Layer *(*hbDNNLayerCreator)();

/**
 * Register layer creator function
 * @param[in] layerName: layer type
 * @param[in] layerCreator: layer creator function
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNRegisterLayerCreator(char const *layerType,
                                  hbDNNLayerCreator layerCreator);

/**
 * Unregister layer creator function
 * @param[in] layerName: layer type
 * @return 0 if success, return defined error code otherwise
 */
int32_t hbDNNUnregisterLayerCreator(char const *layerType);

#endif  // DNN_PLUGIN_HB_DNN_PLUGIN_H_
