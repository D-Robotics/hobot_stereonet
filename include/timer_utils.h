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

#ifndef HOBOT_STEREONET_INCLUDE_TIMER_UTILS_H_
#define HOBOT_STEREONET_INCLUDE_TIMER_UTILS_H_

#include <chrono>
#include <string>
#include "log_macros.h"

class ScopeProcessTime {
public:
  explicit ScopeProcessTime(const rclcpp::Logger &logger, const std::string &name = "",
                            const std::string &level = "debug");
  ~ScopeProcessTime();

private:
  std::string name_;
  rclcpp::Logger logger_;
  std::string level_;
  std::chrono::high_resolution_clock::time_point start_;
};

#endif // HOBOT_STEREONET_INCLUDE_TIMER_UTILS_H_