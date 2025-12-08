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

#include "timer_utils.h"

ScopeProcessTime::ScopeProcessTime(const rclcpp::Logger &logger, const std::string &name, const std::string &level)
    : name_(name), logger_(logger), level_(level), start_(std::chrono::high_resolution_clock::now()) {
}

ScopeProcessTime::~ScopeProcessTime() {
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start_).count();

  if (!name_.empty()) {
    if (level_ == "info")
      LOG_INFO(logger_,
                         "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "warn")
      LOG_WARN(logger_,
                         "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "error")
      LOG_ERROR(logger_,
                          "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "fatal")
      LOG_FATAL(logger_,
                          "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else
      LOG_DEBUG(logger_,
                          "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
  } else {
    if (level_ == "info")
      LOG_INFO(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "warn")
      LOG_WARN(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "error")
      LOG_ERROR(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "fatal")
      LOG_FATAL(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else
      LOG_DEBUG(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
  }
}