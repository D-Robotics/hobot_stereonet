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

#ifndef HOBOT_STEREONET_INCLUDE_LOG_MACROS_H_
#define HOBOT_STEREONET_INCLUDE_LOG_MACROS_H_

#if __has_include(<rclcpp/rclcpp.hpp>)
#define HOBOT_HAS_RCLCPP 1
#pragma message("Detected rclcpp.hpp → ROS2 logging enabled")
#else
#define HOBOT_HAS_RCLCPP 0
#pragma message("rclcpp.hpp not found → non-ROS logging enabled")
#endif

#if HOBOT_HAS_RCLCPP
#include "rclcpp/rclcpp.hpp"

// ===================== ROS2 branch =====================
// Stream style, compatible with RCLCPP_*_STREAM

#define LOG_DEBUG(logger, ...) RCLCPP_DEBUG_STREAM(logger, __VA_ARGS__)
#define LOG_INFO(logger, ...) RCLCPP_INFO_STREAM(logger, __VA_ARGS__)
#define LOG_WARN(logger, ...) RCLCPP_WARN_STREAM(logger, __VA_ARGS__)
#define LOG_ERROR(logger, ...) RCLCPP_ERROR_STREAM(logger, __VA_ARGS__)
#define LOG_FATAL(logger, ...) RCLCPP_FATAL_STREAM(logger, __VA_ARGS__)

#define LOG_DEBUG_ONCE(logger, ...) RCLCPP_DEBUG_STREAM_ONCE(logger, __VA_ARGS__)
#define LOG_INFO_ONCE(logger, ...) RCLCPP_INFO_STREAM_ONCE(logger, __VA_ARGS__)
#define LOG_WARN_ONCE(logger, ...) RCLCPP_WARN_STREAM_ONCE(logger, __VA_ARGS__)
#define LOG_ERROR_ONCE(logger, ...) RCLCPP_ERROR_STREAM_ONCE(logger, __VA_ARGS__)
#define LOG_FATAL_ONCE(logger, ...) RCLCPP_FATAL_STREAM_ONCE(logger, __VA_ARGS__)

#else // !HOBOT_HAS_RCLCPP

// ===================== non-ROS branch =====================
// Stream style, compatible with RCLCPP_*_STREAM
// logger parameter is reserved for compatibility, but not used

#include <iostream>
#include <sstream>
#include <ctime>

namespace rclcpp {
class Logger {
public:
  Logger() = default;
  ~Logger() = default;
};
} // namespace rclcpp

enum LogLevel {
  LOG_LEVEL_DEBUG = 0,
  LOG_LEVEL_INFO = 1,
  LOG_LEVEL_WARN = 2,
  LOG_LEVEL_ERROR = 3,
  LOG_LEVEL_FATAL = 4,
  LOG_LEVEL_NONE = 5,
};

inline LogLevel GLOBAL_LOG_LEVEL = LOG_LEVEL_INFO;

#define LOG_SHOULD_PRINT(level) ((level) >= GLOBAL_LOG_LEVEL)

inline const char *getCurrentTimeStr() {
  static char buf[32];
  std::time_t t = std::time(nullptr);
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
  return buf;
}

#define LOG_PRINT_STREAM(level, stream_expr)                                                                           \
  do {                                                                                                                 \
    std::ostringstream _oss;                                                                                           \
    _oss << stream_expr;                                                                                               \
    std::cout << "[" << level << "] [" << getCurrentTimeStr() << "] " << _oss.str() << std::endl;                      \
  } while (0)

#define LOG_DEBUG(logger, ...)                                                                                         \
  do {                                                                                                                 \
    if (LOG_SHOULD_PRINT(LOG_LEVEL_DEBUG)) LOG_PRINT_STREAM("DEBUG", __VA_ARGS__);                                     \
  } while (0)

#define LOG_INFO(logger, ...)                                                                                          \
  do {                                                                                                                 \
    if (LOG_SHOULD_PRINT(LOG_LEVEL_INFO)) LOG_PRINT_STREAM("INFO", __VA_ARGS__);                                       \
  } while (0)

#define LOG_WARN(logger, ...)                                                                                          \
  do {                                                                                                                 \
    if (LOG_SHOULD_PRINT(LOG_LEVEL_WARN)) LOG_PRINT_STREAM("WARN", __VA_ARGS__);                                       \
  } while (0)

#define LOG_ERROR(logger, ...)                                                                                         \
  do {                                                                                                                 \
    if (LOG_SHOULD_PRINT(LOG_LEVEL_ERROR)) LOG_PRINT_STREAM("ERROR", __VA_ARGS__);                                     \
  } while (0)

#define LOG_FATAL(logger, ...)                                                                                         \
  do {                                                                                                                 \
    if (LOG_SHOULD_PRINT(LOG_LEVEL_FATAL)) LOG_PRINT_STREAM("FATAL", __VA_ARGS__);                                     \
  } while (0)

#define LOG_DEBUG_ONCE(logger, ...)                                                                                    \
  do {                                                                                                                 \
    static bool done = false;                                                                                          \
    if (!done && LOG_SHOULD_PRINT(LOG_LEVEL_DEBUG)) {                                                                  \
      done = true;                                                                                                     \
      LOG_PRINT_STREAM("DEBUG", __VA_ARGS__);                                                                          \
    }                                                                                                                  \
  } while (0)

#define LOG_INFO_ONCE(logger, ...)                                                                                     \
  do {                                                                                                                 \
    static bool done = false;                                                                                          \
    if (!done && LOG_SHOULD_PRINT(LOG_LEVEL_INFO)) {                                                                   \
      done = true;                                                                                                     \
      LOG_PRINT_STREAM("INFO", __VA_ARGS__);                                                                           \
    }                                                                                                                  \
  } while (0)

#define LOG_WARN_ONCE(logger, ...)                                                                                     \
  do {                                                                                                                 \
    static bool done = false;                                                                                          \
    if (!done && LOG_SHOULD_PRINT(LOG_LEVEL_WARN)) {                                                                   \
      done = true;                                                                                                     \
      LOG_PRINT_STREAM("WARN", __VA_ARGS__);                                                                           \
    }                                                                                                                  \
  } while (0)

#define LOG_ERROR_ONCE(logger, ...)                                                                                    \
  do {                                                                                                                 \
    static bool done = false;                                                                                          \
    if (!done && LOG_SHOULD_PRINT(LOG_LEVEL_ERROR)) {                                                                  \
      done = true;                                                                                                     \
      LOG_PRINT_STREAM("ERROR", __VA_ARGS__);                                                                          \
    }                                                                                                                  \
  } while (0)

#define LOG_FATAL_ONCE(logger, ...)                                                                                    \
  do {                                                                                                                 \
    static bool done = false;                                                                                          \
    if (!done && LOG_SHOULD_PRINT(LOG_LEVEL_FATAL)) {                                                                  \
      done = true;                                                                                                     \
      LOG_PRINT_STREAM("FATAL", __VA_ARGS__);                                                                          \
    }                                                                                                                  \
  } while (0)

#endif

#endif // HOBOT_STEREONET_INCLUDE_LOG_MACROS_H_
