#include "timer_utils.h"

ScopeProcessTime::ScopeProcessTime(const rclcpp::Logger &logger, const std::string &name, const std::string &level)
    : name_(name), logger_(logger), level_(level), start_(std::chrono::high_resolution_clock::now()) {
}

ScopeProcessTime::~ScopeProcessTime() {
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start_).count();

  if (!name_.empty()) {
    if (level_ == "info")
      RCLCPP_INFO_STREAM(logger_,
                         "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "warn")
      RCLCPP_WARN_STREAM(logger_,
                         "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "error")
      RCLCPP_ERROR_STREAM(logger_,
                          "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "fatal")
      RCLCPP_FATAL_STREAM(logger_,
                          "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else
      RCLCPP_DEBUG_STREAM(logger_,
                          "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
  } else {
    if (level_ == "info")
      RCLCPP_INFO_STREAM(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "warn")
      RCLCPP_WARN_STREAM(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "error")
      RCLCPP_ERROR_STREAM(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else if (level_ == "fatal")
      RCLCPP_FATAL_STREAM(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
    else
      RCLCPP_DEBUG_STREAM(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
  }
}