#include "timer_utils.h"

ScopeProcessTime::ScopeProcessTime(const rclcpp::Logger &logger, const std::string &name) : name_(name), logger_(logger), start_(std::chrono::high_resolution_clock::now()) {
}

ScopeProcessTime::~ScopeProcessTime() {
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double, std::milli>(end - start_).count();

  if (!name_.empty()) {
    RCLCPP_INFO_STREAM(logger_, "=> " << name_ << " time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
  } else {
    RCLCPP_INFO_STREAM(logger_, "=> time cost: " << duration << " ms, fps: " << 1 / (duration / 1000.0));
  }
}