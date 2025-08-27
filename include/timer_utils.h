#ifndef TIMER_UTILS_H
#define TIMER_UTILS_H

#include <chrono>
#include <string>
#include "rclcpp/rclcpp.hpp"

class ScopeProcessTime {
public:
  explicit ScopeProcessTime(const rclcpp::Logger &logger, const std::string &name = "");
  ~ScopeProcessTime();

private:
  std::string name_;
  rclcpp::Logger logger_;
  std::chrono::high_resolution_clock::time_point start_;
};

#endif