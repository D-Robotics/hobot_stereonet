
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

#ifndef HOBOT_STEREONET_INCLUDE_PERFORMANCE_RECORD_H_
#define HOBOT_STEREONET_INCLUDE_PERFORMANCE_RECORD_H_

#include <unistd.h>
#include <memory>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <sstream>

struct performance_writer {
  performance_writer() {
    record_thread_ = std::make_shared<std::thread>(std::bind(&performance_writer::record, this));
  }

  ~performance_writer() {
    is_running_ = false;
    cd_.notify_all();
    if (record_thread_ && record_thread_->joinable()) {
      record_thread_->join();
    }
  }

  void record_performance(int latency) {
    static auto last_calculation = std::chrono::system_clock::now();

    auto current = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current - last_calculation).count();

    fps_ += 1.0;
    latency_ = latency;

    if (duration >= 1000) {
      true_fps_ = fps_ / (duration / 1000.0);
      fps_ = 0.0;
      cd_.notify_one();
      last_calculation = current;
    }
  }

  double get_fps() {
    return true_fps_;
  }

  int get_cpu_usage() {
    return cpu_usage_;
  }

  int get_bpu_usage() {
    return bpu_ratio_;
  }

  static std::shared_ptr<performance_writer> Get() {
    static std::shared_ptr<performance_writer> instance = nullptr;
    if (instance == nullptr) {
      instance = std::make_shared<performance_writer>();
    }
    return instance;
  }

private:
  std::atomic_bool is_running_{true};

  double fps_ = 0.0;
  double true_fps_ = 0.0;

  std::atomic_uint latency_{0};
  std::atomic_uint bpu_ratio_{0};
  std::atomic_uint cpu_usage_{0};

  std::shared_ptr<std::thread> record_thread_ = nullptr;

  std::mutex mtx_;
  std::condition_variable cd_;

private:
  void record() {
    static pid_t pid = getpid();
    char buffer[128] = {0};

    static std::string pid_str = std::to_string(pid);
    static std::string cmd =
        "top -b -n 1 -p " + pid_str +
        " | tail -n 2 "
        "| awk '/^ *PID/ {for (i=1; i<=NF; i++) {if ($i==\"%CPU\") cpu_col=i}} NR>1 {print $cpu_col}'";

    while (is_running_) {
      std::unique_lock<std::mutex> lock(mtx_);
      cd_.wait(lock);

      FILE *fp = popen(cmd.c_str(), "r");
      if (fp == nullptr) {
        continue;
      }

      int ret = fread(buffer, sizeof(char), sizeof(buffer), fp);
      if (ret <= 0) {
        pclose(fp);
        continue;
      }

      buffer[ret - 1] = '0';
      pclose(fp);

      std::stringstream temp;
      std::ifstream bpu_ratio("/sys/devices/system/bpu/bpu0/ratio", std::ios::in);

      if (bpu_ratio.is_open()) {
        temp << bpu_ratio.rdbuf();
      } else {
        temp << "0\n";
      }

      cpu_usage_ = std::atoi(buffer);
      bpu_ratio_ = std::atoi(temp.str().c_str());
    }
  }
};

#endif // HOBOT_STEREONET_INCLUDE_PERFORMANCE_RECORD_H_
