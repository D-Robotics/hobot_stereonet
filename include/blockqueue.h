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

#include <mutex>
#include <deque>
#include <condition_variable>
#pragma once

template<class T>
struct blockqueue {
  int put(T &&t) {
    int ret;
    {
      std::lock_guard<std::mutex>lck (mtx);
      que.emplace_back(t);
      ret = que.size();
    }
    cv.notify_one();
    return ret;
  }
  int put(T &t) {
     int ret; 
    {
      std::lock_guard<std::mutex>lck (mtx);
      que.push_back(t);
      ret = que.size();
    }
    cv.notify_one();
    return ret;
  }

  bool get(T &t, uint32_t timeout_ms = 300) {
    {
      std::unique_lock<std::mutex>lck (mtx);
      if (!que.empty() || cv.wait_for(
              lck, std::chrono::milliseconds(timeout_ms),
              [&]() {auto sz = que.size();
              //printf("sz:%d\n", sz);
              return sz > 0;})) {
        t = que.front();
        que.pop_front();
        return true;
      }
      return false;
    }
  }

  void pop_front() {
    {
      std::lock_guard<std::mutex>lck (mtx);
      que.pop_front();
     }
  }

  void clear() {
    std::lock_guard<std::mutex>lck (mtx);
    que.clear();
  }

  uint size() {
    std::lock_guard<std::mutex>lck (mtx);
    return que.size();
  }

private:
  std::condition_variable cv;
  std::mutex mtx;
  std::deque<T> que;
};

