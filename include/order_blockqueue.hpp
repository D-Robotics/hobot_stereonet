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

#ifndef HOBOT_STEREONET_INCLUDE_ORDER_BLOCKQUEUE_HPP
#define HOBOT_STEREONET_INCLUDE_ORDER_BLOCKQUEUE_HPP

#include <mutex>
#include <map>
#include <condition_variable>

template <class T> struct order_blockqueue {
  int put(uint64_t ts, T &&t) {
    int ret;
    {
      std::lock_guard<std::mutex> lck(mtx);
      que[ts] = t;
      ret = que.size();
    }
    cv.notify_one();
    return ret;
  }

  int put(uint64_t ts, T &t) {
    int ret;
    {
      std::lock_guard<std::mutex> lck(mtx);
      que[ts] = t;
      ret = que.size();
    }
    cv.notify_one();
    return ret;
  }

  int put_silence(uint64_t ts, T &&t) {
    int ret;
    {
      std::lock_guard<std::mutex> lck(mtx);
      que[ts] = t;
      ret = que.size();
    }
    return ret;
  }

  int put_silence(uint64_t ts, T &t) {
    int ret;
    {
      std::lock_guard<std::mutex> lck(mtx);
      que[ts] = t;
      ret = que.size();
    }
    return ret;
  }

  bool get(T &t, uint32_t timeout_ms = 300) {
    {
      std::unique_lock<std::mutex> lck(mtx);
      if (!que.empty() || cv.wait_for(lck, std::chrono::milliseconds(timeout_ms), [&]() { return que.size() > 0; })) {
        auto data = que.begin();
        t = data->second;
        que.erase(data);
        return true;
      }
      return false;
    }
  }

  void pop_front() {
    {
      std::lock_guard<std::mutex> lck(mtx);
      que.erase(que.begin());
    }
  }

  void clear() {
    std::lock_guard<std::mutex> lck(mtx);
    que.clear();
  }

  unsigned int size() {
    std::lock_guard<std::mutex> lck(mtx);
    return que.size();
  }

private:
  std::condition_variable cv;
  std::mutex mtx;
  std::map<uint64_t, T> que;
};

#endif // HOBOT_STEREONET_INCLUDE_ORDER_BLOCKQUEUE_HPP
