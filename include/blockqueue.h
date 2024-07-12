//
// Created by zhy on 11/22/23.
//

#include <mutex>
#include <deque>
#include <condition_variable>
#pragma once

template<class T>
struct blockqueue {
  void put(T &&t) {
    {
      std::lock_guard<std::mutex>lck (mtx);
      que.emplace_back(t);
      cv.notify_one();
    }
  }
  void put(T &t) {
    {
      std::lock_guard<std::mutex>lck (mtx);
      que.push_back(t);
      cv.notify_one();
    }
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

