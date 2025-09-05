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

#ifndef HOBOT_STEREONET_INCLUDE_FILE_UTILS_H_
#define HOBOT_STEREONET_INCLUDE_FILE_UTILS_H_

#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;

class FileUtils {
public:
  // delete the default constructor
  FileUtils() = delete;

  // utility functions
  static std::vector<std::pair<std::string, std::string>> find_pairs(const std::string &folder_path);
  static void save_to_bin(const std::string &filename, const char *data, size_t size);
  static void save_two_to_bin(const std::string &filename, const char *data1, size_t size1, const char *data2,
                              size_t size2);
  static void save_tensor_to_txt(const std::string &filename, const int32_t *data, size_t count);
  static bool read_camera_intrinsic(const std::string &filename, double &fx, double &fy, double &cx, double &cy,
                                    double &baseline);
};

#endif // HOBOT_STEREONET_INCLUDE_FILE_UTILS_H_