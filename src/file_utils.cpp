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

#include "file_utils.h"

std::vector<std::pair<std::string, std::string>> FileUtils::find_pairs(const std::string &folder_path) {
  std::vector<std::pair<std::string, std::string>> file_pairs;

  if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
    return file_pairs;
  }

  for (const auto &entry : fs::directory_iterator(folder_path)) {
    if (!entry.is_regular_file()) continue;

    auto path = entry.path();
    std::string filename = path.filename().string();
    std::string extension = path.extension().string();

    if ((extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp") &&
        filename.find("left") != std::string::npos) {
      std::string right_filename = filename;
      size_t pos = right_filename.find("left");
      right_filename.replace(pos, 4, "right");
      fs::path right_path = path.parent_path() / right_filename;
      if (fs::exists(right_path)) {
        file_pairs.emplace_back(fs::absolute(path).string(), fs::absolute(right_path).string());
      }
    }
  }

  std::sort(file_pairs.begin(), file_pairs.end(), [](const auto &a, const auto &b) {
    return fs::path(a.first).filename().string() < fs::path(b.first).filename().string();
  });

  return file_pairs;
}

void FileUtils::save_to_bin(const std::string &filename, const char *data, size_t size) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  out.write(reinterpret_cast<const char *>(data), size);
  out.close();
}

void FileUtils::save_two_to_bin(const std::string &filename, const char *data1, size_t size1, const char *data2,
                                size_t size2) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  out.write(reinterpret_cast<const char *>(data1), size1);
  out.write(reinterpret_cast<const char *>(data2), size2);

  out.close();
}

void FileUtils::save_tensor_to_txt(const std::string &filename, const int32_t *data, size_t count) {
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  for (size_t i = 0; i < count; ++i) {
    out << data[i] << " \n";
  }

  out.close();
}

bool FileUtils::read_camera_intrinsic(const std::string &filename, double &fx, double &fy, double &cx, double &cy,
                                      double &baseline) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    return false;
  }

  std::string line;
  // skip all comment lines starting with '#'
  while (std::getline(file, line)) {
    if (!line.empty() && line[0] != '#') {
      break;
    }
  }

  if (line.empty() || line[0] == '#') {
    return false;
  }

  std::istringstream iss(line);
  if (!(iss >> fx >> fy >> cx >> cy >> baseline)) {
    return false;
  }

  return true;
}