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