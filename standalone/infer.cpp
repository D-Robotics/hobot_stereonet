
// Copyright (c) 2025,D-Robotics.
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

#include <algorithm>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

// =============== stereonet ===============
#include "log_macros.h"
#include "camera_intrinsic.h"
#include "stereonet_process.h"
#include "img_convert_utils.h"
#include "file_utils.h"
#include "feature_epipolar_align.h"
// =============== stereonet ===============

namespace fs = std::filesystem;

bool readCameraIntrinsicFromFile(const std::string &file_path, stereonet::CameraIntrinsic &intrinsic) {
  std::ifstream infile(file_path);
  if (!infile.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return false;
  }

  std::vector<double> values;
  std::string line;

  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream ss(line);
    double v;
    while (ss >> v) {
      values.push_back(v);
    }
  }

  // ---------- format 1 ----------
  // fx fy cx cy baseline
  if (values.size() == 5) {
    intrinsic.fx = values[0];
    intrinsic.fy = values[1];
    intrinsic.cx = values[2];
    intrinsic.cy = values[3];
    intrinsic.baseline = values[4];
    return intrinsic.is_valid();
  }

  // ---------- format 2 ----------
  // 3x3 K + baseline
  // fx 0 cx
  // 0 fy cy
  // 0 0 1
  // baseline
  if (values.size() == 10) {
    intrinsic.fx = values[0];
    intrinsic.cx = values[2];
    intrinsic.fy = values[4];
    intrinsic.cy = values[5];
    intrinsic.baseline = values[9];
    return intrinsic.is_valid();
  }

  std::cerr << "Unsupported intrinsic format in: " << file_path << std::endl;
  return false;
}

void saveCameraIntrinsic(const std::string &dir, const stereonet::CameraIntrinsic &intr) {
  fs::create_directories(dir);

  // camera_intrinsic.txt
  std::string file1 = dir + "/camera_intrinsic.txt";
  std::ofstream f1(file1);
  f1 << "# fx fy cx cy baseline(m)" << std::endl;
  f1 << intr.fx << " " << intr.fy << " " << intr.cx << " " << intr.cy << " " << intr.baseline << std::endl;
  f1.close();

  // K.txt
  std::string file2 = dir + "/K.txt";
  std::ofstream f2(file2);
  f2 << intr.fx << " 0.0 " << intr.cx << " 0.0 " << intr.fy << " " << intr.cy << " 0.0 0.0 1.0" << std::endl;
  f2 << intr.baseline << std::endl;
  f2.close();
}

void print_help(const char *prog_name) {
  std::cout << R"(Usage: )" << prog_name << R"( [model_path] [local_img_dir] [uncertainty_th]

Arguments:
  model_path         Path to stereo model (.bin)
                     default: ./model/DStereoV2.4_int16.bin
  local_img_dir      Path to local image directory
                     default: ./img
  uncertainty_th     Uncertainty threshold
                     default: -0.10

Examples:
  )" << prog_name
            << R"( ./model/DStereoV2.4_int16.bin ./img -0.10
)";
}

bool hasIntrinsicFile(const std::string &dir) {
  return fs::exists(fs::path(dir) / "camera_intrinsic.txt") || fs::exists(fs::path(dir) / "K.txt");
}

bool hasImagePairs(const std::string &dir) {
  std::vector<std::pair<std::string, std::string>> img_pairs = FileUtils::find_pairs(dir);
  return !img_pairs.empty();
}

bool hasSingleImages(const std::string &dir) {
  return !FileUtils::find_images(dir).empty();
}

bool hasLeftRightDirs(const std::string &dir) {
  fs::path left_dir = fs::path(dir) / "left";
  fs::path right_dir = fs::path(dir) / "right";
  if (!fs::is_directory(left_dir) || !fs::is_directory(right_dir)) return false;

  // both must contain only image files (no subdirectories)
  auto has_only_images = [](const fs::path &p) {
    if (!fs::is_directory(p)) return false;
    for (const auto &entry : fs::directory_iterator(p)) {
      if (entry.is_directory()) return false;
      if (entry.is_regular_file() && !FileUtils::is_image_file(entry.path().extension().string())) return false;
    }
    return true;
  };

  return has_only_images(left_dir) && has_only_images(right_dir);
}

std::vector<std::pair<std::string, std::string>> findLeftRightPairs(const std::string &dir) {
  std::vector<std::pair<std::string, std::string>> pairs;
  fs::path left_dir = fs::path(dir) / "left";
  fs::path right_dir = fs::path(dir) / "right";

  if (!fs::is_directory(left_dir) || !fs::is_directory(right_dir)) return pairs;

  // collect left images indexed by stem
  std::map<std::string, std::string> left_map;
  for (const auto &entry : fs::directory_iterator(left_dir)) {
    if (!entry.is_regular_file()) continue;
    if (!FileUtils::is_image_file(entry.path().extension().string())) continue;
    std::string stem = entry.path().stem().string();
    left_map[stem] = fs::absolute(entry.path()).string();
  }

  // match against right images
  for (const auto &entry : fs::directory_iterator(right_dir)) {
    if (!entry.is_regular_file()) continue;
    if (!FileUtils::is_image_file(entry.path().extension().string())) continue;
    std::string stem = entry.path().stem().string();
    auto it = left_map.find(stem);
    if (it != left_map.end()) {
      pairs.emplace_back(it->second, fs::absolute(entry.path()).string());
    }
  }

  std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) {
    return fs::path(a.first).filename().string() < fs::path(b.first).filename().string();
  });

  return pairs;
}

bool processOneSceneDir(const std::string &scene_dir, const std::string &root_dir, const std::string &result_root,
                        std::shared_ptr<stereonet::StereonetProcess> stereonet_process, int model_input_w,
                        int model_input_h, float uncertainty_th) {
  std::vector<std::pair<std::string, std::string>> img_pairs = FileUtils::find_pairs(scene_dir);
  std::vector<std::string> single_img_paths;
  bool use_vert_split = false;
  bool use_left_right_dirs = false;
  if (img_pairs.empty()) {
    // try left/right subdirectory mode
    if (hasLeftRightDirs(scene_dir)) {
      img_pairs = findLeftRightPairs(scene_dir);
      if (img_pairs.empty()) {
        LOG_WARN(nullptr, "=> no matching image pairs found in left/right subdirs: " << scene_dir);
        return false;
      }
      use_left_right_dirs = true;
      LOG_INFO(nullptr, "=> found " << img_pairs.size() << " image pairs via left/right subdirs");
    } else {
      // no left/right named image pairs, fall back to vertically stacked images
      // (top half is the left image, bottom half is the right image)
      single_img_paths = FileUtils::find_images(scene_dir);
      if (single_img_paths.empty()) {
        LOG_WARN(nullptr, "=> no image pairs or images found in " << scene_dir);
        return false;
      }
      use_vert_split = true;
    }
  }

  fs::path root_path = fs::weakly_canonical(root_dir);
  fs::path scene_path = fs::weakly_canonical(scene_dir);
  std::string root_name = root_path.filename().string();
  fs::path rel_path;
  try {
    rel_path = fs::relative(scene_path, root_path);
  } catch (...) {
    rel_path = scene_path.filename();
  }
  std::string result_dir;
  if (rel_path.empty() || rel_path == ".") {
    result_dir = (fs::path(result_root) / root_name).string();
  } else {
    result_dir = (fs::path(result_root) / root_name / rel_path).string();
  }
  fs::create_directories(result_dir);

  LOG_INFO(nullptr, "=> ==============================================");
  LOG_INFO(nullptr, "=> processing folder: " << scene_dir);
  LOG_INFO(nullptr, "=> result dir: " << result_dir);

  // read camera intrinsic
  stereonet::CameraIntrinsic camera_intrinsic;
  std::string intrinsic_file;
  std::string file1 = (fs::path(scene_dir) / "camera_intrinsic.txt").string();
  std::string file2 = (fs::path(scene_dir) / "K.txt").string();

  if (fs::exists(file1)) {
    intrinsic_file = file1;
    if (readCameraIntrinsicFromFile(intrinsic_file, camera_intrinsic)) {
      LOG_INFO(nullptr, "=> cam intrinsic [fx,fy,cx,cy,baseline]: "
                            << camera_intrinsic.fx << ", " << camera_intrinsic.fy << ", " << camera_intrinsic.cx << ", "
                            << camera_intrinsic.cy << ", " << camera_intrinsic.baseline);
      saveCameraIntrinsic(result_dir, camera_intrinsic);
    }
  } else if (fs::exists(file2)) {
    intrinsic_file = file2;
    if (readCameraIntrinsicFromFile(intrinsic_file, camera_intrinsic)) {
      LOG_INFO(nullptr, "=> cam intrinsic [fx,fy,cx,cy,baseline]: "
                            << camera_intrinsic.fx << ", " << camera_intrinsic.fy << ", " << camera_intrinsic.cx << ", "
                            << camera_intrinsic.cy << ", " << camera_intrinsic.baseline);
      saveCameraIntrinsic(result_dir, camera_intrinsic);
    }
  } else {
    LOG_WARN(nullptr, "=> no intrinsic file found in " << scene_dir);
  }

  // build processing list: original left/right pairs first, then vertically stacked images
  struct WorkItem {
    std::string left_path;   // left image path, empty when vert_split
    std::string right_path;  // right image path, empty when vert_split
    std::string stacked_path;  // stacked image path, used when vert_split
    std::string prefix;      // output name prefix
    bool vert_split = false; // true: stacked image, top half is left, bottom half is right
    bool left_right_dirs = false; // true: from left/right subdirectories
  };
  std::vector<WorkItem> work_items;
  for (auto &img_pair : img_pairs) {
    WorkItem item;
    item.left_path = img_pair.first;
    item.right_path = img_pair.second;
    item.prefix = fs::path(img_pair.first).stem().string();
    item.left_right_dirs = use_left_right_dirs;
    work_items.push_back(item);
  }
  if (use_vert_split) {
    for (const auto &img_path : single_img_paths) {
      WorkItem item;
      item.stacked_path = img_path;
      item.prefix = fs::path(img_path).stem().string();
      item.vert_split = true;
      work_items.push_back(item);
    }
  }

  bool update_cam_intr = false;
  for (auto &item : work_items) {
    cv::Mat left_img, right_img;
    std::string left_img_name, right_img_name;
    if (!item.vert_split) {
      LOG_INFO(nullptr, "=> processing image pair: " << item.left_path << " " << item.right_path);
      left_img = cv::imread(item.left_path);
      right_img = cv::imread(item.right_path);
      if (left_img.empty() || right_img.empty()) {
        LOG_ERROR(nullptr, "=> image read failed");
        continue;
      }
      if (item.left_right_dirs) {
        left_img_name = "left_" + fs::path(item.left_path).filename().string();
        right_img_name = "right_" + fs::path(item.right_path).filename().string();
      } else {
        left_img_name = fs::path(item.left_path).filename().string();
        right_img_name = fs::path(item.right_path).filename().string();
      }
    } else {
      LOG_INFO(nullptr, "=> processing vertically stacked image: " << item.stacked_path
                              << " (top half is left, bottom half is right)");
      cv::Mat stacked_img = cv::imread(item.stacked_path);
      if (stacked_img.empty()) {
        LOG_ERROR(nullptr, "=> image read failed");
        continue;
      }
      if (stacked_img.rows % 2 != 0) {
        LOG_ERROR(nullptr, "=> stacked image height is odd, cannot split into two halves: " << item.stacked_path);
        continue;
      }
      int half_h = stacked_img.rows / 2;
      left_img = stacked_img(cv::Rect(0, 0, stacked_img.cols, half_h)).clone();
      right_img = stacked_img(cv::Rect(0, half_h, stacked_img.cols, half_h)).clone();
      std::string stacked_name = fs::path(item.stacked_path).filename().string();
      left_img_name = "left_" + stacked_name;
      right_img_name = "right_" + stacked_name;
    }

    // resize
    cv::Mat left_img_resize, right_img_resize;
    if (left_img.cols != model_input_w || left_img.rows != model_input_h) {
      LOG_INFO(nullptr, "=> left img size [" << left_img.cols << ", " << left_img.rows << "], right img size ["
                                             << right_img.cols << ", " << right_img.rows << "], need resize to ["
                                             << model_input_w << ", " << model_input_h << "]");
      cv::resize(left_img, left_img_resize, cv::Size(model_input_w, model_input_h));
      cv::resize(right_img, right_img_resize, cv::Size(model_input_w, model_input_h));
      if (!update_cam_intr && camera_intrinsic.is_valid()) {
        camera_intrinsic.fx = camera_intrinsic.fx * model_input_w / left_img.cols;
        camera_intrinsic.fy = camera_intrinsic.fy * model_input_h / left_img.rows;
        camera_intrinsic.cx = camera_intrinsic.cx * model_input_w / left_img.cols;
        camera_intrinsic.cy = camera_intrinsic.cy * model_input_h / left_img.rows;
        LOG_INFO(nullptr, "=> after resize, cam intrinsic [fx, fy, cx, cy, baseline]: ["
                              << camera_intrinsic.fx << ", " << camera_intrinsic.fy << ", " << camera_intrinsic.cx
                              << ", " << camera_intrinsic.cy << ", " << camera_intrinsic.baseline << "]");
        saveCameraIntrinsic(result_dir, camera_intrinsic);
        update_cam_intr = true;
      }
    } else {
      left_img_resize = left_img;
      right_img_resize = right_img;
    }

    // convert to nv12
    size_t model_input_nv12_size = static_cast<size_t>(model_input_w) * static_cast<size_t>(model_input_h) * 3 / 2;
    std::vector<uint8_t> left_img_nv12(model_input_nv12_size);
    std::vector<uint8_t> right_img_nv12(model_input_nv12_size);
    ImgConvertUtils::bgr_mat_to_nv12(left_img_resize, left_img_nv12.data());
    ImgConvertUtils::bgr_mat_to_nv12(right_img_resize, right_img_nv12.data());

    // infer
    cv::Mat disp, uncert;
    int ret = stereonet_process->forward_sync(left_img_nv12, right_img_nv12, uncertainty_th, disp, uncert);
    if (ret != 0) {
      LOG_ERROR(nullptr, "=> forward_sync failed, ret = " << ret);
      continue;
    }
    cv::Mat depth;
    if (camera_intrinsic.is_valid()) {
      stereonet_process->perspective_disparity_to_depth(disp, depth, camera_intrinsic);
    }

    // epipolar check
    cv::Mat epipolar_visual;
    auto camera_intrinsic_ptr = std::make_shared<stereonet::CameraIntrinsic>();
    camera_intrinsic_ptr->fx = camera_intrinsic.fx;
    camera_intrinsic_ptr->fy = camera_intrinsic.fy;
    camera_intrinsic_ptr->cx = camera_intrinsic.cx;
    camera_intrinsic_ptr->cy = camera_intrinsic.cy;
    camera_intrinsic_ptr->baseline = camera_intrinsic.baseline;
    FeatureEpipolarAlign::check_epipolar_alignment(left_img_resize, right_img_resize, camera_intrinsic_ptr,
                                                   epipolar_visual);

    // save
    std::string prefix = item.prefix;

    cv::imwrite((fs::path(result_dir) / left_img_name).string(), left_img_resize);
    cv::imwrite((fs::path(result_dir) / right_img_name).string(), right_img_resize);
    cv::imwrite((fs::path(result_dir) / ("disp_" + prefix + ".pfm")).string(), disp);
    if (!uncert.empty()) {
      cv::imwrite((fs::path(result_dir) / ("uncert_" + prefix + ".pfm")).string(), uncert);
    }
    if (!epipolar_visual.empty()) {
      cv::imwrite((fs::path(result_dir) / ("epipolar_visual_" + prefix + ".png")).string(), epipolar_visual);
    }

    cv::Mat visual_img_disp = stereonet_process->render_disp_or_depth(disp);
    cv::imwrite((fs::path(result_dir) / ("visual_disp_" + prefix + ".png")).string(), visual_img_disp);
    cv::Mat visual_img_disp_sf =
        stereonet_process->render_disp_or_depth(disp, 0.0f, 192.0f, 0.0f, 10000.0f, true, 100, 2.0, 8);
    cv::imwrite((fs::path(result_dir) / ("visual_disp_sf_" + prefix + ".png")).string(), visual_img_disp_sf);

    if (camera_intrinsic.is_valid()) {
      cv::imwrite((fs::path(result_dir) / ("depth_" + prefix + ".png")).string(), depth);
      cv::Mat visual_img;
      stereonet_process->convert_visual_img(left_img_resize, disp, depth, camera_intrinsic, visual_img);
      cv::imwrite((fs::path(result_dir) / ("visual_" + prefix + ".png")).string(), visual_img);

      std::vector<stereonet::PointXYZRGB> pointcloud;
      stereonet_process->depth_to_pointcloud_rgb(depth, left_img_resize, camera_intrinsic, pointcloud);
      stereonet_process->dump_pcd_file_rgb((fs::path(result_dir) / ("pointcloud_" + prefix + ".pcd")).string(),
                                           pointcloud);
    }
  }

  return true;
}

int main(int argc, char **argv) {
  // help
  if (argc > 1) {
    std::string arg1(argv[1]);
    if (arg1 == "-h" || arg1 == "--help") {
      print_help(argv[0]);
      return 0;
    }
  }

  // parse arguments
  std::string model_path = "./model/DStereoV2.4_int16.bin";
  std::string local_img_dir = "./img";
  float uncertainty_th = -0.10f;

  if (argc > 1) model_path = argv[1];
  if (argc > 2) local_img_dir = argv[2];
  if (argc > 3) uncertainty_th = std::stof(argv[3]);

  if (!fs::exists(model_path)) {
    LOG_ERROR(nullptr, "=> model file not exist: " << model_path);
    return -1;
  }

  if (!fs::exists(local_img_dir) || !fs::is_directory(local_img_dir)) {
    LOG_ERROR(nullptr, "=> local image directory not exist or not directory: " << local_img_dir);
    return -1;
  }

  // init StereoNetProcess
  auto stereonet_process = std::make_shared<stereonet::StereonetProcess>();
  stereonet_process->init(model_path);

  int model_input_w = 0;
  int model_input_h = 0;
  stereonet_process->get_model_input_size(model_input_w, model_input_h);

  // process
  std::string result_root = "./result";
  fs::create_directories(result_root);

  int processed_dir_count = 0;

  // 1. first process the root directory itself, compatible with "directory directly put images and calibration files"
  if (hasImagePairs(local_img_dir) || hasSingleImages(local_img_dir) || hasLeftRightDirs(local_img_dir)) {
    if (processOneSceneDir(local_img_dir, local_img_dir, result_root, stereonet_process, model_input_w, model_input_h,
                           uncertainty_th)) {
      ++processed_dir_count;
    }
  }

  // 2. recursively process all subdirectories, compatible with "subdirectories have subdirectories"
  for (const auto &entry : fs::recursive_directory_iterator(local_img_dir)) {
    if (!entry.is_directory()) continue;

    std::string sub_dir = entry.path().string();
    // skip left/right subdirectories that belong to a left/right-dirs parent,
    // otherwise they would be picked up as vert-split fallback
    if (hasLeftRightDirs(fs::path(sub_dir).parent_path().string())) {
      std::string dirname = fs::path(sub_dir).filename().string();
      if (dirname == "left" || dirname == "right") continue;
    }
    if (!hasImagePairs(sub_dir) && !hasSingleImages(sub_dir) && !hasLeftRightDirs(sub_dir)) continue;

    if (processOneSceneDir(sub_dir, local_img_dir, result_root, stereonet_process, model_input_w, model_input_h,
                           uncertainty_th)) {
      ++processed_dir_count;
    }
  }

  if (processed_dir_count == 0) {
    LOG_WARN(nullptr, "=> no valid scene directory found under: " << local_img_dir);
  }

  LOG_INFO(nullptr, "=> ==============================================");
  LOG_INFO(nullptr, "=> done, processed dir count: " << processed_dir_count);

  stereonet_process.reset();
  return 0;
}