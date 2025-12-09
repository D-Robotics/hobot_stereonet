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

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
// =============== stereonet ===============
#include "log_macros.h"
#include "camera_intrinsic.h"
// =============== stereonet ===============

bool readCameraIntrinsicFromFile(const std::string &file_path, stereonet::CameraIntrinsic &intrinsic) {
  std::ifstream infile(file_path);
  if (!infile.is_open()) {
    std::cerr << "Failed to open file: " << file_path << std::endl;
    return false;
  }

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream ss(line);
    double fx, fy, cx, cy, baseline;
    if (ss >> fx >> fy >> cx >> cy >> baseline) {
      intrinsic.fx = fx;
      intrinsic.fy = fy;
      intrinsic.cx = cx;
      intrinsic.cy = cy;
      intrinsic.baseline = baseline;
      return intrinsic.is_valid();
    } else {
      std::cerr << "Failed to parse line: " << line << std::endl;
      return false;
    }
  }

  std::cerr << "No valid data found in file: " << file_path << std::endl;
  return false;
}

void depthToPointCloud(const cv::Mat &depth, const stereonet::CameraIntrinsic &intr,
                       std::vector<cv::Point3f> &pointCloud) {
  pointCloud.clear();
  for (int v = 0; v < depth.rows; ++v) {
    for (int u = 0; u < depth.cols; ++u) {
      float d = depth.at<uint16_t>(v, u) / 1000.0f; // mm to m
      if (d > 0) {                                  // skip invalid depth values
        float x = (u - intr.cx) * d / intr.fx;
        float y = (v - intr.cy) * d / intr.fy;
        float z = d;
        pointCloud.emplace_back(x, y, z);
      }
    }
  }
}

void savePointCloudToTxt(const std::string &filename, const std::vector<cv::Point3f> &pointCloud) {
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    LOG_ERROR(nullptr, "=> failed to open file: " << filename);
    return;
  }
  for (const auto &point : pointCloud) {
    outFile << point.x << " " << point.y << " " << point.z << "\n";
  }
  outFile.close();
  LOG_INFO(nullptr, "=> save point cloud to: " << filename);
}

int main() {
  stereonet::CameraIntrinsic camera_intrinsic;
  if (readCameraIntrinsicFromFile("./img/camera_intrinsic.txt", camera_intrinsic)) {
    LOG_INFO(nullptr, "=> cam intrinsic [fx, fy, cx, cy, baseline]: ["
                          << camera_intrinsic.fx << ", " << camera_intrinsic.fy << ", " << camera_intrinsic.cx << ", "
                          << camera_intrinsic.cy << ", " << camera_intrinsic.baseline << "]");
  }

  std::string depthFile = "./img/depth.png";
  cv::Mat depth = cv::imread(depthFile, cv::IMREAD_UNCHANGED);
  if (depth.empty()) {
    LOG_ERROR(nullptr, "=> failed to read depth file: " << depthFile);
    return -1;
  }

  std::vector<cv::Point3f> pointCloud;
  depthToPointCloud(depth, camera_intrinsic, pointCloud);
  savePointCloudToTxt("pointcloud.txt", pointCloud);

  return 0;
}