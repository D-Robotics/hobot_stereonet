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

#ifndef HOBOT_STEREONET_INCLUDE_EPIPOLAR_ALIGN_H_
#define HOBOT_STEREONET_INCLUDE_EPIPOLAR_ALIGN_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <numeric>
#include "camera_intrinsic.h"

class EpipolarAlign {
public:
  // delete the default constructor
  EpipolarAlign() = delete;

  // utility functions
  static void check_epipolar_alignment(const cv::Mat &left_img, const cv::Mat &right_img, const cv::Size &pattern_size,
                                       double square_size, const std::shared_ptr<stereonet::CameraIntrinsic> &cam,
                                       cv::Mat &visualize);
};

#endif // HOBOT_STEREONET_INCLUDE_EPIPOLAR_ALIGN_H_