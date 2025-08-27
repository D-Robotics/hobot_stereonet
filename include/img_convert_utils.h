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

#ifndef HOBOT_STEREONET_INCLUDE_IMAGE_CONVERSION_H_
#define HOBOT_STEREONET_INCLUDE_IMAGE_CONVERSION_H_

#include <opencv2/opencv.hpp>
#include <arm_neon.h>

class ImgConvertUtils {
public:
  // delete the default constructor
  ImgConvertUtils() = delete;

  // utility functions
  static void nv12_to_bgr24_neon(uint8_t *nv12, uint8_t *bgr24, int width, int height);
  static void bgr24_to_nv12_neon(uint8_t *bgr24, uint8_t *nv12, int width, int height);
  static void bgr_mat_to_nv12_mat(const cv::Mat &bgr, cv::Mat &nv12);
  static void nv12_mat_to_bgr_mat(const uint8_t *nv12, cv::Mat &bgr24, int width, int height);
  static void bgr_mat_to_nv12(const cv::Mat &bgr, uint8_t *nv12);
};

#endif // HOBOT_STEREONET_INCLUDE_IMAGE_CONVERSION_H_
