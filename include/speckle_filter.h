
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

#ifndef HOBOT_STEREONET_INCLUDE_SPECKLE_FILTER_H_
#define HOBOT_STEREONET_INCLUDE_SPECKLE_FILTER_H_

#include <vector>
#include "opencv2/opencv.hpp"

class SpeckleFilter {
public:
  // delete the default constructor
  SpeckleFilter() = delete;

  /**
   * @brief Filter out small speckles in a single-channel float image.
   *
   * This function identifies connected regions (speckles) in the input image where pixel values
   * differ by no more than `maxDiff`. If the size of a connected region is less than or equal to
   * `maxSpeckleSize`, all pixels in that region are set to `newVal`.
   *
   * @param img Input/output single-channel float image (CV_32FC1). The image is modified in place.
   * @param newVal Value to assign to pixels in small speckles.
   * @param maxSpeckleSize Maximum size of a speckle to be considered for filtering.
   * @param maxDiff Maximum allowed difference between pixel values to be considered part of the same speckle.
   */
  static void filter(cv::Mat &img, float newVal, int maxSpeckleSize, float maxDiff);
};

#endif // HOBOT_STEREONET_SRC_SPECKLE_FILTER_H_