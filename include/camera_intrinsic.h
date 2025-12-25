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

#ifndef HOBOT_STEREONET_INCLUDE_CAMERA_INTRINSIC_H_
#define HOBOT_STEREONET_INCLUDE_CAMERA_INTRINSIC_H_

namespace stereonet {

/**
 * @struct CameraIntrinsic
 * @brief Structure to hold camera intrinsic parameters.
 */
struct CameraIntrinsic {
  double cx = 0.0;
  double cy = 0.0;
  double fx = 0.0;
  double fy = 0.0;
  double baseline = 0.0; // in meters
  double doffs = 0.0;

  bool is_valid() const {
    return (fx > 0.0 && fy > 0.0 && cx >= 0.0 && cy >= 0.0 && baseline > 0.0);
  }
};

} // namespace stereonet

#endif // HOBOT_STEREONET_INCLUDE_CAMERA_INTRINSIC_H_