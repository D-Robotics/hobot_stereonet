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

#ifndef HOBOT_STEREONET_INCLUDE_CAMERA_INTRINSIC_H_
#define HOBOT_STEREONET_INCLUDE_CAMERA_INTRINSIC_H_

#include <cmath>
#include <string>

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

  // Output projection of the rectified image, independent of the distortion
  // model. Supported values:
  //   "RECTIFY_PERSPECTIVE"  pinhole. Depth = Z = fx * baseline / disparity.
  //   "RECTIFY_LONGLATI"     equirectangular (spherical) unrolling.
  //   "RECTIFY_CYLINDRICAL"  cylindrical projection.
  // For the two angular models the horizontal mapping is identical
  // (u = fx * phi, with phi the azimuth), so they share the same depth formula
  //   Z = baseline / (tan(phi_l) - tan(phi_r)),  phi_r = phi_l - disparity / fx
  // where fx is the longitude pixels-per-radian (=W/HFOV_rad) of the rectified
  // angular image and Z is the forward depth along the optical axis. They
  // differ only in the vertical mapping (equirectangular: v = fy*lambda;
  // cylindrical: v = fy*tan(lambda)), which affects only the point cloud
  // back-projection.
  std::string rectify_model = "RECTIFY_PERSPECTIVE";

  // Rotation applied to the point cloud so a vertical (top/bottom) rig outputs
  // upright points, consistent with a horizontal (left/right) rig. For a
  // vertical baseline, stereoRectify rotates the rectified frame ~90 deg about
  // the optical axis, so the rectified X/Y axes are swapped relative to the
  // physical camera frame. Values:
  //   0  : no rotation (horizontal baseline, or online camera_info path)
  //   +1 : (X, Y) -> (-Y, X)
  //   -1 : (X, Y) -> ( Y, -X)
  // Only populated by the "custom" calibration path.
  int pointcloud_rotate_dir = 0;

  bool is_valid() const {
    return (fx > 0.0 && fy > 0.0 && cx >= 0.0 && cy >= 0.0 && baseline > 0.0);
  }
};

/**
 * @brief Compute the horizontal/vertical field-of-view (degrees) of a rectified
 * image of size (W, H) with focal (fx, fy), honoring the projection model:
 *   RECTIFY_PERSPECTIVE : pinhole,      HFOV = 2*atan(W/2fx), VFOV = 2*atan(H/2fy)
 *   RECTIFY_LONGLATI    : equirect,     HFOV = W/fx,          VFOV = H/fy
 *   RECTIFY_CYLINDRICAL : cylindrical,  HFOV = W/fx,          VFOV = 2*atan(H/2fy)
 * For the two angular models fx/fy are pixels-per-radian (the longitude focal),
 * so the horizontal FOV is simply W/fx rad; re-applying the pinhole formula to a
 * px/rad focal yields a meaningless number.
 */
inline void compute_fov_deg(const std::string &rectify_model, double fx, double fy, int W, int H,
                            double &hfov_deg, double &vfov_deg) {
  constexpr double kRad2Deg = 180.0 / 3.14159265358979323846;
  if (rectify_model == "RECTIFY_LONGLATI") {
    hfov_deg = (fx > 0.0) ? (W / fx) * kRad2Deg : 0.0;
    vfov_deg = (fy > 0.0) ? (H / fy) * kRad2Deg : 0.0;
  } else if (rectify_model == "RECTIFY_CYLINDRICAL") {
    hfov_deg = (fx > 0.0) ? (W / fx) * kRad2Deg : 0.0;
    vfov_deg = (fy > 0.0) ? 2.0 * std::atan(H / (2.0 * fy)) * kRad2Deg : 0.0;
  } else {
    hfov_deg = (fx > 0.0) ? 2.0 * std::atan(W / (2.0 * fx)) * kRad2Deg : 0.0;
    vfov_deg = (fy > 0.0) ? 2.0 * std::atan(H / (2.0 * fy)) * kRad2Deg : 0.0;
  }
}

} // namespace stereonet

#endif // HOBOT_STEREONET_INCLUDE_CAMERA_INTRINSIC_H_