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

#ifndef HOBOT_STEREONET_INCLUDE_GROUND_SEGMENTATION_H_
#define HOBOT_STEREONET_INCLUDE_GROUND_SEGMENTATION_H_

#include <vector>

#include <Eigen/Core>
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

namespace stereonet {

/**
 * @brief Segment the ground surface from a colored point cloud using rtabmap's
 *        normal-based flat-surface method (LocalGridMaker::segmentObstaclesFromGround):
 *          1. voxel downsample to `voxel_size` (Grid/PreVoxelFiltering + CellSize),
 *             so normal estimation runs on a coarse copy instead of the dense cloud,
 *          2. estimate per-point normals (KNN, oriented toward the sensor),
 *          3. keep flat points (normal within max_angle_deg of `up`) that are also no
 *             higher than `max_height` above the ground level (Grid/MaxGroundHeight anchor),
 *          4. euclidean-cluster those flat points and take the biggest cluster as
 *             the ground,
 *          5. map ground labels back to the original cloud (nearest voxel-centroid).
 *        Range/height limiting is expected to be done by the caller via the
 *        pointcloud_depth_max / pointcloud_height_min/max filters.
 * @param cloud Input point cloud (PointXYZRGB)
 * @param up Up vector of the cloud's coordinate frame (e.g. +Z for ROS, -Y for camera)
 * @param voxel_size Voxel leaf size (m) for the pre-segmentation downsample; <=0 = no downsample
 * @param normal_k KNN neighbors for normal estimation
 * @param max_angle_deg Max angle (deg) between a point normal and `up` to be a ground candidate
 * @param max_height Max height (m) a flat point may rise above the ground level (the
 *        lowest flat surface) and still count as ground (Grid/MaxGroundHeight anchor);
 *        <=0 = no height anchor
 * @param cluster_radius Euclidean cluster tolerance (m)
 * @param min_cluster_size Min points per cluster
 * @param[out] ground_indices Indices (into the input cloud) of points classified as ground
 * @return true if a ground surface was found
 */
bool segment_ground(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, const Eigen::Vector3f &up, double voxel_size,
                    int normal_k, double max_angle_deg, double max_height, double cluster_radius, int min_cluster_size,
                    std::vector<int> &ground_indices);

} // namespace stereonet

#endif // HOBOT_STEREONET_INCLUDE_GROUND_SEGMENTATION_H_
