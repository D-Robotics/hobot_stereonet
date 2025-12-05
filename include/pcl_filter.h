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

#ifndef HOBOT_STEREONET_INCLUDE_PCL_FILTER_H_
#define HOBOT_STEREONET_INCLUDE_PCL_FILTER_H_

#include "pcl/point_types.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/filters/radius_outlier_removal.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include <omp.h>
#include <arm_neon.h>

struct GridIndex {
  int i, j, k;

  GridIndex(int i = 0, int j = 0, int k = 0) : i(i), j(j), k(k) {
  }

  bool operator==(const GridIndex &other) const {
    return i == other.i && j == other.j && k == other.k;
  }
};

// struct GridIndexHash {
//     size_t operator()(const std::tuple<int, int, int>& grid_idx) const
//     {
//         const auto& [i, j, k] = grid_idx;
//         return std::hash<int>()(i) ^ (std::hash<int>()(j) << 16) ^ (std::hash<int>()(k) << 32);
//     }
// };

struct GridIndexHash {
  size_t operator()(const GridIndex &idx) const {
    return ((static_cast<size_t>(idx.i) * 73856093u) ^ (static_cast<size_t>(idx.j) * 19349663u) ^
            (static_cast<size_t>(idx.k) * 83492791u));
  }
};

class PCLFilterUtils {
public:
  // delete the default constructor
  PCLFilterUtils() = delete;

  /**
   * @brief Filter the point cloud using voxel grid
   * @param input_cloud The input point cloud
   * @param voxel_leaf_size The voxel leaf size
   * @param mean_k The mean k
   * @param std_dev_mul_thresh The standard deviation multiple threshold
   * @return The filtered point cloud
   */
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  statisticalOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, float voxel_leaf_size = 0.02f,
                            int mean_k = 50, double std_dev_mul_thresh = 1.0);

  /**
   * @brief Filter the point cloud using radius outlier removal
   * @param input_cloud The input point cloud
   * @param voxel_leaf_size The voxel leaf size
   * @param radius_search The radius search
   * @param min_neighbors The minimum neighbors
   * @return The filtered point cloud
   */
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  radiusOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, float voxel_leaf_size = 0.02f,
                       double radius_search = 0.05, int min_neighbors = 5);


  /**
   * @brief Filter the point cloud using grid based outlier removal
   * @param input_cloud The input point cloud
   * @param grid_size The grid size
   * @param grid_min_point_count The grid min point count threshold
   * @return The filtered point cloud
   */
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  gridBasedOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, float grid_size = 0.1f,
                          int grid_min_point_count = 5);
};

#endif // HOBOT_STEREONET_INCLUDE_PCL_FILTER_H_
