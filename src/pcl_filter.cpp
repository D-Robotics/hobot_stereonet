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

#include "pcl_filter.h"

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
PCLFilterUtils::statisticalOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud,
                                          float voxel_leaf_size, int mean_k, double std_dev_mul_thresh) {
  if (!input_cloud || input_cloud->empty()) {
    return nullptr;
  }

  // 1. VoxelGrid downsampling
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
  voxel_filter.setInputCloud(input_cloud);
  voxel_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
  voxel_filter.filter(*cloud_downsampled);

  if (cloud_downsampled->empty()) {
    return nullptr;
  }

  // 2. StatisticalOutlierRemoval
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
  sor.setInputCloud(cloud_downsampled);
  sor.setMeanK(mean_k);
  sor.setStddevMulThresh(std_dev_mul_thresh);
  sor.filter(*filtered_cloud);

  return filtered_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
PCLFilterUtils::radiusOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, float voxel_leaf_size,
                                     double radius_search, int min_neighbors) {
  if (!input_cloud || input_cloud->empty()) {
    return nullptr;
  }

  // 1. VoxelGrid downsampling
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
  voxel_filter.setInputCloud(input_cloud);
  voxel_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
  voxel_filter.filter(*cloud_downsampled);

  if (cloud_downsampled->empty()) {
    return nullptr;
  }

  // 2. RadiusOutlierRemoval
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
  ror.setInputCloud(cloud_downsampled);
  ror.setRadiusSearch(radius_search);
  ror.setMinNeighborsInRadius(min_neighbors);
  ror.filter(*filtered_cloud);

  return filtered_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr
PCLFilterUtils::gridBasedOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, float grid_size,
                                        int grid_min_point_count) {
  if (!input_cloud || input_cloud->empty()) {
    return nullptr;
  }

  // 1. make grid
  const float invGridSize = 1.0f / grid_size;

  std::unordered_map<GridIndex, int, GridIndexHash> grid_point_count;
  grid_point_count.reserve(input_cloud->size());
  std::vector<GridIndex> point_grid_indices;
  point_grid_indices.reserve(input_cloud->size());

  for (const auto &point : *input_cloud) {
    GridIndex idx(static_cast<int>(std::floor((point.x) * invGridSize)),
                  static_cast<int>(std::floor((point.y) * invGridSize)),
                  static_cast<int>(std::floor((point.z) * invGridSize)));

    point_grid_indices.push_back(idx);
    grid_point_count[idx]++;
  }

  // 2. filter
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  size_t valid_count = 0;
  for (const auto &[idx, count] : grid_point_count) {
    if (count >= grid_min_point_count) {
      valid_count += count;
    }
  }
  output_cloud->reserve(valid_count);

  size_t point_index = 0;
  for (const auto &point : *input_cloud) {
    const GridIndex &grid_idx = point_grid_indices[point_index++];
    if (grid_point_count[grid_idx] >= grid_min_point_count) {
      output_cloud->push_back(point);
    }
  }

  output_cloud->width = output_cloud->size();
  output_cloud->height = 1;
  output_cloud->is_dense = false;

  return output_cloud;
}