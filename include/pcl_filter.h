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

class PCLFilterUtils {
public:
  /**
   * @brief 对输入点云进行VoxelGrid下采样 + 统计滤波，去除离群点
   * @param input_cloud 输入点云
   * @param voxel_leaf_size 体素下采样大小（单位与点云坐标一致）
   * @param mean_k 邻域点数，默认50
   * @param std_mul 标准差倍数阈值，默认1.0
   * @return 滤波后的点云
   */
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  statisticalOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, float voxel_leaf_size = 0.02f,
                            int mean_k = 50, double std_dev_mul_thresh = 1.0) {
    if (!input_cloud || input_cloud->empty()) {
      return nullptr;
    }

    // 1. VoxelGrid 下采样
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_filter.filter(*cloud_downsampled);

    if (cloud_downsampled->empty()) {
      return nullptr;
    }

    // 2. StatisticalOutlierRemoval 去离群点
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud_downsampled);
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(std_dev_mul_thresh);
    sor.filter(*filtered_cloud);

    return filtered_cloud;
  }

  /**
   * @brief 对输入点云进行VoxelGrid下采样 + 半径滤波，去除孤立点
   * @param input_cloud 输入点云
   * @param voxel_leaf_size 体素下采样大小（单位与点云一致）
   * @param radius_search 邻域搜索半径
   * @param min_neighbors 邻域内最少点数，小于该值会被去掉
   * @return 滤波后的点云
   */
  static pcl::PointCloud<pcl::PointXYZRGB>::Ptr
  radiusOutlierRemoval(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &input_cloud, float voxel_leaf_size = 0.02f,
                       double radius_search = 0.05, int min_neighbors = 5) {
    if (!input_cloud || input_cloud->empty()) {
      return nullptr;
    }

    // 1. VoxelGrid 下采样
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
    voxel_filter.filter(*cloud_downsampled);

    if (cloud_downsampled->empty()) {
      return nullptr;
    }

    // 2. RadiusOutlierRemoval 去孤立点
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
    ror.setInputCloud(cloud_downsampled);
    ror.setRadiusSearch(radius_search);
    ror.setMinNeighborsInRadius(min_neighbors);
    ror.filter(*filtered_cloud);

    return filtered_cloud;
  }
};

#endif // HOBOT_STEREONET_INCLUDE_PCL_FILTER_H_
