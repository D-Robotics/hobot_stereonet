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

#include "ground_segmentation.h"

#include <algorithm>
#include <cmath>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

namespace stereonet {

namespace {
constexpr double kPi = 3.14159265358979323846;

double deg2rad(double deg) {
  return deg * kPi / 180.0;
}
} // namespace

bool segment_ground(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, const Eigen::Vector3f &up, double voxel_size,
                    int normal_k, double max_angle_deg, double max_height, double cluster_radius, int min_cluster_size,
                    std::vector<int> &ground_indices) {
  ground_indices.clear();
  if (!cloud || cloud->points.size() < 3 || normal_k <= 0 || min_cluster_size <= 0 || cluster_radius <= 0.0) {
    return false;
  }
  const double angle_max = deg2rad(std::max(0.0, max_angle_deg));
  const Eigen::Vector3f up_dir = up.normalized();
  if (up_dir.squaredNorm() < 1e-6f) {
    return false;
  }
  const bool anchor_height = max_height > 0.0;

  // 0. voxel downsample (rtabmap Grid/PreVoxelFiltering + CellSize): normals and
  //    clustering run on a coarse copy, not the dense cloud, which is what makes
  //    this fast enough for per-frame use.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr work(new pcl::PointCloud<pcl::PointXYZRGB>);
  bool downsampled = false;
  if (voxel_size > 0.0) {
    pcl::VoxelGrid<pcl::PointXYZRGB> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(static_cast<float>(voxel_size), static_cast<float>(voxel_size), static_cast<float>(voxel_size));
    vg.filter(*work);
    downsampled = true;
  } else {
    *work = *cloud;
  }
  if (work->points.size() < 3) return false;

  // 1. per-point normals, oriented toward the sensor origin so the up-angle test
  //    is consistent (ground normals point up, toward the camera)
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
  ne.setInputCloud(work);
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  ne.setSearchMethod(tree);
  ne.setKSearch(std::max(1, std::min(normal_k, static_cast<int>(work->points.size()))));
  ne.setViewPoint(0.0f, 0.0f, 0.0f);
  ne.compute(*normals);

  // 2. flat (ground-candidate) points: normal within max_angle of up.
  std::vector<int> flat;
  flat.reserve(work->points.size());
  for (size_t i = 0; i < work->points.size(); ++i) {
    const pcl::Normal &n = normals->points[i];
    if (!std::isfinite(n.normal_x) || !std::isfinite(n.normal_y) || !std::isfinite(n.normal_z)) continue;
    Eigen::Vector3f nvec(n.normal_x, n.normal_y, n.normal_z);
    // angle between the unit normal and up = acos(dot), equivalent to pcl::getAngle3D
    double cos_angle = static_cast<double>(nvec.dot(up_dir));
    if (cos_angle < -1.0) cos_angle = -1.0;
    if (cos_angle > 1.0) cos_angle = 1.0;
    if (std::acos(cos_angle) < angle_max) {
      flat.push_back(static_cast<int>(i));
    }
  }
  if (flat.size() < static_cast<size_t>(min_cluster_size)) return false;

  // 2b. height anchor (rtabmap Grid/MaxGroundHeight): the ground is the lowest flat
  //     surface, so drop flat points that rise more than max_height above that ground
  //     level. This is what keeps an elevated obstacle top (flat but above the floor)
  //     from being colored as ground. max_height <= 0 disables the anchor.
  if (anchor_height) {
    std::vector<double> up_coords;
    up_coords.reserve(flat.size());
    for (int idx : flat) up_coords.push_back(static_cast<double>(work->points[idx].getVector3fMap().dot(up_dir)));
    std::sort(up_coords.begin(), up_coords.end());
    // low percentile as the ground reference (robust to a few low outliers)
    const size_t ref_idx = up_coords.size() * 5 / 100;
    const double ground_ref = up_coords[ref_idx];
    std::vector<int> kept;
    kept.reserve(flat.size());
    for (int idx : flat) {
      if (static_cast<double>(work->points[idx].getVector3fMap().dot(up_dir)) - ground_ref <= max_height) {
        kept.push_back(idx);
      }
    }
    flat.swap(kept);
    if (flat.size() < static_cast<size_t>(min_cluster_size)) return false;
  }

  // 3. euclidean clustering on the flat points
  pcl::IndicesPtr flat_ptr(new std::vector<int>(flat));
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr cluster_tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  cluster_tree->setInputCloud(work);
  std::vector<pcl::PointIndices> clusters;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance(cluster_radius);
  ec.setMinClusterSize(min_cluster_size);
  ec.setMaxClusterSize(static_cast<unsigned int>(work->points.size()));
  ec.setSearchMethod(cluster_tree);
  ec.setIndices(flat_ptr);
  ec.setInputCloud(work);
  ec.extract(clusters);
  if (clusters.empty()) return false;

  // 4. biggest flat cluster is the ground
  size_t biggest = 0;
  for (size_t i = 1; i < clusters.size(); ++i) {
    if (clusters[i].indices.size() > clusters[biggest].indices.size()) biggest = i;
  }
  const std::vector<int> &ground_work = clusters[biggest].indices;

  if (!downsampled) {
    ground_indices.assign(ground_work.begin(), ground_work.end());
    return !ground_indices.empty();
  }

  // 5. backfill: an original point is ground if its nearest voxel-centroid is
  //    ground (within a small radius of a ground centroid).
  std::vector<bool> is_ground_work(work->points.size(), false);
  for (int idx : ground_work) is_ground_work[static_cast<size_t>(idx)] = true;

  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr back_tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  back_tree->setInputCloud(work);
  const double max_dist2 = (voxel_size * 1.5) * (voxel_size * 1.5);
  std::vector<int> nn(1);
  std::vector<float> nn_sq(1);
  ground_indices.reserve(cloud->points.size() / 2);
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    back_tree->nearestKSearch(cloud->points[i], 1, nn, nn_sq);
    if (nn_sq[0] <= max_dist2 && is_ground_work[static_cast<size_t>(nn[0])]) {
      ground_indices.push_back(static_cast<int>(i));
    }
  }

  return !ground_indices.empty();
}

} // namespace stereonet
