//
// Created by zhy on 7/25/24.
//

#ifndef STEREONET_MODEL_INCLUDE_PCL_FILTER_H_
#define STEREONET_MODEL_INCLUDE_PCL_FILTER_H_
#include <rclcpp/rclcpp.hpp>

#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <sensor_msgs/msg/point_cloud2.hpp>

struct pcl_filter {
  void static applyfilter(sensor_msgs::msg::PointCloud2 &pcd_msg,
      float leaf_size = 0.03, int KMean = 5, float stdv = 0.01) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    auto start = std::chrono::high_resolution_clock::now();
    pcl::fromROSMsg(pcd_msg, *cloud);
    auto start2 = std::chrono::high_resolution_clock::now();

    RCLCPP_DEBUG(rclcpp::get_logger(""), "origin pcd size: %d", cloud->size());

    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size); // 设置体素大小
    voxel_grid.filter(*cloud_filtered);
    RCLCPP_DEBUG(rclcpp::get_logger(""), "after voxel, pcd size: %d", cloud_filtered->size());

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setMeanK(KMean);
    sor.setStddevMulThresh(stdv);
    sor.filter(*cloud_filtered);
    RCLCPP_DEBUG(rclcpp::get_logger(""), "after StatisticalOutlierRemoval, pcd size: %d", cloud_filtered->size());

//  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
//  pcl::SACSegmentation<pcl::PointXYZ> segmentation;
//  segmentation.setInputCloud(cloud_filtered);
//  segmentation.setModelType(pcl::SACMODEL_PLANE);
//  segmentation.setMethodType(pcl::SAC_RANSAC);
//  segmentation.setDistanceThreshold(0.15); // 设置距离阈值，点到平面的距离小于该阈值的点将被认为是地面点
//  segmentation.segment(*inliers, *coefficients);
//
//  // 创建一个提取对象，用于提取地面点
//  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
//  pcl::ExtractIndices<pcl::PointXYZ> extract;
//  extract.setInputCloud(cloud_filtered);
//  extract.setIndices(inliers);
//  extract.setNegative(false); // 提取地面点，即保留inliers对应的点
//  extract.filter(*cloud_ground);
//
//  // 创建一个提取对象，用于提取非地面点
//  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_non_ground(new pcl::PointCloud<pcl::PointXYZ>);
//  extract.setNegative(true); // 提取非地面点，即去除inliers对应的点
//  extract.filter(*cloud_non_ground);

    auto start3 = std::chrono::high_resolution_clock::now();
    pcl::toROSMsg(*cloud_filtered, pcd_msg);
    auto start4 = std::chrono::high_resolution_clock::now();

    auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start).count();
    auto d2 = std::chrono::duration_cast<std::chrono::milliseconds>(start3 - start2).count();
    auto d3 = std::chrono::duration_cast<std::chrono::milliseconds>(start4 - start3).count();
    RCLCPP_DEBUG(rclcpp::get_logger(""),
        "ROS to PCL: %dms, "
        "PCL filter: %dms,, "
        "PCL to ROS: %dms", d1, d2, d3);
  }
};



#endif //STEREONET_MODEL_INCLUDE_PCL_FILTER_H_
