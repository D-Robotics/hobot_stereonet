cmake_minimum_required(VERSION 3.5)

#SET(CMAKE_C_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
#SET(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)

project(hobot_stereonet) # Project name

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")
set(CMAKE_BUILD_TYPE "Release")

include_directories(
  ${PROJECT_SOURCE_DIR}/include/
)

find_package(OpenCV)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)

set(SYS_ROOT ${CMAKE_SYSROOT})
message("PREFIX_PATH is " ${PREFIX_PATH})
message("SYS_ROOT is " ${SYS_ROOT})

set(PCL_INCLUDE_DIRS
  ${SYS_ROOT}/usr/include/pcl-1.12/
  ${SYS_ROOT}/usr/include/eigen3/
  /opt/ros/humble/include/
  /opt/ros/humble/include/pcl_conversions/
  /opt/ros/humble/include/message_filters/
  /opt/ros/humble/include/pcl_msgs/
)

set(PCL_LIBRARIES
  pcl_common
  pcl_kdtree
  pcl_octree
  pcl_search
  pcl_sample_consensus
  pcl_filters
  pcl_io
  pcl_features
  pcl_ml
  pcl_segmentation
  pcl_visualization
  pcl_surface
  pcl_registration
  pcl_keypoints
  pcl_tracking
  pcl_recognition
  pcl_stereo
  pcl_apps
  pcl_outofcore
  pcl_people
)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}/cv_bridge/
  ${PCL_INCLUDE_DIRS}
  ${SYS_ROOT}/usr/include
)

link_directories(
  ${SYS_ROOT}/usr/lib/hbbpu/
  ${SYS_ROOT}/usr/lib/
  ${SYS_ROOT}/usr/hobot/lib/
  ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/
)

set(DNN_LIBS "")

if(PLATFORM_S100)
  message("hobot_stereonet build platform S100")
  add_definitions(-DPLATFORM_S100)
  set(LAUNCH_DIR "s100")
  list(APPEND DNN_LIBS dnn hbucp)
else()
  message("hobot_stereonet build platform X5")
  add_definitions(-DPLATFORM_X5)
  set(LAUNCH_DIR "x5")
  list(APPEND DNN_LIBS dnn cnn_intf hbrt_bayes_aarch64)
endif()

#include_directories(
#        platform_private/x5/
#        platform_private/x5/dnn
#        platform_private/s100/
#)

# ================================ stereonet_model_node ================================
add_executable(stereonet_model_node
  ${PROJECT_SOURCE_DIR}/src/stereonet_component.cpp
  ${PROJECT_SOURCE_DIR}/src/stereonet_process.cpp
  ${PROJECT_SOURCE_DIR}/src/image_conversion.cpp
  ${PROJECT_SOURCE_DIR}/src/node.cpp
)

ament_target_dependencies(stereonet_model_node
  rclcpp
  rclcpp_components
  cv_bridge
  std_msgs
  sensor_msgs
  message_filters
  tf2_ros
  geometry_msgs
)

target_link_libraries(stereonet_model_node
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${OpenCV_LIBRARIES}
  #opencv_world
  alog
  gflags
  ${DNN_LIBS}
  pthread
  ${PCL_LIBRARIES}
)

# ================================ stereonet_model =====================================
add_library(stereonet_model SHARED
  ${PROJECT_SOURCE_DIR}/src/stereonet_component.cpp
  ${PROJECT_SOURCE_DIR}/src/stereonet_process.cpp
  ${PROJECT_SOURCE_DIR}/src/image_conversion.cpp
)

ament_target_dependencies(stereonet_model
  rclcpp
  rclcpp_components
  cv_bridge
  std_msgs
  sensor_msgs
  message_filters
  tf2_ros
  geometry_msgs
)

target_link_libraries(stereonet_model
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${OpenCV_LIBRARIES}
  #opencv_world
  alog
  gflags
  ${DNN_LIBS}
  pthread
  ${PCL_LIBRARIES}
)

rclcpp_components_register_node(stereonet_model
  PLUGIN "stereonet::StereoNetNode"
  EXECUTABLE stereonet_model_component_node
)

# ================================ stereonet_model_sub_node ============================
add_executable(stereonet_model_sub_node
  ${PROJECT_SOURCE_DIR}/src/stereonet_intra_sub.cpp
  ${PROJECT_SOURCE_DIR}/src/stereonet_sub_node.cpp
)

ament_target_dependencies(stereonet_model_sub_node
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  cv_bridge
  message_filters
)

target_link_libraries(stereonet_model_sub_node
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${OpenCV_LIBRARIES}
  pthread
)

# ================================ stereonet_model_intra_sub ===========================
add_library(stereonet_model_intra_sub SHARED
  ${PROJECT_SOURCE_DIR}/src/stereonet_intra_sub.cpp
)
ament_target_dependencies(stereonet_model_intra_sub
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  cv_bridge
  message_filters
)

target_link_libraries(stereonet_model_intra_sub
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${OpenCV_LIBRARIES}
  pthread
)

rclcpp_components_register_node(stereonet_model_intra_sub
  PLUGIN "stereonet::StereoNetSubNode"
  EXECUTABLE stereonet_model_intra_sub_node
)

# ================================ data_collection =====================================
add_executable(
  data_collection
  ${PROJECT_SOURCE_DIR}/src/data_collection.cpp
)

ament_target_dependencies(data_collection
  rclcpp
  rclcpp_components
  cv_bridge
  std_msgs
  sensor_msgs
)

target_link_libraries(data_collection
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${OpenCV_LIBRARIES}
  pthread
)

install(
  TARGETS stereonet_model_node stereonet_model_intra_sub_node data_collection
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS stereonet_model stereonet_model_intra_sub
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
file(GLOB LAUNCH_FILES
  launch/*launch.py
  launch/${LAUNCH_DIR}/*launch.py
)
install(FILES ${LAUNCH_FILES} DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY data DESTINATION share/${PROJECT_NAME})

ament_package()
