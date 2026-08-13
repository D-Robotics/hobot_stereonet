cmake_minimum_required(VERSION 3.5)
project(hobot_stereonet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")
set(CMAKE_BUILD_TYPE "Release")

# =================================== find packages ===================================
find_package(OpenCV)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenMP REQUIRED)

# =================================== FetchContent ====================================
# include(FetchContent)
# set(FETCHCONTENT_QUIET OFF)
# FetchContent_Declare(
#   concurrentqueue
#   GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
#   GIT_TAG master
# )
# FetchContent_Declare(
#   magic_enum
#   GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
#   GIT_TAG v0.9.7
# )
# FetchContent_Declare(
#   bs_thread_pool
#   GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
#   GIT_TAG master
# )
# FetchContent_MakeAvailable(concurrentqueue magic_enum bs_thread_pool)

set(concurrentqueue_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/concurrentqueue)
set(magic_enum_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/magic_enum/include)
set(bs_thread_pool_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/thread-pool/include)

# =================================== set include & link dirs =========================
set(SYS_ROOT ${CMAKE_SYSROOT})
message("SYS_ROOT is " ${SYS_ROOT})

if(DEFINED ENV{ROS_DISTRO})
  set(PREFERRED_ROS_DISTRO $ENV{ROS_DISTRO})
else()
  if(EXISTS "/opt/ros/jazzy")
    set(PREFERRED_ROS_DISTRO jazzy)
  elseif(EXISTS "/opt/ros/humble")
    set(PREFERRED_ROS_DISTRO humble)
  else()
    set(PREFERRED_ROS_DISTRO humble)
  endif()
endif()

if(DEFINED PREFERRED_ROS_DISTRO)
  message(STATUS "Preferred ROS distribution: ${PREFERRED_ROS_DISTRO}")
endif()


if("${PREFERRED_ROS_DISTRO}" STREQUAL "jazzy")
  add_definitions(-DCV_BRIDGE_CPP)
endif()

set(PCL_INCLUDE_DIRS
  ${SYS_ROOT}/usr/include/pcl-1.12/
  ${SYS_ROOT}/usr/include/pcl-1.14/
  ${SYS_ROOT}/usr/include/eigen3/
  /opt/ros/${PREFERRED_ROS_DISTRO}/include/
  /opt/ros/${PREFERRED_ROS_DISTRO}/include/pcl_conversions/
  /opt/ros/${PREFERRED_ROS_DISTRO}/include/message_filters/
  /opt/ros/${PREFERRED_ROS_DISTRO}/include/pcl_msgs/
)

set(PCL_LIBRARIES
  pcl_common
  pcl_filters
  pcl_io
)

set(DNN_LIBS "")
if(PLATFORM_S100)
  message("hobot_stereonet build platform S100")
  add_definitions(-DPLATFORM_S100)
  set(LAUNCH_DIR "s100")
  list(APPEND DNN_LIBS dnn hbucp)
elseif(PLATFORM_S600)
  message("hobot_stereonet build platform S600")
  add_definitions(-DPLATFORM_S600)
  set(LAUNCH_DIR "s600")
  list(APPEND DNN_LIBS dnn hbucp)
else()
  message("hobot_stereonet build platform X5")
  add_definitions(-DPLATFORM_X5)
  set(LAUNCH_DIR "x5")
  list(APPEND DNN_LIBS dnn cnn_intf hbrt_bayes_aarch64)
endif()

include_directories(
  ${PCL_INCLUDE_DIRS}
  ${concurrentqueue_DIR}
  ${magic_enum_DIR}
  ${bs_thread_pool_DIR}
)

link_directories(
  ${SYS_ROOT}/usr/lib/hbbpu/
  ${SYS_ROOT}/usr/lib/
  ${SYS_ROOT}/usr/hobot/lib/
  ${SYS_ROOT}/usr/lib/aarch64-linux-gnu/
)

# =================================== build stereonet model component =================
add_library(stereonet_model_component SHARED
  src/stereonet_component.cpp
  src/stereonet_process.cpp
  src/timer_utils.cpp
  src/img_convert_utils.cpp
  src/file_utils.cpp
  src/speckle_filter.cpp
  src/stereo_rectify.cpp
  src/pcl_filter.cpp
  src/epipolar_align.cpp
  src/feature_epipolar_align.cpp
)
target_compile_features(stereonet_model_component PUBLIC c_std_99 cxx_std_17) # Require C99 and C++17
target_include_directories(stereonet_model_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(stereonet_model_component
  rclcpp
  rclcpp_components
  cv_bridge
  std_msgs
  sensor_msgs
  message_filters
  tf2_ros
  geometry_msgs
)
target_link_libraries(stereonet_model_component
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${OpenCV_LIBRARIES}
  alog
  gflags
  ${DNN_LIBS}
  pthread
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)

# compile executable file
rclcpp_components_register_node(stereonet_model_component
  PLUGIN "stereonet::StereoNetNode"
  EXECUTABLE stereonet_model_node
)

# =================================== install files ===================================
# install(TARGETS stereonet_model_component RUNTIME DESTINATION lib/${PROJECT_NAME})
install(
  TARGETS stereonet_model_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS stereonet_model_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
file(GLOB LAUNCH_FILES
  launch/*launch.py
  launch/${LAUNCH_DIR}/*launch.py
)
install(FILES ${LAUNCH_FILES} DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY data DESTINATION share/${PROJECT_NAME})
install(DIRECTORY script DESTINATION share/${PROJECT_NAME})

ament_package()
