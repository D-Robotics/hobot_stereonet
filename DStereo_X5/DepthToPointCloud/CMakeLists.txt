cmake_minimum_required(VERSION 3.10)
project(DepthToPointCloud)

set(CMAKE_CXX_STANDARD 17)

# Release
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Debug
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# =================================== set platform ====================================
add_definitions(-DPLATFORM_X5)

# =================================== find packages ===================================
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lib_opencv4.5.4/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

# =================================== set include & link dirs =========================
include_directories(
    ../../include
)

# =================================== build executable ================================
add_executable(depth_to_pointcloud main.cpp)
target_link_libraries(depth_to_pointcloud ${OpenCV_LIBS})
