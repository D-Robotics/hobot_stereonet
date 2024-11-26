cmake_minimum_required(VERSION 3.5)

#SET(CMAKE_C_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
#SET(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)

project(stereonet_matcher) # Project name
find_package(OpenCV)
find_package(rclcpp REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

include_directories(
        ${PROJECT_SOURCE_DIR}/../include/
)

add_executable(
        stereonet_matcher
        ${PROJECT_SOURCE_DIR}/stereonet_matcher.cpp
)
ament_target_dependencies(stereonet_matcher
        rclcpp
        cv_bridge
        sensor_msgs)

target_link_libraries(stereonet_matcher
        ${rclcpp_LIBRARIES}
        ${OpenCV_LIBRARIES}
        pthread)

install(
        TARGETS stereonet_matcher
        RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()        
