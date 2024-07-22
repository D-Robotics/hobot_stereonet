cmake_minimum_required(VERSION 3.5)

#SET(CMAKE_C_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
#SET(CMAKE_CXX_COMPILER /opt/gcc-linaro-6.5.0-2018.12-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)

project(stereonet_model) # Project name

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

set(SYS_ROOT ${CMAKE_SYSROOT})
message("PREFIX_PATH is " ${PREFIX_PATH})
message("SYS_ROOT is " ${SYS_ROOT})

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${rclcpp_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}/cv_bridge/
)

include_directories(${SYS_ROOT}/usr/include)

link_directories(
  ${SYS_ROOT}/usr/lib/hbbpu/
  ${SYS_ROOT}/usr/lib/
  ${SYS_ROOT}/usr/hobot/lib/
)

add_executable(
        stereonet_model_node
        ${PROJECT_SOURCE_DIR}/src/stereonet_component.cpp
        ${PROJECT_SOURCE_DIR}/src/stereonet_process.cpp
        ${PROJECT_SOURCE_DIR}/src/image_conversion.cpp
        ${PROJECT_SOURCE_DIR}/src/node.cpp
)

target_link_libraries(stereonet_model_node
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${OpenCV_LIBRARIES}
        #opencv_world
        alog
        gflags
        dnn
        cnn_intf
        hbrt_bayes_aarch64
        pthread)

ament_target_dependencies(stereonet_model_node
        rclcpp
        rclcpp_components
        cv_bridge
        std_msgs
        sensor_msgs)

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
        sensor_msgs)

target_link_libraries(stereonet_model
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${OpenCV_LIBRARIES}
        #opencv_world
        alog
        gflags
        dnn
        cnn_intf
        hbrt_bayes_aarch64
        pthread)

rclcpp_components_register_node(stereonet_model
        PLUGIN "stereonet::StereoNetNode"
        EXECUTABLE stereonet_model_component_node)


add_executable(
        stereonet_model_sub_node
        ${PROJECT_SOURCE_DIR}/src/stereonet_intra_sub.cpp
        ${PROJECT_SOURCE_DIR}/src/stereonet_sub_node.cpp
)
ament_target_dependencies(stereonet_model_sub_node
        rclcpp
        rclcpp_components
        std_msgs
        sensor_msgs
        cv_bridge)

target_link_libraries(stereonet_model_sub_node
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${OpenCV_LIBRARIES}
        pthread)

add_library(
        stereonet_model_intra_sub SHARED
        ${PROJECT_SOURCE_DIR}/src/stereonet_intra_sub.cpp
)
ament_target_dependencies(stereonet_model_intra_sub
        rclcpp
        rclcpp_components
        std_msgs
        sensor_msgs
        cv_bridge)

target_link_libraries(stereonet_model_intra_sub
        ${rclcpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${OpenCV_LIBRARIES}
        pthread)


rclcpp_components_register_node(stereonet_model_intra_sub
        PLUGIN "stereonet::StereoNetSubNode"
        EXECUTABLE stereonet_model_intra_sub_node)

install(
        TARGETS stereonet_model_node stereonet_model stereonet_model_intra_sub_node stereonet_model_intra_sub
        RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
        ${PROJECT_SOURCE_DIR}/config/
        DESTINATION share/${PROJECT_NAME}/config/
        )

install(DIRECTORY
        launch
        DESTINATION share/${PROJECT_NAME}/
        )

ament_package()        
