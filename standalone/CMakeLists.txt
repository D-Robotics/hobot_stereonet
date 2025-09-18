SET(CMAKE_C_COMPILER /opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
SET(CMAKE_CXX_COMPILER /opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)

cmake_minimum_required(VERSION 3.10)

project(StereoInfer)

add_definitions(-DPLATFORM_X5)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lib_opencv4.5.4/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn
)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn/hobot/lib
)

add_executable(${PROJECT_NAME} main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/stereonet_process.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/image_conversion.cpp)

target_link_libraries(${PROJECT_NAME}
        dnn cnn_intf hbmem hbrt_bayes_aarch64 alog ${OpenCV_LIBS})
