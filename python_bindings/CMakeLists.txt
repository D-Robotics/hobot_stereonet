cmake_minimum_required(VERSION 3.5)
project(stereonet_pybind)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =================================== set platform ====================================
if(PLATFORM_X5)
    add_definitions(-DPLATFORM_X5)
elseif(PLATFORM_S100)
    add_definitions(-DPLATFORM_S100)
endif()

# =================================== find packages ===================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../standalone/3rdparty/lib_opencv4.5.4/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

set(concurrentqueue_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/concurrentqueue)
set(magic_enum_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/magic_enum/include)
set(bs_thread_pool_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/thread-pool/include)

# =================================== set include & link dirs =========================
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../standalone/3rdparty/eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}/../standalone/3rdparty/libdnn
    ${concurrentqueue_DIR}
    ${magic_enum_DIR}
    ${bs_thread_pool_DIR}
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../standalone/3rdparty/libdnn
    ${CMAKE_CURRENT_SOURCE_DIR}/../standalone/3rdparty/libdnn/hobot/lib
)

# =================================== build library ===================================
pybind11_add_module(dstereonet
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stereonet_process.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/timer_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/img_convert_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stereonet_pybind.cpp
)

target_link_libraries(dstereonet PRIVATE
    dnn cnn_intf hbmem hbrt_bayes_aarch64 alog
    ${OpenCV_LIBS}
)