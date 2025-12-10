cmake_minimum_required(VERSION 3.10)
project(StereoInfer)

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

# =================================== FetchContent ====================================
include(FetchContent)
FetchContent_Declare(
    magic_enum
    GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
    GIT_TAG v0.9.7
)
FetchContent_Declare(
    concurrentqueue
    GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
    GIT_TAG master
)
FetchContent_Declare(
    bs_thread_pool
    GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
    GIT_TAG master
)
FetchContent_MakeAvailable(magic_enum concurrentqueue bs_thread_pool)

# =================================== set include & link dirs =========================
include_directories(
    ../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn
)
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn/hobot/lib
)

# =================================== build executable ================================
add_executable(stereo_infer
    main.cpp
    ../../src/stereonet_process.cpp
    ../../src/timer_utils.cpp
    ../../src/img_convert_utils.cpp
)
target_include_directories(stereo_infer PUBLIC
    ${concurrentqueue_SOURCE_DIR}
    ${bs_thread_pool_SOURCE_DIR}/include
)
target_link_libraries(stereo_infer dnn cnn_intf hbmem hbrt_bayes_aarch64 alog ${OpenCV_LIBS} magic_enum::magic_enum)
