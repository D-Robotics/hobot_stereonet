cmake_minimum_required(VERSION 3.10)
project(StereoInfer)

set(CMAKE_CXX_STANDARD 17)

# Release
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Debug
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# =================================== set platform ====================================
if(PLATFORM_X5)
    add_definitions(-DPLATFORM_X5)
elseif(PLATFORM_S100)
    add_definitions(-DPLATFORM_S100)
endif()

# =================================== find packages ===================================
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/lib_opencv4.5.4/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

# =================================== FetchContent ====================================
# include(FetchContent)
# set(FETCHCONTENT_QUIET OFF)
# FetchContent_Declare(
#     magic_enum
#     GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
#     GIT_TAG v0.9.7
# )
# FetchContent_Declare(
#     concurrentqueue
#     GIT_REPOSITORY https://github.com/cameron314/concurrentqueue.git
#     GIT_TAG master
# )
# FetchContent_Declare(
#     bs_thread_pool
#     GIT_REPOSITORY https://github.com/bshoshany/thread-pool.git
#     GIT_TAG master
# )
# FetchContent_MakeAvailable(magic_enum concurrentqueue bs_thread_pool)

set(concurrentqueue_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/concurrentqueue)
set(magic_enum_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/magic_enum/include)
set(bs_thread_pool_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/thread-pool/include)

# =================================== set include & link dirs =========================
include_directories(
    ../include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen3
    ${concurrentqueue_DIR}
    ${magic_enum_DIR}
    ${bs_thread_pool_DIR}
)
# link_directories(
#     ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn
#     ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn/hobot/lib
#     ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/lib
# )

if(PLATFORM_X5)
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn
    )
    link_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libdnn/hobot/lib
    )
elseif(PLATFORM_S100)
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/include
    )
    link_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/lib
    )
endif()

# =================================== build executable ================================
add_executable(test_perf
    test_perf.cpp
    ../src/stereonet_process.cpp
    ../src/timer_utils.cpp
    ../src/img_convert_utils.cpp
)
if(PLATFORM_X5)
    target_link_libraries(test_perf dnn cnn_intf hbmem hbrt_bayes_aarch64 alog ${OpenCV_LIBS})
elseif(PLATFORM_S100)
    target_link_libraries(test_perf dnn hbucp ${OpenCV_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libalog.so.1.2.2
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libbpu.so.2
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libcjson.so.1.7.15
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libhbipcfhal.so.1.0.0
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libhbmem.so.1.0.0
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libvdsp.so.1.0.0
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libjsoncpp.so.1.9.5
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libjsoncpp.so.1.9.0
    )
endif()

add_executable(infer
    infer.cpp
    ../src/stereonet_process.cpp
    ../src/timer_utils.cpp
    ../src/img_convert_utils.cpp
    ../src/file_utils.cpp
    ../src/feature_epipolar_align.cpp
)
if(PLATFORM_X5)
    target_link_libraries(infer dnn cnn_intf hbmem hbrt_bayes_aarch64 alog ${OpenCV_LIBS})
elseif(PLATFORM_S100)
    target_link_libraries(infer dnn hbucp ${OpenCV_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libalog.so.1.2.2
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libbpu.so.2
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libcjson.so.1.7.15
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libhbipcfhal.so.1.0.0
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libhbmem.so.1.0.0
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libvdsp.so.1.0.0
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libjsoncpp.so.1.9.5
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ucp_3.13.6/libjsoncpp.so.1.9.0
    )
endif()
