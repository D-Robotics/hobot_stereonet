cmake_minimum_required(VERSION 3.10)
project(stereonet_pybind)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not embed rpath: shared libraries are resolved at runtime by the board's
# ldconfig (/etc/ld.so.conf.d already covers /usr/hobot/lib and /usr/lib/aarch64-linux-gnu).
set(CMAKE_SKIP_RPATH TRUE)

# =================================== set platform ====================================
if(PLATFORM_X5)
    add_definitions(-DPLATFORM_X5)
elseif(PLATFORM_S100)
    add_definitions(-DPLATFORM_S100)
elseif(PLATFORM_S600)
    add_definitions(-DPLATFORM_S600)
endif()

# =================================== sysroot =========================================
# SYSROOT points to the board's "/usr" (or its mirror under sysroot_docker/usr_xxx on
# the cross-compile host).
#   - cross-compile: -DSYSROOT=<sysroot_docker>/usr_x5 (or usr_s100 / usr_s600)
#   - native build : -DSYSROOT=/usr (default)
if(NOT DEFINED SYSROOT OR SYSROOT STREQUAL "")
    set(SYSROOT "/usr")
endif()
message(STATUS "SYSROOT = ${SYSROOT}")

# Let the linker resolve transitive dependencies (SONAME) of shared libraries:
# libopencv_core needs libtbb.so.2/libz.so.1, libhbipcfhal needs libjsoncpp.so.25,
# libopencv_imgcodecs needs libgdal.so.30 (under ${SYSROOT}/lib).
# Use -rpath-link instead of -L: it only resolves transitive deps at link time, does not
# write an rpath into the output, and does not shadow the cross toolchain's own libc.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link,${SYSROOT}/lib:${SYSROOT}/lib/aarch64-linux-gnu")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-rpath-link,${SYSROOT}/lib:${SYSROOT}/lib/aarch64-linux-gnu")

# =================================== find packages ===================================
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
set(OpenCV_DIR ${SYSROOT}/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV REQUIRED)

set(concurrentqueue_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/concurrentqueue)
set(magic_enum_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/magic_enum/include)
set(bs_thread_pool_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/thread-pool/include)

# =================================== set include dirs ================================
include_directories(
    ../include
    ${SYSROOT}/include/eigen3
    ${concurrentqueue_DIR}
    ${magic_enum_DIR}
    ${bs_thread_pool_DIR}
)
# ${SYSROOT}/include contains stale OpenCV 3.x headers (include/opencv2) that would shadow
# the 4.x headers. Add both as SYSTEM includes, with opencv4 before include, so that the
# 4.x headers (include/opencv4/opencv2) take precedence.
include_directories(SYSTEM ${SYSROOT}/include/opencv4)
include_directories(SYSTEM ${SYSROOT}/include)

# =================================== link dirs =======================================
if(PLATFORM_X5)
    link_directories(
        ${SYSROOT}/lib
        ${SYSROOT}/hobot/lib
    )
elseif(PLATFORM_S100 OR PLATFORM_S600)
    # Only link hobot/lib. Transitive deps of libdnn/libhbucp are resolved at runtime by
    # ldconfig via their DT_NEEDED; they do not need to be listed at link time.
    link_directories(
        ${SYSROOT}/hobot/lib
    )
endif()

# =================================== build module ====================================
pybind11_add_module(dstereonet
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/stereonet_process.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/timer_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/img_convert_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/feature_epipolar_align.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stereonet_pybind.cpp
)

if(PLATFORM_X5)
    target_link_libraries(dstereonet PRIVATE dnn cnn_intf hbmem hbrt_bayes_aarch64 alog ${OpenCV_LIBS})
elseif(PLATFORM_S100 OR PLATFORM_S600)
    target_link_libraries(dstereonet PRIVATE dnn hbucp ${OpenCV_LIBS})
endif()
