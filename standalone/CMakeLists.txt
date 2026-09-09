cmake_minimum_required(VERSION 3.10)
project(StereoInfer)

set(CMAKE_CXX_STANDARD 17)

# Do not embed rpath: shared libraries are resolved at runtime by the board's
# ldconfig (/etc/ld.so.conf.d already covers /usr/hobot/lib and /usr/lib/aarch64-linux-gnu).
set(CMAKE_SKIP_RPATH TRUE)

# Release
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Debug
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

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

# The Arm GNU Toolchain (aarch64-none-linux-gnu-*) build is fully self-contained:
# headers, link libraries and the runtime dependency closure all come from the
# committed standalone/3rdparty/, not from the sysroot. This lets buildroot users
# build and run without the D-Robotics sysroot once 3rdparty/ is populated.
if(CMAKE_CXX_COMPILER MATCHES "aarch64-none-linux-gnu")
    set(ARM_GNU TRUE)
else()
    set(ARM_GNU FALSE)
endif()

if(ARM_GNU)
    # The sysroot's prebuilt C++ libs (libgdal.so.30 via opencv_imgcodecs, libvtk* via
    # opencv_viz) reference GLIBCXX_3.4.30, which only GCC >= 12 provides. The ARM GNU
    # toolchain (GCC 11.3.1) tops out at GLIBCXX_3.4.29, so defer those versioned
    # references to runtime: the bundled libstdc++ in 3rdparty/deps (>= 3.4.30)
    # resolves them.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
    # Both compile and run depend on 3rdparty/, sorted by run_build_X5.sh into
    # hobot/ opencv/ gdal/ deps/ (libraries) and include/ (headers):
    #   - link_directories + -rpath-link let the linker resolve the bundled libs
    #     (hobot BPU libs, OpenCV, gdal and its transitive deps).
    #   - the $ORIGIN-relative rpath lets the deployed executables find them at
    #     runtime. --disable-new-dtags forces DT_RPATH instead of DT_RUNPATH:
    #     DT_RUNPATH only applies to the executable's own DT_NEEDED, whereas
    #     DT_RPATH also applies to each bundled library's transitive dependencies
    #     (e.g. libgdal.so.30 and its ~114 deps).
    link_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/hobot
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gdal
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/deps
    )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/hobot:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gdal:${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/deps")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,\$ORIGIN/3rdparty/hobot:\$ORIGIN/3rdparty/opencv:\$ORIGIN/3rdparty/gdal:\$ORIGIN/3rdparty/deps -Wl,--disable-new-dtags")
else()
    # Let the linker resolve transitive dependencies (SONAME) of shared libraries:
    # libopencv_core needs libtbb.so.2/libz.so.1, libhbipcfhal needs libjsoncpp.so.25,
    # libopencv_imgcodecs needs libgdal.so.30 (under ${SYSROOT}/lib).
    # Use -rpath-link instead of -L: it only resolves transitive deps at link time, does not
    # write an rpath into the output, and does not shadow the cross toolchain's own libc.
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${SYSROOT}/lib:${SYSROOT}/lib/aarch64-linux-gnu")
endif()

# =================================== find packages ===================================
if(NOT ARM_GNU)
    set(OpenCV_DIR ${SYSROOT}/lib/aarch64-linux-gnu/cmake/opencv4)
    # Only pull the OpenCV modules actually used. This keeps opencv_viz (-> VTK) out of the
    # link; VTK, like libgdal.so.30 above, was built with a newer GCC and needs GLIBCXX_3.4.30.
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d)
else()
    # Self-contained build: link the bundled OpenCV modules by name. The unversioned
    # .so symlinks next to the bundled libs are created by run_build_X5.sh.
    set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_features2d opencv_flann)
endif()

set(concurrentqueue_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/concurrentqueue)
set(magic_enum_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/magic_enum/include)
set(bs_thread_pool_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/thread-pool/include)

# =================================== set include & link dirs =========================
if(ARM_GNU)
    include_directories(
        ../include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/include/eigen3
        ${concurrentqueue_DIR}
        ${magic_enum_DIR}
        ${bs_thread_pool_DIR}
    )
    # DNN (BPU) headers live under 3rdparty/include/dnn, OpenCV 4.x under
    # 3rdparty/include/opencv4. Add opencv4 before include so the 4.x headers win.
    include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/include/opencv4)
    include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/include)
else()
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
endif()

if(PLATFORM_X5)
    if(NOT ARM_GNU)
        link_directories(
            ${SYSROOT}/lib
            ${SYSROOT}/hobot/lib
        )
    endif()
elseif(PLATFORM_S100 OR PLATFORM_S600)
    # Only link hobot/lib. Transitive deps of libdnn/libhbucp (bpu/cjson/jsoncpp/hbmem/alog/vdsp/hbipcfhal)
    # are resolved at runtime by ldconfig via their DT_NEEDED; they do not need to be listed at link
    # time, and this avoids exposing the sysroot system lib dir (-L) to the linker, which would shadow
    # the cross toolchain's own libc.
    link_directories(
        ${SYSROOT}/hobot/lib
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
elseif(PLATFORM_S100 OR PLATFORM_S600)
    target_link_libraries(test_perf dnn hbucp ${OpenCV_LIBS})
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
elseif(PLATFORM_S100 OR PLATFORM_S600)
    target_link_libraries(infer dnn hbucp ${OpenCV_LIBS})
endif()
