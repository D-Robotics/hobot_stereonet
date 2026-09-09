#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

# ================= toolchain selection =================
#   (no arg)  : aarch64 native (deps under /usr), or cross-compile with
#               /usr/bin/aarch64-linux-gnu-* (deps under sysroot_docker/usr_x5)
#   arm       : interactive, prompt for the ARM GNU toolchain bin dir (buildroot users)
#   <bin_dir> : use <bin_dir> as the ARM GNU toolchain bin dir
ARCH=$(uname -m)
ARM_GNU_DEFAULT_DIR="/opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin"
BUNDLE_RUNTIME=0
if [ "$ARCH" = "aarch64" ]; then
    CC="gcc"
    CXX="g++"
    SYSROOT="/usr"
elif [ -n "$1" ]; then
    # ARM GNU toolchain requested (buildroot users).
    if [ "$1" = "arm" ]; then
        read -r -p "ARM GNU toolchain bin dir [default: $ARM_GNU_DEFAULT_DIR]: " ARM_DIR
        ARM_DIR="${ARM_DIR:-$ARM_GNU_DEFAULT_DIR}"
    else
        ARM_DIR="$1"
    fi
    CC="$ARM_DIR/aarch64-none-linux-gnu-gcc"
    CXX="$ARM_DIR/aarch64-none-linux-gnu-g++"
    if [ ! -x "$CC" ] || [ ! -x "$CXX" ]; then
        echo "!! ARM GNU toolchain not found in: $ARM_DIR"
        echo "   expected: $ARM_DIR/aarch64-none-linux-gnu-gcc"
        exit 1
    fi
    SYSROOT="$(cd ../../../../sysroot_docker/usr_x5 2>/dev/null && pwd)"
    # buildroot lacks the D-Robotics / OpenCV / gdal libraries, so the Arm GNU build
    # is self-contained: headers + the full runtime dependency closure live in the
    # committed standalone/3rdparty/ and are shipped in the package.
    BUNDLE_RUNTIME=1
else
    CC="/usr/bin/aarch64-linux-gnu-gcc"
    CXX="/usr/bin/aarch64-linux-gnu-g++"
    SYSROOT="$(cd ../../../../sysroot_docker/usr_x5 2>/dev/null && pwd)"
fi
# For the Arm GNU build the sysroot is only needed to (re)populate 3rdparty/ from
# scratch; once 3rdparty/ is committed, the build works without it. Every other
# cross build requires the sysroot.
if [ -z "$SYSROOT" ]; then
    if [ "$BUNDLE_RUNTIME" = "1" ]; then
        echo "=> sysroot not found; using the committed 3rdparty/ SDK as-is"
    else
        echo "!! sysroot not found: ../../../../sysroot_docker/usr_x5"
        exit 1
    fi
fi
echo "=> arch=$ARCH  CC=$CC  SYSROOT=${SYSROOT:-<none>}"

# ================= self-contained SDK (Arm GNU only) =================
# The functions below populate standalone/3rdparty/ from the sysroot so that the
# Arm GNU build does not depend on the sysroot for either compile (headers + link
# libraries) or run (runtime dependency closure). glibc's own libraries are skipped
# because every glibc-based buildroot already ships them, and bundling a Debian
# glibc next to a buildroot dynamic linker would cause version-skew crashes.

_find_lib() {
    local soname="$1" d
    for d in "hobot/lib" "lib/aarch64-linux-gnu" "lib" "usr/lib/aarch64-linux-gnu" "usr/lib"; do
        if [ -e "$SYSROOT/$d/$soname" ]; then
            echo "$SYSROOT/$d/$soname"
            return 0
        fi
    done
    find "$SYSROOT" -name "$soname" -print -quit 2>/dev/null
}

_is_glibc_lib() {
    case "$1" in
        ld-linux-aarch64.so.1|libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0| \
        librt.so.1|libresolv.so.2|libutil.so.1|libnsl.so.1|libanl.so.1| \
        libmvec.so.1|libnss_*.so.2|libthread_db.so.1) return 0 ;;
        *) return 1 ;;
    esac
}

_resolve_lib() {
    local file="$1" dep
    grep -qxF "$file" "$SEEN_FILE" && return
    echo "$file" >> "$SEEN_FILE"
    readelf -d "$file" 2>/dev/null \
        | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p' \
        | while IFS= read -r soname; do
            _is_glibc_lib "$soname" && continue
            dep="$(_find_lib "$soname")"
            if [ -z "$dep" ]; then
                echo "!! unresolved: $soname (required by $file)" >&2
                continue
            fi
            _resolve_lib "$dep"
        done
}

_classify_lib() {
    # Sort a bundled library into a category subdirectory. The D-Robotics BPU
    # libraries (libdnn/libhbrt_bayes_aarch64) live in $SYSROOT/lib, not in
    # hobot/lib, so they are matched by name in addition to the hobot/lib path.
    local name="${1##*/}"
    case "$1" in
        "$SYSROOT/hobot/lib/"*) echo "hobot" ;;
        *)
            case "$name" in
                libdnn.so*|libcnn_intf.so*|libhbmem.so*|libhbrt*|libalog.so*) echo "hobot" ;;
                libopencv_*) echo "opencv" ;;
                libgdal*)    echo "gdal" ;;
                *)           echo "deps" ;;
            esac ;;
    esac
}

_seed_lib() {
    # Resolve a library by its linker name (e.g. libopencv_core.so) to the file that
    # carries its SONAME (e.g. libopencv_core.so.4.5d), then record that file and its
    # full DT_NEEDED closure. This reproduces exactly what linking the executables
    # pulls in, so the bundle is complete without inspecting the built binary first.
    local dev="$1" real soname file
    real="$(_find_lib "$dev")"
    if [ -z "$real" ]; then
        echo "!! unresolved: $dev" >&2
        return 1
    fi
    soname="$(readelf -d "$real" 2>/dev/null | sed -n 's/.*Library soname: \[\([^]]*\)\].*/\1/p')"
    if [ -n "$soname" ]; then
        file="$(_find_lib "$soname")"
    else
        file="$real"
    fi
    [ -n "$file" ] && _resolve_lib "$file"
}

_collect_runtime_libs() {
    local out="$1" count=0 f sub
    shift
    # Clear the category subdirs (not the whole dir, which also holds include/).
    for sub in hobot opencv gdal deps; do
        rm -rf "$out/$sub"
    done
    mkdir -p "$out"
    SEEN_FILE="$(mktemp)"
    for dev in "$@"; do
        _seed_lib "$dev"
    done
    while IFS= read -r f; do
        sub="$(_classify_lib "$f")"
        mkdir -p "$out/$sub"
        cp -L --preserve=mode,timestamps "$f" "$out/$sub/" && count=$((count + 1))
    done < "$SEEN_FILE"
    rm -f "$SEEN_FILE"
    echo "=> bundled $count runtime libraries into $out (hobot/ opencv/ gdal/ deps/)"
}

_copy_headers() {
    # Copy the compile-time headers (Eigen, OpenCV 4.x, DNN/BPU) into 3rdparty/include/
    # so the Arm GNU build compiles without the sysroot.
    local out="$1"
    rm -rf "$out/include/eigen3" "$out/include/opencv4" "$out/include/dnn"
    mkdir -p "$out/include"
    cp -r "$SYSROOT/include/eigen3" "$out/include/"
    cp -r "$SYSROOT/include/opencv4" "$out/include/"
    cp -r "$SYSROOT/include/dnn" "$out/include/"
    echo "=> copied headers into $out/include (eigen3/ opencv4/ dnn/)"
}

_create_dev_symlinks() {
    # Create unversioned .so symlinks next to the bundled versioned libs so the linker
    # can resolve -ldnn / -lopencv_core / ... against 3rdparty at build time.
    local out="$1" dir f base target
    for dir in hobot opencv; do
        [ -d "$out/$dir" ] || continue
        for f in "$out/$dir"/*.so.*; do
            [ -e "$f" ] || continue
            base="${f##*/}"
            target="${base%%.so.*}"
            [ -n "$target" ] || continue
            ln -sf "$base" "$out/$dir/${target}.so"
        done
    done
    echo "=> created unversioned .so symlinks in $out/hobot and $out/opencv"
}

# Libraries linked by the Arm GNU build; their full DT_NEEDED closure is what gets
# bundled into 3rdparty/. Keep this in sync with target_link_libraries in CMakeLists.
LINK_LIBS="libdnn.so libcnn_intf.so libhbmem.so libhbrt_bayes_aarch64.so libalog.so \
libopencv_core.so libopencv_imgproc.so libopencv_imgcodecs.so libopencv_features2d.so libopencv_flann.so"

# Populate 3rdparty/ from the sysroot (headers + libs + dev symlinks). Only run when
# the sysroot is present; otherwise the committed 3rdparty/ is used as-is.
if [ "$BUNDLE_RUNTIME" = "1" ] && [ -n "$SYSROOT" ] && [ -d "$SYSROOT" ]; then
    _copy_headers ./3rdparty
    _collect_runtime_libs ./3rdparty $LINK_LIBS
    _create_dev_symlinks ./3rdparty
fi

echo "=> ================="
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. \
  -DPLATFORM_X5=ON \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DSYSROOT=${SYSROOT:-/usr}
make -j$(nproc)

echo "=> ================="
mkdir model
cp -r ../../config/DStereoV*.bin ./model/
cp -r ../img ./
mkdir result
if [ "$BUNDLE_RUNTIME" = "1" ]; then
    tar -zcf StereoInfer_X5.tar.gz \
    --transform 's,^\./,StereoInfer/,' \
    --transform 's,^\.\./,StereoInfer/,' \
    ./test_perf ./infer ./model  ./img  ./result  ../3rdparty
else
    tar -zcf StereoInfer_X5.tar.gz \
    --transform 's,^,StereoInfer/,' \
    ./test_perf ./infer ./model  ./img  ./result
fi
echo "=> output file: StereoInfer_X5.tar.gz"

echo "=> ================="
md5sum ./infer
md5sum ./test_perf
echo "=> ================="
