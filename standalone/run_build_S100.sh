#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

# ================= auto-detect build environment =================
#   aarch64 : native build on the RDK board (deps under /usr)
#   x86_64  : cross-compile (deps under sysroot_docker/usr_s100)
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    CC="gcc"
    CXX="g++"
    SYSROOT="/usr"
else
    CC="/usr/bin/aarch64-linux-gnu-gcc"
    CXX="/usr/bin/aarch64-linux-gnu-g++"
    SYSROOT="$(cd ../../../../sysroot_docker/usr_s100 && pwd)"
    if [ ! -d "$SYSROOT" ]; then
        echo "!! sysroot not found: $SYSROOT"
        exit 1
    fi
fi
echo "=> arch=$ARCH  CC=$CC  SYSROOT=$SYSROOT"

echo "=> ================="
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. \
  -DPLATFORM_S100=ON \
  -DCMAKE_C_COMPILER=$CC \
  -DCMAKE_CXX_COMPILER=$CXX \
  -DSYSROOT=$SYSROOT
make -j$(nproc)

echo "=> ================="
mkdir model
cp -r ../../config/dstereo_s100*.hbm ./model/
cp -r ../img ./
mkdir result
tar -zcf StereoInfer_S100.tar.gz \
--transform 's,^,StereoInfer/,' \
./test_perf ./infer ./model  ./img  ./result
echo "=> output file: StereoInfer_S100.tar.gz"

echo "=> ================="
md5sum ./test_perf
md5sum ./infer
echo "=> ================="
