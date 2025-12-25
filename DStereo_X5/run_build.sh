#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

echo "=> ================="
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. \
  -DCMAKE_C_COMPILER=/opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++
make -j$(nproc)

echo "=> ================="
mkdir model
cp -r ../../config/DStereoV*.bin ./model/
cp -r ../img ./
mkdir -p ./3rdparty/lib_opencv4.5.4/
cp -r ../3rdparty/lib_opencv4.5.4/ ./3rdparty/
cp -r ../make_ln.sh ./
mkdir result
tar -zcf StereoInfer.tar.gz ./stereo_infer ./3rdparty ./model  ./img  ./make_ln.sh ./result
echo "=> output file: StereoInfer.tar.gz"

echo "=> ================="
md5sum ./stereo_infer
echo "=> ================="
