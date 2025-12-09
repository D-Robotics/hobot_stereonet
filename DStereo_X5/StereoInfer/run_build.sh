#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

# echo "=> ================="
# rm -rfv build
# echo "=> ================="

# mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. \
  -DCMAKE_C_COMPILER=/root/dockershare/1_RosCode/work_humble_ws_x5/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/root/dockershare/1_RosCode/work_humble_ws_x5/compiler/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++
make -j$(nproc)

echo "=> ================="
cp -rv ../../../config/DStereoV*.bin ./
cp -rv ../img ./
echo "=> ================="
md5sum ./stereo_infer
echo "=> ================="
