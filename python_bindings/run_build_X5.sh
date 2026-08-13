#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

echo "=> ================="
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. \
  -DPLATFORM_X5=ON \
  -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++
make -j$(nproc)
