#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

echo "=> ================="
rm -rfv build
echo "=> ================="

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

echo "=> ================="
cp -rv ../../config/DStereoV2.4_int16.bin ./
cp -rv ../*.png ./
echo "=> ================="

