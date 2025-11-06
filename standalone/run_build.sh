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
mkdir -p ./3rdparty/lib_opencv4.5.4/
cp -rvf ../3rdparty/lib_opencv4.5.4/ ./3rdparty/
tar -cvf StereoInfer.tar ./3rdparty ../../config/DStereoV2.4_int16.bin  ../../config/DStereoV2.4_int16_uncertainty.bin ./StereoInfer ../*.png  ../make_ln.sh
echo "=> ================="
