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
  -DCMAKE_C_COMPILER=/opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/opt/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++
  # -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
  # -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++
make -j$(nproc)

echo "=> ================="
mkdir model
cp -r ../../config/DStereoV*.bin ./model/
cp -r ../img ./
mkdir -p ./3rdparty/lib_opencv4.5.4/
cp -r ../3rdparty/lib_opencv4.5.4/ ./3rdparty/
cp -r ../make_ln.sh ./
mkdir result
tar -zcf StereoInfer_X5.tar.gz  \
--transform 's,^,StereoInfer/,' \
./test_perf ./infer ./3rdparty ./model  ./img  ./make_ln.sh ./result
echo "=> output file: StereoInfer_X5.tar.gz"

echo "=> ================="
md5sum ./infer
md5sum ./test_perf
echo "=> ================="
