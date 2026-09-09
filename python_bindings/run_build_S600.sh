#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

# Native build on the RDK S600 board (deps under /usr, no cross-compile).
echo "=> ================="
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. -DPLATFORM_S600=ON
make -j$(nproc)
