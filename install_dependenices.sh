#!/bin/bash

cd object_slam/Thirdparty/g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2