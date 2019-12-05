#!/bin/bash

cd object_slam/Thirdparty/g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
cd ../../../..

cd orb_object_slam/Thirdparty/g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
cd ../../../..

cd orb_object_slam/Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2
cd ../../../..
