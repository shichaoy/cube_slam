# Description #
This package contains to single image cuboid detection in C++. Given 2D object detection, it generates many 3D cuboid proposal and selects the best one. It matches a [matlab implementation](https://github.com/shichaoy/matlab_cuboid_detect). Due to different canny edge and distancen transform, the final output might be slightly differently. For understanding and debuging purpose, it is suggested to use matlab implementation.



**Authors:** Shichao Yang (shichaoy@andrew.cmu.edu)

# How to run.
1. catkin_make.
2. `rosrun detect_3d_cuboid detect_3d_cuboid_node`

See the main file ```src/main.cpp``` for more details.
