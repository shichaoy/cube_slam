# Description

This package contains line detection using LSD and Edline, line descriptors using LBD, line matching and visualization. Most of code is adapted from: https://github.com/opencv/opencv_contrib/tree/master/modules/line_descriptor
Actually, opencv3.0 already has line detection/matching implementation:
http://docs.opencv.org/3.0-beta/modules/line_descriptor/doc/tutorial.html


# How to run:
1. `catkin make`
2. `roslaunch line_lbd line_detect.launch`

In the launch file, choose to set some parameters.


# Maintainer and contacts
Shichao Yang (shichaoy@andrew.cmu.edu)
