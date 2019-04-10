# Cube SLAM #
This code contains a basic implementation for Cube SLAM. Given RGB and 2D object detection, the algorithm detects 3D cuboids from each frame then formulate an object SLAM to optimize both camera pose and cuboid poses. ```object_slam``` is main package. ```detect_3d_cuboid``` is the C++ version of single image cuboid detection, corresponding to a [matlab version](https://github.com/shichaoy/matlab_cuboid_detect).

**Authors:** [Shichao Yang](https://shichaoy.github.io./)

**Related Paper:**

* **CubeSLAM: Monocular 3D Object SLAM**, IEEE Transactions on Robotics 2019, S. Yang, S. Scherer  [**PDF**](https://arxiv.org/abs/1806.00557)

If you use the code in your research work, please cite the above paper. Feel free to contact the authors if you have any further questions.



## Installation

### Prerequisites
This code contains several ros packages. We test it in **ROS indigo/kinetic, Ubuntu 14.04/16.04, Opencv 2/3**. Create or use existing a ros workspace.
```bash
mkdir -p ~/cubeslam_ws/src
cd ~/cubeslam_ws/src
catkin_init_workspace
git clone git@github.com:shichaoy/cube_slam.git
cd cube_slam
```

### Compile dependency g2o
```bash
sh install_dependenices.sh
```


### Compile
```bash
cd ~/cubeslam_ws
catkin_make -j4
```


## Running #
```bash
source devel/setup.bash
roslaunch object_slam object_slam_example.launch
```
You will see results in Rviz. Default rviz file is for ros indigo. A kinetic version is also provided.


### Notes

1. This package utilizes cuboid object as the only SLAM landmark. The integration with ORB SLAM point features is not provided yet. Therefore, it may not produce accurate results for general scenarios.

2. In the launch file (```object_slam_example.launch```), if ```online_detect_mode=false```, it requires the matlab saved cuboid images, cuboid pose txts and camera pose txts.  if ```true```, it reads the 2D object bounding box txt then online detects 3D cuboids poses using C++.

3. ```object_slam/data/``` contains all the preprocessing data. ```depth_imgs/``` is just for visualization. ```pred_3d_obj_overview/``` is the offline matlab cuboid detection images. ```detect_cuboids_saved.txt``` is the offline cuboid poses in local ground frame, in the format "3D position, 1D yaw, 3D scale, score". ```pop_cam_poses_saved.txt``` is the camera poses to generate offline cuboids (camera x/y/yaw = 0, truth camera roll/pitch/height) ```truth_cam_poses.txt``` is mainly used for visulization and comparison.

	```filter_2d_obj_txts/``` is the 2D object bounding box txt. We use Yolo to detect 2D objects. Other similar methods can also be used. ```preprocessing/2D_object_detect``` is our prediction code to save images and txts. Sometimes there might be overlapping box of the same object instance. We need to filter and clean some detections. See the [```filter_match_2d_boxes.m```](https://github.com/shichaoy/matlab_cuboid_detect/blob/master/filter_match_2d_boxes.m) in our matlab detection package.
