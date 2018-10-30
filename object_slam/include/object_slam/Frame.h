#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "object_slam/g2o_Object.h"

class object_landmark;

class tracking_frame{
public:  

    int frame_seq_id;    // image topic sequence id, fixed
    cv::Mat frame_img;
    cv::Mat cuboids_2d_img;
    
    g2o::VertexSE3Expmap* pose_vertex;
    
    std::vector<object_landmark*> observed_cuboids; // generated cuboid from this frame. maynot be actual SLAM landmark
    
    g2o::SE3Quat cam_pose_Tcw;	     // optimized pose  world to cam
    g2o::SE3Quat cam_pose_Twc;	     // optimized pose  cam to world
    
};