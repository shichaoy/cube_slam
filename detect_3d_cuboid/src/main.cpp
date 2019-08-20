// std c
#include <stdio.h>
#include <iostream>
#include <string>

// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

// ros
#include <ros/ros.h>
#include <ros/package.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// ours
#include "detect_3d_cuboid/detect_3d_cuboid.h"
#include "tictoc_profiler/profiler.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "detect_3d_cuboid");
    ros::NodeHandle nh;
    ca::Profiler::enable();

    string base_folder = ros::package::getPath("detect_3d_cuboid") + "/data/";

    Matrix3d Kalib;
    Kalib << 529.5000, 0, 365.0000,
        0, 529.5000, 265.0000,
        0, 0, 1.0000;

    Matrix4d transToWolrd;
    transToWolrd << 1, 0.0011, 0.0004, 0, // hard coded  NOTE if accurate camera roll/pitch, could sample it!
        0, -0.3376, 0.9413, 0,
        0.0011, -0.9413, -0.3376, 1.35,
        0, 0, 0, 1;

    MatrixXd obj_bbox_coors(1, 5);                // hard coded
    obj_bbox_coors << 188, 189, 201, 311, 0.8800; // [x y w h prob]
    obj_bbox_coors.leftCols<2>().array() -= 1;    // change matlab coordinate to c++, minus 1

    int frame_index = 0;
    char frame_index_c[256];
    sprintf(frame_index_c, "%04d", frame_index); // format into 4 digit

    // read images
    cv::Mat rgb_img = cv::imread(base_folder + frame_index_c + "_rgb_raw.jpg", 1);

    // read edges
    Eigen::MatrixXd all_lines_raw(100, 4); // 100 is some large frame number,   the txt edge index start from 0
    read_all_number_txt(base_folder + "edge_detection/LSD/" + frame_index_c + "_edge.txt", all_lines_raw);

    detect_3d_cuboid detect_cuboid_obj;
    detect_cuboid_obj.whether_plot_detail_images = false;
    detect_cuboid_obj.whether_plot_final_images = true;
    detect_cuboid_obj.print_details = false; // false  true
    detect_cuboid_obj.set_calibration(Kalib);
    detect_cuboid_obj.whether_sample_bbox_height = false;
    detect_cuboid_obj.whether_sample_cam_roll_pitch = false;

    std::vector<ObjectSet> all_object_cuboids;
    detect_cuboid_obj.detect_cuboid(rgb_img, transToWolrd, obj_bbox_coors, all_lines_raw, all_object_cuboids);

    ca::Profiler::print_aggregated(std::cout);
    return 0;
}