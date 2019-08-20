#pragma once

// std c
#include <string>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>

#include "detect_3d_cuboid/matrix_utils.h"

class cuboid // matlab cuboid struct. cuboid on ground. only has yaw, no obj roll/pitch
{
    public:
      Eigen::Vector3d pos;
      Eigen::Vector3d scale;
      double rotY;

      Eigen::Vector2d box_config_type;       // configurations, vp1 left/right
      Eigen::Matrix2Xi box_corners_2d;       // 2*8
      Eigen::Matrix3Xd box_corners_3d_world; // 3*8

      Eigen::Vector4d rect_detect_2d; //% 2D bounding box (might be expanded by me)
      double edge_distance_error;
      double edge_angle_error;
      double normalized_error; // normalized distance+angle
      double skew_ratio;
      double down_expand_height;
      double camera_roll_delta;
      double camera_pitch_delta;

      void print_cuboid(); // print pose information
};
typedef std::vector<cuboid *> ObjectSet; // for each 2D box, the set of generated 3D cuboids

struct cam_pose_infos
{
      Eigen::Matrix4d transToWolrd;
      Eigen::Matrix3d Kalib;

      Eigen::Matrix3d rotationToWorld;
      Eigen::Vector3d euler_angle;
      Eigen::Matrix3d invR;
      Eigen::Matrix3d invK;
      Eigen::Matrix<double, 3, 4> projectionMatrix;
      Eigen::Matrix3d KinvR; // K*invR
      double camera_yaw;
};

class detect_3d_cuboid
{
    public:
      cam_pose_infos cam_pose;
      cam_pose_infos cam_pose_raw;
      void set_calibration(const Eigen::Matrix3d &Kalib);
      void set_cam_pose(const Eigen::Matrix4d &transToWolrd);

      // object detector needs image, camera pose, and 2D bounding boxes(n*5, each row: xywh+prob)  long edges: n*4.  all number start from 0
      void detect_cuboid(const cv::Mat &rgb_img, const Eigen::Matrix4d &transToWolrd, const Eigen::MatrixXd &obj_bbox_coors, Eigen::MatrixXd edges,
                         std::vector<ObjectSet> &all_object_cuboids);

      bool whether_plot_detail_images = false;
      bool whether_plot_final_images = false;
      bool whether_save_final_images = false;
      cv::Mat cuboids_2d_img; // save to this opencv mat
      bool print_details = false;

      // important mode parameters for proposal generation.
      bool consider_config_1 = true; // false true
      bool consider_config_2 = true;
      bool whether_sample_cam_roll_pitch = false; // sample camera roll pitch in case don't have good camera pose
      bool whether_sample_bbox_height = false;    // sample object height as raw detection might not be accurate

      int max_cuboid_num = 1;        //final return best N cuboids
      double nominal_skew_ratio = 1; // normally this 1, unless there is priors
      double max_cut_skew = 3;
};