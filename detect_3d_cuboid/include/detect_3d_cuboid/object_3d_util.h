#pragma once

#include <string>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>

#include "detect_3d_cuboid/detect_3d_cuboid.h"

using namespace Eigen;

// NOTE all the functions here correspond to matlab functions.

Eigen::Matrix3Xd compute3D_BoxCorner(const cuboid &cube_obj);

// ploting functions
void get_object_edge_visibility(MatrixXi &visible_hidden_edge_pts, const Vector2d &box_config_type, bool new_object_type = false);
void get_cuboid_draw_edge_markers(MatrixXi &edge_markers, const Vector2d &box_config_type, bool new_object_type = false);
void plot_image_with_cuboid_edges(cv::Mat &plot_img, const MatrixXi &box_corners_2d, const MatrixXi &edge_markers);
void plot_image_with_cuboid(cv::Mat &plot_img, const cuboid *cube_obj);
// each line is x1 y1 x2 y2   color: Scalar(255,0,0) eg
void plot_image_with_edges(const cv::Mat &rgb_img, cv::Mat &output_img, MatrixXd &all_lines, const cv::Scalar &color);

// check if point lies inside 2d rect
bool check_inside_box(const Vector2d &pt, const Vector2d &box_left_top, const Vector2d &box_right_bottom);

// make sure edges start from left to right
void align_left_right_edges(MatrixXd &all_lines);

// line intersection functions
Vector2d seg_hit_boundary(const Vector2d &pt_start, const Vector2d &pt_end, const Vector4d &line_segment2);
Vector2d lineSegmentIntersect(const Vector2d &pt1_start, const Vector2d &pt1_end, const Vector2d &pt2_start, const Vector2d &pt2_end,
							  bool infinite_line = true);
Vector2f lineSegmentIntersect_f(const Vector2f &pt1_start, const Vector2f &pt1_end, const Vector2f &pt2_start, const Vector2f &pt2_end,
								float &extcond_1, float &extcond_2, bool infinite_line = true);
cv::Point2f lineSegmentIntersect_f(const cv::Point2f &pt1_start, const cv::Point2f &pt1_end, const cv::Point2f &pt2_start, const cv::Point2f &pt2_end,
								   float &extcond_1, float &extcond_2, bool infinite_line = true);

void normalize_to_pi_vec(const VectorXd &raw_angles, VectorXd &new_angles);

// (-pi,pi]
void atan2_vector(const VectorXd &y_vec, const VectorXd &x_vec, VectorXd &all_angles);

void smooth_jump_angles(const VectorXd &raw_angles, VectorXd &new_angles);

// merge then remove short lines
void merge_break_lines(const MatrixXd &all_lines, MatrixXd &merge_lines_out, double pre_merge_dist_thre, double pre_merge_angle_thre_degree,
					   double edge_length_threshold = -1);

// find VP supported lines
Eigen::MatrixXd VP_support_edge_infos(Eigen::MatrixXd &VPs, Eigen::MatrixXd &edge_mid_pts, Eigen::VectorXd &edge_angles,
									  Eigen::Vector2d vp_support_angle_thres);

// two important errors
double box_edge_sum_dists(const cv::Mat &dist_map, const MatrixXd &box_corners_2d, const MatrixXi &edge_pt_ids, bool reweight_edge_distance = false);

double box_edge_alignment_angle_error(const MatrixXd &all_vp_bound_edge_angles, const MatrixXi &vps_box_edge_pt_ids, const MatrixXd &box_corners_2d);

// fuse and normalize different scores
void fuse_normalize_scores_v2(const VectorXd &dist_error, const VectorXd &angle_error, VectorXd &combined_scores, std::vector<int> &final_keep_inds,
							  double weight_vp_angle, bool whether_normalize);

void ray_plane_interact(const MatrixXd &rays, const Eigen::Vector4d &plane, MatrixXd &intersections);

// KinvR = K*invR
void getVanishingPoints(const Matrix3d &KinvR, double yaw_esti, Vector2d &vp_1, Vector2d &vp_2, Vector2d &vp_3);

void change_2d_corner_to_3d_object(const MatrixXd &box_corners_2d_float, const Vector3d &configs, const Vector4d &ground_plane_sensor,
								   const Matrix4d &transToWolrd, const Matrix3d &invK, Eigen::Matrix<double, 3, 4> &projectionMatrix, cuboid &sample_obj);

// compute IoU overlap ratio between two rectangles [x y w h]
float bboxOverlapratio(const cv::Rect &rect1, const cv::Rect &rect2);
int pointBoundaryDist(const cv::Rect &rect, const cv::Point2f &kp);