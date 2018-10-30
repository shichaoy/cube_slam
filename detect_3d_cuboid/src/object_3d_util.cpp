#include "detect_3d_cuboid/object_3d_util.h"
#include "detect_3d_cuboid/matrix_utils.h"

#include <iostream>
// opencv
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"



using namespace Eigen;
using namespace std;

Matrix4d similarityTransformation(const cuboid& cube_obj)
{
    Matrix3d rot;
    rot<<cos(cube_obj.rotY),-sin(cube_obj.rotY),0,
	 sin(cube_obj.rotY),cos(cube_obj.rotY),0,
	 0, 0, 1;
    Matrix3d scale_mat = cube_obj.scale.asDiagonal();
    
    Matrix4d res=Matrix4d::Identity();
    res.topLeftCorner<3,3>() = rot*scale_mat;
    res.col(3).head(3) = cube_obj.pos;
    return res;
}

    
void cuboid::print_cuboid()
{
    std::cout<<"printing cuboids info...."<<std::endl;
    std::cout<<"pos   "<<pos.transpose()<<std::endl;
    std::cout<<"scale   "<<scale.transpose()<<std::endl;
    std::cout<<"rotY   "<<rotY<<std::endl;
    std::cout<<"box_config_type   "<<box_config_type.transpose()<<std::endl;
    std::cout<<"box_corners_2d \n"<<box_corners_2d<<std::endl;
    std::cout<<"box_corners_3d_world \n"<<box_corners_3d_world<<std::endl;
}


Matrix3Xd compute3D_BoxCorner(const cuboid& cube_obj)
{
    MatrixXd corners_body;corners_body.resize(3,8);
    corners_body<< 1, 1, -1, -1, 1, 1, -1, -1,
		    1, -1, -1, 1, 1, -1, -1, 1,
		  -1, -1, -1, -1, 1, 1, 1, 1;
    MatrixXd corners_world = homo_to_real_coord<double>(similarityTransformation(cube_obj)*real_to_homo_coord<double>(corners_body));
    return corners_world;
}

// Output: n*2  each row is a edge's start and end pt id. 
// box_config_type  [configuration_id, vp_1_on_left_or_right]      cuboid struct has this field.
void get_object_edge_visibility( MatrixXi& visible_hidden_edge_pts,const Vector2d& box_config_type, bool final_universal_object)
{
    visible_hidden_edge_pts.resize(12,2);
    if (final_universal_object){  // final saved cuboid struct
	if (box_config_type(0)==1){        // look at get_cuboid_face_ids to know the faces and pt id using my old box format
	    if (box_config_type(1)==1)
		visible_hidden_edge_pts<<3,4, 4,1, 4,8,    1,2, 2,3, 2,6, 1,5, 3,7, 5,6, 6,7, 7,8, 8,5;
	    else
		visible_hidden_edge_pts<<2,3, 3,4, 3,7,    1,2, 1,4, 2,6, 1,5, 4,8, 5,6, 6,7, 7,8, 8,5;
	}
	else
	    visible_hidden_edge_pts<<2,3, 3,4, 4,1, 3,7, 4,8,    1,2, 2,6, 1,5, 5,6, 6,7, 7,8, 8,5;      
    }
    else{  // 2D box corners index only used in cuboids genetation process
	if (box_config_type(0)==1)
	    visible_hidden_edge_pts<<7,8, 7,6, 7,1,    1,2, 2,3, 3,4, 4,1, 2,6, 3,5, 4,8, 5,8, 5,6; // hidden + visible
	else
	    visible_hidden_edge_pts<<7,8, 7,6, 7,1, 8,4, 8,5,    1,2, 2,3, 3,4, 4,1, 2,6, 3,5, 5,6;
    }
}


// output: edge_markers  each row [ edge_start_pt_id, edge_end_pt_id,  edge_marker_type_id in line_marker_type ]
// box_config_type  [configuration_id, vp_1_on_left_or_right]      cuboid struct has this field.
void get_cuboid_draw_edge_markers(MatrixXi& edge_markers, const Vector2d& box_config_type, bool final_universal_object)
{
    MatrixXi visible_hidden_edge_pts;
    get_object_edge_visibility(visible_hidden_edge_pts,box_config_type, final_universal_object);
    VectorXi edge_line_markers(12);
    if (final_universal_object)  // final saved cuboid struct  
    {
	if (box_config_type(0)==1){
	    if (box_config_type(1)==1)
		edge_line_markers<<4,2,6,3,1,5,5,5,3,1,3,1;		
	    else
		edge_line_markers<<2,4,6,3,1,5,5,5,3,1,3,1;
	}
	else
	    edge_line_markers<<2,4,2,6,6,3,5,5,3,1,3,1;
    }
    else  // 2D box corners index only used in cuboids genetation process
    {
	if (box_config_type(0)==1)
	    edge_line_markers<<4,2,6,1,3,1,3,5,5,5,1,3;   // each row: edge_start_id,edge_end_id,edge_marker_type_id
	else
	    edge_line_markers<<4,2,6,6,2,1,3,1,3,5,5,3;
    }
    
    edge_markers.resize(12,3);
    edge_markers<<visible_hidden_edge_pts,edge_line_markers;
    edge_markers.array() -=1;  // to match c++ index
}

// each line is x1 y1 x2 y2   color: Scalar(255,0,0) eg
void plot_image_with_cuboid_edges(cv::Mat& plot_img, const MatrixXi& box_corners_2d, const MatrixXi& edge_markers)
{
    MatrixXi line_markers(6,4); // each row is  BGR, line_thickness
    line_markers<<0,0,255,2, 0,0,255,1, 0,255,0,2, 0,255,0,1, 255,0,0,2, 255,0,0,1;
    
    for (int edge_id=0;edge_id<edge_markers.rows();edge_id++)
    {      
      VectorXi edge_conds=edge_markers.row(edge_id);
      cv::line(plot_img,cv::Point(box_corners_2d(0, edge_conds(0)),box_corners_2d(1, edge_conds(0))),
			  cv::Point(box_corners_2d(0, edge_conds(1)),box_corners_2d(1, edge_conds(1))), 
			  cv::Scalar(line_markers(edge_conds(2),0),line_markers(edge_conds(2),1),line_markers(edge_conds(2),2)),
			  line_markers(edge_conds(2),3), CV_AA, 0);
    }
}

void plot_image_with_cuboid(cv::Mat& plot_img, const cuboid* cube_obj)
{
    MatrixXi edge_markers;  get_cuboid_draw_edge_markers(edge_markers, cube_obj->box_config_type, true);
    plot_image_with_cuboid_edges(plot_img, cube_obj->box_corners_2d, edge_markers);
}


// each line is x1 y1 x2 y2   color: Scalar(255,0,0) eg
void plot_image_with_edges(const cv::Mat& rgb_img, cv::Mat& output_img, MatrixXd& all_lines, const cv::Scalar& color)
{
    output_img = rgb_img.clone();
    for (int i=0;i<all_lines.rows();i++)
	cv::line(output_img,cv::Point(all_lines(i,0),all_lines(i,1)),cv::Point(all_lines(i,2),all_lines(i,3)), cv::Scalar(255,0,0), 2, 8, 0);
}


bool check_inside_box(const Vector2d& pt, const Vector2d& box_left_top, const Vector2d& box_right_bottom)
{
     return box_left_top(0)<=pt(0) && pt(0)<=box_right_bottom(0) && box_left_top(1)<=pt(1) && pt(1)<=box_right_bottom(1);
}

// make sure edges start from left to right
void align_left_right_edges(MatrixXd& all_lines)
{
    for (int line_id=0;line_id<all_lines.rows();line_id++)
    {
	if (all_lines(line_id,2)<all_lines(line_id,0))
	{
	      Vector2d temp = all_lines.row(line_id).tail<2>();
	      all_lines.row(line_id).tail<2>() = all_lines.row(line_id).head<2>();
	      all_lines.row(line_id).head<2>() = temp;
	}
    }
}


void normalize_to_pi_vec(const VectorXd& raw_angles, VectorXd& new_angles)
{
    new_angles.resize(raw_angles.rows());
    for (int i=0;i<raw_angles.rows();i++)
	new_angles(i)=normalize_to_pi<double>(raw_angles(i));
}

void atan2_vector(const VectorXd& y_vec, const VectorXd& x_vec, VectorXd& all_angles)
{
    all_angles.resize(y_vec.rows());
    for (int i=0;i<y_vec.rows();i++)
	all_angles(i)=std::atan2(y_vec(i),x_vec(i));  // don't need normalize_to_pi, because my edges is from left to right, always [-90 90]
}

// remove the jumping angles from -pi to pi.   to make the raw angles smoothly change.
void smooth_jump_angles(const VectorXd& raw_angles,VectorXd& new_angles)
{
    new_angles = raw_angles;
    if (raw_angles.rows()==0)
        return;

    double angle_base = raw_angles(0);  // choose a new base angle.   (assume that the all the angles lie in [-pi pi] around the base)
    for (int i=0;i<raw_angles.rows();i++)
    {
        if ( (raw_angles(i)-angle_base)<-M_PI )
            new_angles(i) = raw_angles(i)+2*M_PI;
        else if ( (raw_angles(i)-angle_base)>M_PI )
            new_angles(i) = raw_angles(i)-2*M_PI;
    }
}

// line_1  4d  line_segment2 4d  the output is float point.
// compute the intersection of line_1 (from start to end) with line segments (not infinite line). if not found, return [-1 -1]
// the second line segments are either horizontal or vertical.   a simplified version of lineSegmentIntersect
Vector2d seg_hit_boundary(const Vector2d& pt_start, const Vector2d& pt_end, const Vector4d& line_segment2 )
{         
    Vector2d boundary_bgn = line_segment2.head<2>();
    Vector2d boundary_end = line_segment2.tail<2>();
  
    Vector2d direc = pt_end-pt_start;
    Vector2d hit_pt(-1,-1);
    
    // line equation is (p_u,p_v)+lambda*(delta_u,delta_v)  parameterized by lambda
    if ( boundary_bgn(1)==boundary_end(1) )   // if an horizontal edge
    {
        double lambd=(boundary_bgn(1)-pt_start(1))/direc(1);
        if (lambd>=0)  // along ray direction
	{
            Vector2d hit_pt_tmp = pt_start+lambd*direc;
            if ( (boundary_bgn(0)<=hit_pt_tmp(0)) && (hit_pt_tmp(0)<=boundary_end(0)) )  // inside the segments
	    {
                hit_pt = hit_pt_tmp;
                hit_pt(1)= boundary_bgn(1);  // floor operations might have un-expected things
	    }
	}
    }    
    if ( boundary_bgn(0)==boundary_end(0) )   // if an vertical edge
    {
        double lambd=(boundary_bgn(0)-pt_start(0))/direc(0);
        if (lambd>=0)  // along ray direction
	{
            Vector2d hit_pt_tmp = pt_start+lambd*direc;
            if ( (boundary_bgn(1)<=hit_pt_tmp(1)) && (hit_pt_tmp(1)<=boundary_end(1)) )  // inside the segments
	    {
                hit_pt = hit_pt_tmp;
                hit_pt(0)= boundary_bgn(0);  // floor operations might have un-expected things
	    }
	}
    }
    return hit_pt;
}

// compute two line intersection points, a simplified version compared to matlab
Vector2d lineSegmentIntersect(const Vector2d& pt1_start, const Vector2d& pt1_end, const Vector2d& pt2_start, const Vector2d& pt2_end, 
			      bool infinite_line)
{
    // treat as [x1 y1 x2 y2]    [x3 y3 x4 y4]
      double X2_X1 = pt1_end(0)-pt1_start(0);
      double Y2_Y1 = pt1_end(1)-pt1_start(1);
      double X4_X3 = pt2_end(0)-pt2_start(0);
      double Y4_Y3 = pt2_end(1)-pt2_start(1);
      double X1_X3 = pt1_start(0)-pt2_start(0);
      double Y1_Y3 = pt1_start(1)-pt2_start(1);
      double u_a = (X4_X3*Y1_Y3-Y4_Y3*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);
      double u_b = (X2_X1*Y1_Y3-Y2_Y1*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);      
      double INT_X = pt1_start(0)+X2_X1*u_a;
      double INT_Y = pt1_start(1)+Y2_Y1*u_a;
      double INT_B = double((u_a >= 0) && (u_a <= 1) && (u_b >= 0) && (u_b <= 1));
      if (infinite_line)
	  INT_B=1;
      
      return Vector2d(INT_X*INT_B, INT_Y*INT_B);      
}
Vector2f lineSegmentIntersect_f(const Vector2f& pt1_start, const Vector2f& pt1_end, const Vector2f& pt2_start, const Vector2f& pt2_end,
			      float& extcond_1, float& extcond_2, bool infinite_line)
{
    // treat as [x1 y1 x2 y2]    [x3 y3 x4 y4]
      float X2_X1 = pt1_end(0)-pt1_start(0);
      float Y2_Y1 = pt1_end(1)-pt1_start(1);
      float X4_X3 = pt2_end(0)-pt2_start(0);
      float Y4_Y3 = pt2_end(1)-pt2_start(1);
      float X1_X3 = pt1_start(0)-pt2_start(0);
      float Y1_Y3 = pt1_start(1)-pt2_start(1);
      float u_a = (X4_X3*Y1_Y3-Y4_Y3*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);
      float u_b = (X2_X1*Y1_Y3-Y2_Y1*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);      
      float INT_X = pt1_start(0)+X2_X1*u_a;
      float INT_Y = pt1_start(1)+Y2_Y1*u_a;
      float INT_B = float((u_a >= 0) && (u_a <= 1) && (u_b >= 0) && (u_b <= 1));
      if (infinite_line)
	  INT_B=1;
      
      extcond_1 = u_a; extcond_2 = u_b;
      return Vector2f(INT_X*INT_B, INT_Y*INT_B);      
}

cv::Point2f lineSegmentIntersect_f(const cv::Point2f& pt1_start, const cv::Point2f& pt1_end, const cv::Point2f& pt2_start, const cv::Point2f& pt2_end, 
			      float& extcond_1, float& extcond_2, bool infinite_line)
{
    // treat as [x1 y1 x2 y2]    [x3 y3 x4 y4]
      float X2_X1 = pt1_end.x-pt1_start.x;
      float Y2_Y1 = pt1_end.y-pt1_start.y;
      float X4_X3 = pt2_end.x-pt2_start.x;
      float Y4_Y3 = pt2_end.y-pt2_start.y;
      float X1_X3 = pt1_start.x-pt2_start.x;
      float Y1_Y3 = pt1_start.y-pt2_start.y;
      float u_a = (X4_X3*Y1_Y3-Y4_Y3*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);
      float u_b = (X2_X1*Y1_Y3-Y2_Y1*X1_X3)/ (Y4_Y3*X2_X1-X4_X3*Y2_Y1);      
      float INT_X = pt1_start.x+X2_X1*u_a;
      float INT_Y = pt1_start.y+Y2_Y1*u_a;
      float INT_B = float((u_a >= 0) && (u_a <= 1) && (u_b >= 0) && (u_b <= 1));
      if (infinite_line)
	  INT_B=1;
      
      extcond_1 = u_a; extcond_2 = u_b;
      return cv::Point2f(INT_X*INT_B, INT_Y*INT_B);

}


// merge short edges into long. edges n*4  each edge should start from left to right! 
void merge_break_lines(const MatrixXd& all_lines, MatrixXd& merge_lines_out, double pre_merge_dist_thre,
		       double pre_merge_angle_thre_degree,double edge_length_threshold)
{
    bool can_force_merge = true;
    merge_lines_out = all_lines;
    int total_line_number = merge_lines_out.rows();  // line_number will become smaller and smaller, merge_lines_out doesn't change
    int counter=0;
    double pre_merge_angle_thre = pre_merge_angle_thre_degree/180.0*M_PI;
    while ((can_force_merge) && (counter<500)){
	    counter++;
	    can_force_merge=false;
	    MatrixXd line_vector = merge_lines_out.topRightCorner(total_line_number,2)-merge_lines_out.topLeftCorner(total_line_number,2);
	    VectorXd all_angles; atan2_vector(line_vector.col(1),line_vector.col(0),all_angles); // don't need normalize_to_pi, because my edges is from left to right, always [-90 90]
	    for (int seg1=0;seg1<total_line_number-1;seg1++) {
		for (int seg2=seg1+1;seg2<total_line_number;seg2++){
		      double diff = std::abs(all_angles(seg1)-all_angles(seg2));
		      double angle_diff = std::min(diff,M_PI-diff);
		      if (angle_diff<pre_merge_angle_thre){
			  double dist_1ed_to_2=(merge_lines_out.row(seg1).tail(2)-merge_lines_out.row(seg2).head(2)).norm();
			  double dist_2ed_to_1=(merge_lines_out.row(seg2).tail(2)-merge_lines_out.row(seg1).head(2)).norm();
			  
			  if ((dist_1ed_to_2<pre_merge_dist_thre) || (dist_2ed_to_1<pre_merge_dist_thre))
			  {
				Vector2d merge_start,merge_end;
				if (merge_lines_out(seg1,0)<merge_lines_out(seg2,0))
				    merge_start = merge_lines_out.row(seg1).head(2);
				else
				    merge_start = merge_lines_out.row(seg2).head(2);
				
				if (merge_lines_out(seg1,2)>merge_lines_out(seg2,2))
				    merge_end = merge_lines_out.row(seg1).tail(2);
				else
				    merge_end = merge_lines_out.row(seg2).tail(2);

				double merged_angle = std::atan2(merge_end(1)-merge_start(1),merge_end(0)-merge_start(0));
				double temp=std::abs(all_angles(seg1)-merged_angle);
				double merge_angle_diff = std::min( temp, M_PI-temp );
				if (merge_angle_diff<pre_merge_angle_thre)
				{
				    merge_lines_out.row(seg1).head(2) = merge_start;
				    merge_lines_out.row(seg1).tail(2) = merge_end;
				    fast_RemoveRow(merge_lines_out,seg2,total_line_number);  //also decrease  total_line_number
				    can_force_merge=true;
				    break;
				}
			  }
		      }
		}
		if (can_force_merge)
		      break;			
	    }
    }
//     std::cout<<"total_line_number after mege  "<<total_line_number<<std::endl;
    if (edge_length_threshold>0)
    {
	MatrixXd line_vectors = merge_lines_out.topRightCorner(total_line_number,2)-merge_lines_out.topLeftCorner(total_line_number,2);
	VectorXd line_lengths=line_vectors.rowwise().norm();
	int long_line_number=0;
	MatrixXd long_merge_lines(total_line_number,4);
	for (int i=0;i<total_line_number;i++){
	  if (line_lengths(i)>edge_length_threshold)
	  {
	    long_merge_lines.row(long_line_number)=merge_lines_out.row(i);
	    long_line_number++;
	  }
	}
	merge_lines_out = long_merge_lines.topRows(long_line_number);
    }
    else
	merge_lines_out.conservativeResize(total_line_number,NoChange);
}

// VPs 3*2   edge_mid_pts: n*2   vp_support_angle_thres 1*2
// output: 3*2  each row is a VP's two boundary supported edges' angle.  if not found, nan for that entry
Eigen::MatrixXd VP_support_edge_infos(Eigen::MatrixXd& VPs, Eigen::MatrixXd& edge_mid_pts, Eigen::VectorXd& edge_angles,
				      Eigen::Vector2d vp_support_angle_thres)
{
    MatrixXd all_vp_bound_edge_angles=MatrixXd::Ones(3,2)*nan(""); // initialize as nan  use isnan to check
    if (edge_mid_pts.rows()>0)
    {
	for (int vp_id=0;vp_id<VPs.rows();vp_id++)
	{
	    double vp_angle_thre;
	    if (vp_id!=2)
		vp_angle_thre = vp_support_angle_thres(0)/180.0*M_PI;
	    else
		vp_angle_thre = vp_support_angle_thres(1)/180.0*M_PI;
    
	    std::vector<int> vp_inlier_edge_id;
	    VectorXd vp_edge_midpt_angle_raw_inlier(edge_angles.rows());
	    for (int edge_id=0;edge_id<edge_angles.rows();edge_id++)
	    {
		double vp1_edge_midpt_angle_raw_i = atan2(edge_mid_pts(edge_id,1)-VPs(vp_id,1),edge_mid_pts(edge_id,0)-VPs(vp_id,0));
		double vp1_edge_midpt_angle_norm_i = normalize_to_pi<double>(vp1_edge_midpt_angle_raw_i);
		double angle_diff_i = std::abs(edge_angles(edge_id) - vp1_edge_midpt_angle_norm_i);
		angle_diff_i = std::min(angle_diff_i,M_PI-angle_diff_i);
		if (angle_diff_i<vp_angle_thre)
		{
		    vp_edge_midpt_angle_raw_inlier(vp_inlier_edge_id.size())=vp1_edge_midpt_angle_raw_i;
		    vp_inlier_edge_id.push_back(edge_id);
		}
	    }
	    if (vp_inlier_edge_id.size()>0) // if found inlier edges
	    {
		VectorXd vp1_edge_midpt_angle_raw_inlier_shift; smooth_jump_angles(vp_edge_midpt_angle_raw_inlier.head(vp_inlier_edge_id.size()),
										   vp1_edge_midpt_angle_raw_inlier_shift);
		int vp1_low_edge_id;	vp1_edge_midpt_angle_raw_inlier_shift.maxCoeff(&vp1_low_edge_id);
		int vp1_top_edge_id;	vp1_edge_midpt_angle_raw_inlier_shift.minCoeff(&vp1_top_edge_id);
		if (vp_id>0)
		  std::swap(vp1_low_edge_id,vp1_top_edge_id);  // match matlab code
		all_vp_bound_edge_angles(vp_id,0) = edge_angles(vp_inlier_edge_id[vp1_low_edge_id]);   // it will be 0*1 matrix if not found inlier edges.
		all_vp_bound_edge_angles(vp_id,1) = edge_angles(vp_inlier_edge_id[vp1_top_edge_id]);
	    }
	}
    }    
    return all_vp_bound_edge_angles;
}



double box_edge_sum_dists(const cv::Mat& dist_map, const MatrixXd& box_corners_2d, const MatrixXi& edge_pt_ids, bool reweight_edge_distance)
{
// give some edges, sample some points on line then sum up distance from dist_map
// input: visible_edge_pt_ids is n*2  each row stores an edge's two end point's index from box_corners_2d
// if weight_configs: for configuration 1, there are more visible edges compared to configuration2, so we need to re-weight
// [1 2;2 3;3 4;4 1;2 6;3 5;4 8;5 8;5 6];  reweight vertical edge id 5-7 by 2/3, horizontal edge id 8-9 by 1/2
    float sum_dist=0;
    for (int edge_id=0;edge_id<edge_pt_ids.rows();edge_id++)
    {
	Vector2d corner_tmp1=box_corners_2d.col(edge_pt_ids(edge_id,0));
	Vector2d corner_tmp2=box_corners_2d.col(edge_pt_ids(edge_id,1));
	for (double sample_ind=0;sample_ind<11;sample_ind++)
	{
            Vector2d sample_pt = sample_ind/10.0*corner_tmp1+(1-sample_ind/10.0)*corner_tmp2;
            float dist1 = dist_map.at<float>(int(sample_pt(1)),int(sample_pt(0)));//make sure dist_map is float type
            if (reweight_edge_distance)
	    {
                if ((4<=edge_id) && (edge_id<=5))
                    dist1=dist1*3.0/2.0;
                if (6==edge_id)
                    dist1=dist1*2.0;
	    }
            sum_dist=sum_dist+dist1;
	}
    }
    return double(sum_dist);
}


double box_edge_alignment_angle_error(const MatrixXd& all_vp_bound_edge_angles,const MatrixXi& vps_box_edge_pt_ids, const MatrixXd& box_corners_2d)
{
// compute the difference of box edge angle with angle of actually VP aligned image edges. for evaluating the box
// all_vp_bound_edge_angles: VP aligned actual image angles. 3*2  if not found, nan.      box_corners_2d: 2*8
// vps_box_edge_pt_ids: % six edges. each row represents two edges [e1_1 e1_2   e2_1 e2_2;...] of one VP
    double total_angle_diff = 0;
    double not_found_penalty = 30.0/180.0*M_PI*2;    // if not found any VP supported lines, give each box edge a constant cost (45 or 30 ? degree)
    for (int vp_id=0;vp_id<vps_box_edge_pt_ids.rows();vp_id++)
    {
        Vector2d vp_bound_angles=all_vp_bound_edge_angles.row(vp_id);
	std::vector<double> vp_bound_angles_valid;
	for (int i=0;i<2;i++)
	    if (!std::isnan(vp_bound_angles(i)))
	       vp_bound_angles_valid.push_back(vp_bound_angles(i));
	if (vp_bound_angles_valid.size()>0) //  exist valid edges
	{
	    for (int ee_id=0;ee_id<2;ee_id++) // find cloeset from two boundary edges. we could also do left-left right-right compare. but pay close attention different vp locations  
	    {
		Vector2d two_box_corners_1 = box_corners_2d.col( vps_box_edge_pt_ids(vp_id,2*ee_id) ); // [ x1;y1 ]
		Vector2d two_box_corners_2 = box_corners_2d.col( vps_box_edge_pt_ids(vp_id,2*ee_id+1) ); // [ x2;y2 ]
		
		double box_edge_angle = normalize_to_pi( atan2(two_box_corners_2(1)-two_box_corners_1(1), two_box_corners_2(0)-two_box_corners_1(0)));  // [-pi/2 -pi/2]
		double angle_diff_temp=100;
		for (int i=0;i<vp_bound_angles_valid.size();i++)
		{
		    double temp = std::abs(box_edge_angle-vp_bound_angles_valid[i]);
		    temp = std::min( temp, M_PI-temp );
		    if (temp<angle_diff_temp)
			angle_diff_temp = temp;
		}
                total_angle_diff=total_angle_diff+angle_diff_temp;
	    }
	}
	else
            total_angle_diff=total_angle_diff+not_found_penalty;
    }
    return total_angle_diff;
}

// weighted sum different score
void fuse_normalize_scores_v2(const VectorXd& dist_error, const VectorXd& angle_error, VectorXd& combined_scores, std::vector<int>& final_keep_inds, 
			      double weight_vp_angle, bool whether_normalize)
{
    int raw_data_size = dist_error.rows();
    if (raw_data_size>4)
    {
	int breaking_num = round(float(raw_data_size)/3.0*2.0);
	std::vector<int> dist_sorted_inds(raw_data_size); std::iota(dist_sorted_inds.begin(), dist_sorted_inds.end(), 0);
	std::vector<int> angle_sorted_inds = dist_sorted_inds;
	
	sort_indexes(dist_error, dist_sorted_inds, breaking_num);
	sort_indexes(angle_error, angle_sorted_inds, breaking_num);

	std::vector<int> dist_keep_inds = std::vector<int>(dist_sorted_inds.begin(),dist_sorted_inds.begin()+breaking_num-1);  // keep best 2/3
	
	if ( angle_error(angle_sorted_inds[breaking_num-1])>angle_error(angle_sorted_inds[breaking_num-2]) )
	{
	    std::vector<int> angle_keep_inds = std::vector<int>(angle_sorted_inds.begin(),angle_sorted_inds.begin()+breaking_num-1);  // keep best 2/3
	    
	    std::sort(dist_keep_inds.begin(),dist_keep_inds.end());
	    std::sort(angle_keep_inds.begin(),angle_keep_inds.end());
	    std::set_intersection(dist_keep_inds.begin(), dist_keep_inds.end(),
				  angle_keep_inds.begin(), angle_keep_inds.end(),
				  std::back_inserter(final_keep_inds));
	}
	else  //don't need to consider angle.   my angle error has maximum. may already saturate at breaking pt.
	{
	    final_keep_inds = dist_keep_inds;
	}
    }
    else
    {
	final_keep_inds.resize(raw_data_size);   //don't change anything.
	std::iota(final_keep_inds.begin(), final_keep_inds.end(), 0);
    }
  
    int new_data_size = final_keep_inds.size();
    // find max/min of kept errors.
    double min_dist_error=1e6; double max_dist_error=-1;double min_angle_error=1e6;double max_angle_error=-1;
    VectorXd dist_kept(new_data_size);  VectorXd angle_kept(new_data_size);
    for (int i=0;i<new_data_size;i++)
    {
	double temp_dist = dist_error(final_keep_inds[i]);	double temp_angle = angle_error(final_keep_inds[i]);
	min_dist_error = std::min(min_dist_error,temp_dist);	max_dist_error = std::max(max_dist_error,temp_dist);
	min_angle_error = std::min(min_angle_error,temp_angle); max_angle_error = std::max(max_angle_error,temp_angle);
	dist_kept(i) = temp_dist;  angle_kept(i) = temp_angle;
    }
    
    if (whether_normalize && (new_data_size>1))
    {
	combined_scores  = (dist_kept.array()-min_dist_error)/(max_dist_error-min_dist_error);
	if ((max_angle_error-min_angle_error)>0)
	{
	    angle_kept = (angle_kept.array()-min_angle_error)/(max_angle_error-min_angle_error);
	    combined_scores = (combined_scores + weight_vp_angle*angle_kept)/(1+weight_vp_angle);
	}
	else
	    combined_scores = (combined_scores + weight_vp_angle*angle_kept)/(1+weight_vp_angle);
    }
    else
	combined_scores = (dist_kept+weight_vp_angle*angle_kept)/(1+weight_vp_angle);    
}


//rays is 3*n, each column is a ray staring from origin  plane is (4，1） parameters, compute intersection  output is 3*n 
void ray_plane_interact(const MatrixXd &rays,const Eigen::Vector4d &plane,MatrixXd &intersections)
{  
    VectorXd frac=-plane[3]/(plane.head(3).transpose()*rays).array();   //n*1 
    intersections= frac.transpose().replicate<3,1>().array() * rays.array();
}

void plane_hits_3d(const Matrix4d& transToWolrd, const Matrix3d& invK, const Vector4d& plane_sensor,MatrixXd pixels, Matrix3Xd& pts_3d_world)
// compute ray intersection with plane in 3D.
// transToworld: 4*4 camera pose.   invK: inverse of calibration.   plane: 1*4  plane equation in sensor frame. 
// pixels  2*n; each column is a pt [x;y] x is horizontal,y is vertical   outputs: pts3d 3*n in world frame
{
    pixels.conservativeResize(3,NoChange);
    pixels.row(2)=VectorXd::Ones(pixels.cols());
    MatrixXd pts_ray=invK*pixels;    //each column is a 3D world coordinate  3*n    	
    MatrixXd pts_3d_sensor;  ray_plane_interact(pts_ray,plane_sensor,pts_3d_sensor);
    pts_3d_world = homo_to_real_coord<double>(transToWolrd*real_to_homo_coord<double>(pts_3d_sensor)); //
}

Vector4d get_wall_plane_equation(const Vector3d& gnd_seg_pt1, const Vector3d& gnd_seg_pt2)
// 1*6 a line segment in 3D. [x1 y1 z1  x2 y2 z2]  z1=z2=0  or  two 1*3
{

    Vector3d partwall_normal_world = (gnd_seg_pt1-gnd_seg_pt2).cross(Vector3d(0,0,1)); // [0,0,1] is world ground plane
    partwall_normal_world.array() /= partwall_normal_world.norm();
    double dist=-partwall_normal_world.transpose()*gnd_seg_pt1;
    Vector4d plane_equation;plane_equation<<partwall_normal_world,
					    dist;        // wall plane in world frame
    if (dist<0)
        plane_equation = -plane_equation;   // make all the normal pointing inside the room. neamly, pointing to the camera
    return plane_equation;
}

void getVanishingPoints(const Matrix3d& KinvR, double yaw_esti, Vector2d& vp_1, Vector2d& vp_2, Vector2d& vp_3)
{
    vp_1 = homo_to_real_coord_vec<double>(KinvR*Vector3d( cos(yaw_esti),sin(yaw_esti),0));  // for object x axis
    vp_2 = homo_to_real_coord_vec<double>(KinvR*Vector3d(-sin(yaw_esti),cos(yaw_esti),0));  // for object y axis
    vp_3 = homo_to_real_coord_vec<double>(KinvR*Vector3d(0,0,1));  // for object z axis
}


// box_corners_2d_float is 2*8    change to my object struct from 2D box corners.
void change_2d_corner_to_3d_object(const MatrixXd& box_corners_2d_float,const Vector3d& configs, const Vector4d& ground_plane_sensor, 
				   const Matrix4d& transToWolrd, const Matrix3d& invK, Eigen::Matrix<double, 3, 4>& projectionMatrix,
				   cuboid& sample_obj)
{
    Matrix3Xd obj_gnd_pt_world_3d; plane_hits_3d(transToWolrd, invK, ground_plane_sensor, box_corners_2d_float.rightCols(4), obj_gnd_pt_world_3d);//% 3*n each column is a 3D point  floating point
    
    double length_half = (obj_gnd_pt_world_3d.col(0)-obj_gnd_pt_world_3d.col(3)).norm()/2;  // along object x direction   corner 5-8
    double width_half = (obj_gnd_pt_world_3d.col(0)-obj_gnd_pt_world_3d.col(1)).norm()/2;  // along object y direction   corner 5-6
    
    Vector4d partwall_plane_world = get_wall_plane_equation(obj_gnd_pt_world_3d.col(0),obj_gnd_pt_world_3d.col(1));//% to compute height, need to unproject-hit-planes formed by 5-6 corner
    Vector4d partwall_plane_sensor = transToWolrd.transpose()*partwall_plane_world;  // wall plane in sensor frame
    
    Matrix3Xd obj_top_pt_world_3d; plane_hits_3d(transToWolrd,invK,partwall_plane_sensor,box_corners_2d_float.col(1),obj_top_pt_world_3d);  // should match obj_gnd_pt_world_3d  % compute corner 2
    double height_half = obj_top_pt_world_3d(2,0)/2;
    
    double mean_obj_x = obj_gnd_pt_world_3d.row(0).mean(); double mean_obj_y = obj_gnd_pt_world_3d.row(1).mean();
    
    double vp_1_position = configs(1); double yaw_esti = configs(2);  
    sample_obj.pos = Vector3d(mean_obj_x,mean_obj_y,height_half);  sample_obj.rotY = yaw_esti;
    sample_obj.scale = Vector3d(length_half,width_half,height_half);
    sample_obj.box_config_type = configs.head<2>();
    VectorXd cuboid_to_raw_boxstructIds(8);
    if (vp_1_position==1)  // vp1 on left, for all configurations
        cuboid_to_raw_boxstructIds<<6, 5, 8, 7, 2, 3, 4, 1;
    if (vp_1_position==2)  // vp1 on right, for all configurations
        cuboid_to_raw_boxstructIds<<5, 6, 7, 8, 3, 2, 1, 4;

    Matrix2Xi box_corners_2d_int = box_corners_2d_float.cast<int>();
    sample_obj.box_corners_2d.resize(2,8);
    for (int i=0;i<8;i++)
	sample_obj.box_corners_2d.col(i)=box_corners_2d_int.col( cuboid_to_raw_boxstructIds(i)-1 ); // minius one to match index
    
    sample_obj.box_corners_3d_world = compute3D_BoxCorner(sample_obj);
}


float bboxOverlapratio(const cv::Rect& rect1, const cv::Rect& rect2)
{
    int overlap_area = (rect1&rect2).area();
    return (float)overlap_area/((float)(rect1.area()+rect2.area()-overlap_area));
}


int pointBoundaryDist(const cv::Rect& rect, const cv::Point2f& kp )
{
    int mid_x = rect.x + rect.width/2;
    int mid_y = rect.y + rect.height/2;
    int min_x_bound_dist = 0;int min_y_bound_dist = 0;
    if (kp.x<mid_x)
	min_x_bound_dist = abs(kp.x-rect.x);
    else
	min_x_bound_dist = abs(kp.x-rect.x-rect.width);
    if (kp.y<mid_y)
	min_y_bound_dist = abs(kp.y-rect.y);
    else
	min_y_bound_dist = abs(kp.y-rect.y-rect.height);
    return std::min(min_x_bound_dist,min_y_bound_dist);
}