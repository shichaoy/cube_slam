#include <iostream>
#include <fstream>
#include <string> 
#include <sstream>
#include <ctime>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Geometry> 

#include <ros/ros.h> 
#include <ros/package.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"


#include <object_slam/Object_landmark.h>
#include <object_slam/Frame.h>
#include <object_slam/g2o_Object.h>

#include "detect_3d_cuboid/matrix_utils.h"
#include "detect_3d_cuboid/detect_3d_cuboid.h"

#include "line_lbd/line_lbd_allclass.h"


using namespace std;
using namespace Eigen;


typedef pcl::PointCloud<pcl::PointXYZRGB> CloudXYZRGB;


// global variable
std::string base_folder;
bool online_detect_mode;
bool save_results_to_txt;
cv::Mat_<float> matx_to3d_, maty_to3d_;

// for converting depth image to point cloud.
void set_up_calibration(const Eigen::Matrix3f& calibration_mat,const int im_height,const int im_width)
{  
    matx_to3d_.create(im_height, im_width);
    maty_to3d_.create(im_height, im_width);
    float center_x=calibration_mat(0,2);  //cx
    float center_y=calibration_mat(1,2);  //cy
    float fx_inv=1.0/calibration_mat(0,0);  // 1/fx
    float fy_inv=1.0/calibration_mat(1,1);  // 1/fy
    for (int v = 0; v < im_height; v++) {
	for (int u = 0; u < im_width; u++) {
	  matx_to3d_(v,u) = (u - center_x) * fx_inv;
	  maty_to3d_(v,u) = (v - center_y) * fy_inv;
	}
    }
}

// depth img is already in m unit.
void depth_to_cloud(const cv::Mat& rgb_img, const cv::Mat& depth_img,const Eigen::Matrix4f transToWorld, CloudXYZRGB::Ptr& point_cloud,bool downsample=false)
{
    pcl::PointXYZRGB pt;
    pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> vox_grid_;
    float close_depth_thre = 0.1;
    float far_depth_thre = 3.0;
      far_depth_thre = 3;
    int im_width = rgb_img.cols; int im_height= rgb_img.rows;
    for (int32_t i=0; i<im_width*im_height; i++) {      // row by row
	int ux=i % im_width; int uy=i / im_width;       
	float pix_depth= depth_img.at<float>(uy,ux);
	if (pix_depth>close_depth_thre && pix_depth<far_depth_thre){
	      pt.z=pix_depth; pt.x=matx_to3d_(uy,ux)*pix_depth; pt.y=maty_to3d_(uy,ux)*pix_depth;
	      Eigen::VectorXf global_pt=homo_to_real_coord_vec<float>(transToWorld*Eigen::Vector4f(pt.x,pt.y,pt.z,1));  // change to global position
	      pt.x=global_pt(0); pt.y=global_pt(1); pt.z=global_pt(2);
	      pt.r = rgb_img.at<cv::Vec3b>(uy,ux)[2]; pt.g = rgb_img.at<cv::Vec3b>(uy,ux)[1]; pt.b = rgb_img.at<cv::Vec3b>(uy,ux)[0];
	      point_cloud->points.push_back(pt);
	}
    }    
    if (downsample)
    {
	vox_grid_.setLeafSize(0.02,0.02,0.02);
	vox_grid_.setDownsampleAllData(true);
	vox_grid_.setInputCloud(point_cloud);
	vox_grid_.filter(*point_cloud);
    }
}

// one cuboid need front and back markers...
void cuboid_corner_to_marker(const Matrix38d& cube_corners,visualization_msgs::Marker& marker, int bodyOrfront)
{
    Eigen::VectorXd edge_pt_ids;
    if (bodyOrfront==0) { // body edges
	edge_pt_ids.resize(16); edge_pt_ids<<1,2,3,4,1,5,6,7,8,5,6,2,3,7,8,4;edge_pt_ids.array()-=1;
    }else { // front face edges
	edge_pt_ids.resize(5); edge_pt_ids<<1,2,6,5,1;edge_pt_ids.array()-=1;
    }
    marker.points.resize(edge_pt_ids.rows());
    for (int pt_id=0; pt_id<edge_pt_ids.rows(); pt_id++)
    {
	marker.points[pt_id].x = cube_corners(0, edge_pt_ids(pt_id));
	marker.points[pt_id].y = cube_corners(1, edge_pt_ids(pt_id));
	marker.points[pt_id].z = cube_corners(2, edge_pt_ids(pt_id));
    }
}

// one cuboid need front and back markers...  rgbcolor is 0-1 based
visualization_msgs::MarkerArray cuboids_to_marker(object_landmark* obj_landmark, Vector3d rgbcolor) 
{
    visualization_msgs::MarkerArray plane_markers;  visualization_msgs::Marker marker;
    if (obj_landmark==nullptr)
	return plane_markers;

    marker.header.frame_id="/world";  marker.header.stamp=ros::Time::now();
    marker.id = 0; //0
    marker.type = visualization_msgs::Marker::LINE_STRIP;   marker.action = visualization_msgs::Marker::ADD;
    marker.color.r = rgbcolor(0); marker.color.g = rgbcolor(1); marker.color.b = rgbcolor(2); marker.color.a = 1.0;
    marker.scale.x = 0.02;

    g2o::cuboid cube_opti = obj_landmark->cube_vertex->estimate();
    Eigen::MatrixXd cube_corners = cube_opti.compute3D_BoxCorner();
    
    for (int ii=0;ii<2;ii++) // each cuboid needs two markers!!! one for all edges, one for front facing edge, could with different color.
    {
	marker.id++;
	cuboid_corner_to_marker(cube_corners,marker, ii);
	plane_markers.markers.push_back(marker);
    }
    return plane_markers;
}

geometry_msgs::Pose posenode_to_geomsgs(const g2o::SE3Quat &pose_Twc)
{
    geometry_msgs::Pose pose_msg;    
    Eigen::Vector3d pose_trans = pose_Twc.translation();	
    pose_msg.position.x=pose_trans(0);
    pose_msg.position.y=pose_trans(1);
    pose_msg.position.z=pose_trans(2);
    Eigen::Quaterniond pose_quat = pose_Twc.rotation();
    pose_msg.orientation.x = pose_quat.x();  
    pose_msg.orientation.y = pose_quat.y();
    pose_msg.orientation.z = pose_quat.z();
    pose_msg.orientation.w = pose_quat.w();
    return pose_msg;
}

nav_msgs::Odometry posenode_to_odommsgs(const g2o::SE3Quat &pose_Twc,const std_msgs::Header &img_header)
{
    nav_msgs::Odometry odom_msg;
    odom_msg.pose.pose=posenode_to_geomsgs(pose_Twc);    
    odom_msg.header=img_header;
    return odom_msg;
}


// publish every frame's raw and optimized results
void publish_all_poses(std::vector<tracking_frame*> all_frames,std::vector<object_landmark*> cube_landmarks_history,
		       std::vector<object_landmark*> all_frame_rawcubes, Eigen::MatrixXd& truth_frame_poses)
{
    ros::NodeHandle n;
    ros::Publisher pub_slam_all_poses = n.advertise<geometry_msgs::PoseArray>("/slam_pose_array", 10);
    ros::Publisher pub_slam_odompose = n.advertise<nav_msgs::Odometry>("/slam_odom_pose", 10);
    ros::Publisher pub_truth_all_poses = n.advertise<geometry_msgs::PoseArray>("/truth_pose_array", 10);
    ros::Publisher pub_truth_odompose = n.advertise<nav_msgs::Odometry>("/truth_odom_pose", 10);
    ros::Publisher pub_slam_path = n.advertise<nav_msgs::Path>( "/slam_pose_paths", 10 );
    ros::Publisher pub_truth_path = n.advertise<nav_msgs::Path>( "/truth_pose_paths", 10 );
    ros::Publisher pub_final_opti_cube = n.advertise<visualization_msgs::MarkerArray>("/cubes_opti", 10); //final unique cube pose
    ros::Publisher pub_history_opti_cube = n.advertise<visualization_msgs::MarkerArray>("/cubes_opti_hist", 10); // landmark cube pose after each optimization
    ros::Publisher pub_frame_raw_cube = n.advertise<visualization_msgs::MarkerArray>("/cubes_raw_frame", 10);
    ros::Publisher pub_2d_cuboid_project = n.advertise<sensor_msgs::Image>("/cuboid_project_img", 10);
    ros::Publisher raw_cloud_pub = n.advertise<CloudXYZRGB> ("/raw_point_cloud", 50);
    
    int total_frame_number = all_frames.size();
    
    // prepare all paths messages.
    geometry_msgs::PoseArray all_pred_pose_array;    std::vector<nav_msgs::Odometry> all_pred_pose_odoms;
    geometry_msgs::PoseArray all_truth_pose_array;    std::vector<nav_msgs::Odometry> all_truth_pose_odoms;
    nav_msgs::Path path_truths,path_preds;
    std_msgs::Header pose_header;    pose_header.frame_id = "/world";    pose_header.stamp = ros::Time::now();
    path_preds.header = pose_header;    path_truths.header = pose_header;    
    for (int i = 0; i < total_frame_number; i++)
    {
	all_pred_pose_array.poses.push_back(posenode_to_geomsgs(all_frames[i]->cam_pose_Twc));
	all_pred_pose_odoms.push_back(posenode_to_odommsgs(all_frames[i]->cam_pose_Twc,pose_header) );
	
	geometry_msgs::PoseStamped postamp;
	postamp.pose = posenode_to_geomsgs(all_frames[i]->cam_pose_Twc);
	postamp.header = pose_header;
	path_preds.poses.push_back(postamp);
    }
    if (truth_frame_poses.rows()>0)
    {
	for (int i=0; i < total_frame_number;i++)
	{
	    geometry_msgs::Pose pose_msg;
	    pose_msg.position.x=truth_frame_poses(i,1);    pose_msg.position.y=truth_frame_poses(i,2);    pose_msg.position.z=truth_frame_poses(i,3);
	    pose_msg.orientation.x = truth_frame_poses(i,4);	pose_msg.orientation.y = truth_frame_poses(i,5);
	    pose_msg.orientation.z = truth_frame_poses(i,6);	pose_msg.orientation.w = truth_frame_poses(i,7);
	    all_truth_pose_array.poses.push_back(pose_msg);
	    nav_msgs::Odometry odom_msg;odom_msg.pose.pose=pose_msg;
	    odom_msg.header = pose_header;
	    all_truth_pose_odoms.push_back(odom_msg);
	    
	    geometry_msgs::PoseStamped postamp;
	    postamp.pose = pose_msg;
	    postamp.header = pose_header;
	    path_truths.poses.push_back(postamp);
	}
    }
    all_pred_pose_array.header.stamp=ros::Time::now();    all_pred_pose_array.header.frame_id="/world";
    all_truth_pose_array.header.stamp=ros::Time::now();    all_truth_pose_array.header.frame_id="/world";
    
        
    if (save_results_to_txt)  // record cam pose and object pose
    {
	ofstream resultsFile;
	string resultsPath = base_folder + "output_cam_poses.txt";
	cout<<"resultsPath  "<<resultsPath<<endl;
	resultsFile.open(resultsPath.c_str());
	resultsFile << "# timestamp tx ty tz qx qy qz qw"<<"\n";
	for (int i=0;i<total_frame_number;i++)
	{
	    double time_string=truth_frame_poses(i,0);
	    ros::Time time_img(time_string);
	    resultsFile << time_img<<"  ";	    
	    resultsFile << all_frames[i]->cam_pose_Twc.toVector().transpose()<<"\n";
	}
	resultsFile.close();
		
	ofstream objresultsFile;
	string objresultsPath = base_folder + "output_obj_poses.txt";
	objresultsFile.open(objresultsPath.c_str());
	for (size_t j=0;j<cube_landmarks_history.size();j++)
	{
	    g2o::cuboid cube_opti = cube_landmarks_history[j]->cube_vertex->estimate();
	    // transform it to local ground plane.... suitable for matlab processing.
	    objresultsFile << cube_opti.toMinimalVector().transpose()<<" "<<"\n";
	}
	objresultsFile.close();
    }    
    
    // sensor parameter for TUM cabinet data!
    Eigen::Matrix3f calib;
    float depth_map_scaling = 5000;
    calib<<535.4,  0, 320.1,
	    0,  539.2, 247.6,
	    0,      0,     1;
    set_up_calibration(calib,480,640);
    
    visualization_msgs::MarkerArray finalcube_markers = cuboids_to_marker(cube_landmarks_history.back(),Vector3d(0,1,0));
    
    bool show_truth_cloud = true;  // show point cloud using camera pose. for visualization purpose
        
    pcl::PCLPointCloud2 pcd_cloud2;
    
    ros::Rate loop_rate(5);  //5
    int frame_number = -1;
    while ( n.ok() )
    {
	frame_number++;
	
	if (0) // directly show final results
	{
	    pub_slam_all_poses.publish(all_pred_pose_array);	pub_truth_all_poses.publish(all_truth_pose_array);
	    pub_slam_path.publish(path_preds);	pub_truth_path.publish(path_truths);
	}
	pub_final_opti_cube.publish(finalcube_markers);
	
	if (frame_number<total_frame_number)
	{
	    // publish cuboid landmarks, after each frame's g2o optimization
	    if (cube_landmarks_history[frame_number]!=nullptr)
		pub_history_opti_cube.publish(cuboids_to_marker(cube_landmarks_history[frame_number],Vector3d(1,0,0)));
	    
	    // publish raw detected cube in each frame, before optimization
	    if (all_frame_rawcubes.size()>0 && all_frame_rawcubes[frame_number]!=nullptr)
		pub_frame_raw_cube.publish(cuboids_to_marker(all_frame_rawcubes[frame_number],Vector3d(0,0,1)));

	    // publish camera pose estimation of this frame
	    pub_slam_odompose.publish(all_pred_pose_odoms[frame_number]);
	    pub_truth_odompose.publish(all_truth_pose_odoms[frame_number]);
	    
// 	    std::cout<<"Frame position x/y:   "<<frame_number<<"        "<<all_pred_pose_odoms[frame_number].pose.pose.position.x<<"  "<<
// 			  all_pred_pose_odoms[frame_number].pose.pose.position.y <<std::endl;

	    char frame_index_c[256];	sprintf(frame_index_c,"%04d",frame_number);  // format into 4 digit
	    
	    cv::Mat cuboid_2d_proj_img = all_frames[frame_number]->cuboids_2d_img;
	    
	    std::string raw_rgb_img_name = base_folder+"raw_imgs/" + std::string(frame_index_c) + "_rgb_raw.jpg";
	    cv::Mat raw_rgb_img = cv::imread(raw_rgb_img_name, 1);

	    if (show_truth_cloud && (truth_frame_poses.rows()>0))
		if (frame_number%2==0) // show point cloud every N frames
		{
		    std::string raw_depth_img_name = base_folder+"depth_imgs/" + std::string(frame_index_c) + "_depth_raw.png";
		    cv::Mat raw_depth_img = cv::imread(raw_depth_img_name, CV_LOAD_IMAGE_ANYDEPTH);
		    raw_depth_img.convertTo(raw_depth_img, CV_32FC1, 1.0/depth_map_scaling,0);

		    CloudXYZRGB::Ptr point_cloud(new CloudXYZRGB());
		    Eigen::Matrix4f truth_pose_matrix=g2o::SE3Quat(truth_frame_poses.row(frame_number).segment<7>(1)).to_homogeneous_matrix().cast<float>();

		    depth_to_cloud(raw_rgb_img, raw_depth_img, truth_pose_matrix, point_cloud, true); // need to downsample cloud, otherwise too many
		    ros::Time curr_time=ros::Time::now();

		    point_cloud->header.frame_id = "/world";
		    point_cloud->header.stamp = (curr_time.toNSec() / 1000ull);
		    raw_cloud_pub.publish(point_cloud);
		}
	    
	    cv_bridge::CvImage out_image;
	    out_image.header.stamp=ros::Time::now();
	    out_image.image=cuboid_2d_proj_img;
	    out_image.encoding=sensor_msgs::image_encodings::TYPE_8UC3;
	    pub_2d_cuboid_project.publish(out_image.toImageMsg());
	}
	
	if (frame_number==int(all_pred_pose_odoms.size()))
	{
	    cout<<"Finish all visulialization!"<<endl;
	}
	
	ros::spinOnce();
	loop_rate.sleep();
    }  
}


//NOTE offline_pred_objects and init_frame_poses are not used in online_detect_mode! truth cam pose of first frame is used.
void incremental_build_graph(Eigen::MatrixXd& offline_pred_frame_objects, Eigen::MatrixXd& init_frame_poses, Eigen::MatrixXd& truth_frame_poses)
{  
    Eigen::Matrix3d calib; 
    calib<<535.4,  0,  320.1,   // for TUM cabinet data.
	    0,  539.2, 247.6,
	    0,      0,     1;    
    
    int total_frame_number = truth_frame_poses.rows();

    // detect all frames' cuboids.
    detect_3d_cuboid detect_cuboid_obj;
    detect_cuboid_obj.whether_plot_detail_images = false;
    detect_cuboid_obj.whether_plot_final_images = false;
    detect_cuboid_obj.print_details = false;  // false  true
    detect_cuboid_obj.set_calibration(calib);
    detect_cuboid_obj.whether_sample_bbox_height = false;
    detect_cuboid_obj.nominal_skew_ratio = 2;
    detect_cuboid_obj.whether_save_final_images = true;
    

    line_lbd_detect line_lbd_obj;
    line_lbd_obj.use_LSD = true;
    line_lbd_obj.line_length_thres = 15;  // remove short edges
    
    
    // graph optimization.
    //NOTE in this example, there is only one object!!! perfect association
    g2o::SparseOptimizer graph;
    g2o::BlockSolverX::LinearSolverType* linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    graph.setAlgorithm(solver);    graph.setVerbose(false);

    
    // only first truth pose is used. to directly visually compare with truth pose. also provide good roll/pitch
    g2o::SE3Quat fixed_init_cam_pose_Twc(truth_frame_poses.row(0).tail<7>());
    
    // save optimization results of each frame
    std::vector<object_landmark*> cube_pose_opti_history(total_frame_number, nullptr);  //landmark pose after each frame's optimization
    std::vector<object_landmark*> cube_pose_raw_detected_history(total_frame_number, nullptr); //raw detected cuboid frame each frame. before optimization

    int offline_cube_obs_row_id = 0;
    
    std::vector<tracking_frame*> all_frames(total_frame_number);    
    g2o::VertexCuboid* vCube;
    
    // process each frame online and incrementally
    for (int frame_index=0;frame_index<total_frame_number;frame_index++)
    {
	  g2o::SE3Quat curr_cam_pose_Twc;
	  g2o::SE3Quat odom_val; // from previous frame to current frame
	  
	  if (frame_index==0)
		curr_cam_pose_Twc = fixed_init_cam_pose_Twc;
	  else
	  {
		g2o::SE3Quat prev_pose_Tcw = all_frames[frame_index-1]->cam_pose_Tcw;
		if (frame_index>1)  // from third frame, use constant motion model to initialize camera.
		{
		    g2o::SE3Quat prev_prev_pose_Tcw = all_frames[frame_index-2]->cam_pose_Tcw;
		    odom_val = prev_pose_Tcw*prev_prev_pose_Tcw.inverse();
		}
		curr_cam_pose_Twc = (odom_val*prev_pose_Tcw).inverse();
	  }
            
	  
	  tracking_frame* currframe = new tracking_frame();
	  currframe->frame_seq_id = frame_index;
	  all_frames[frame_index] = currframe;

	  
	  bool has_detected_cuboid = false;
	  g2o::cuboid cube_local_meas; double proposal_error;
	  char frame_index_c[256];	sprintf(frame_index_c,"%04d",frame_index);  // format into 4 digit
	  
	  // read or detect cuboid
	  if (online_detect_mode)
	  {
	      //start detect cuboid
	      cv::Mat raw_rgb_img = cv::imread(base_folder+"raw_imgs/"+frame_index_c+"_rgb_raw.jpg", 1);
	    
	      //edge detection
	      cv::Mat all_lines_mat;
	      line_lbd_obj.detect_filter_lines(raw_rgb_img, all_lines_mat);
	      Eigen::MatrixXd all_lines_raw(all_lines_mat.rows,4);
	      for (int rr=0;rr<all_lines_mat.rows;rr++)
		for (int cc=0;cc<4;cc++)
		  all_lines_raw(rr,cc) = all_lines_mat.at<float>(rr,cc);
	      
	      
	      //read cleaned yolo 2d object detection
	      Eigen::MatrixXd raw_2d_objs(10,5);  // 2d rect [x1 y1 width height], and prob
	      if (!read_all_number_txt(base_folder+"/filter_2d_obj_txts/"+frame_index_c+"_yolo2_0.15.txt", raw_2d_objs))
		  return;
	      raw_2d_objs.leftCols<2>().array() -=1;   // change matlab coordinate to c++, minus 1
	      
	      Matrix4d transToWolrd;
	      detect_cuboid_obj.whether_sample_cam_roll_pitch = (frame_index!=0); // first frame doesn't need to sample cam pose. could also sample. doesn't matter much
	      if (detect_cuboid_obj.whether_sample_cam_roll_pitch) //sample around first frame's pose
		  transToWolrd = fixed_init_cam_pose_Twc.to_homogeneous_matrix();
	      else
		  transToWolrd = curr_cam_pose_Twc.to_homogeneous_matrix();
	      
	      std::vector<ObjectSet> frames_cuboids; // each 2d bbox generates an ObjectSet, which is vector of sorted proposals
	      detect_cuboid_obj.detect_cuboid(raw_rgb_img,transToWolrd,raw_2d_objs,all_lines_raw, frames_cuboids);
	      currframe->cuboids_2d_img = detect_cuboid_obj.cuboids_2d_img;
	      
	      has_detected_cuboid = frames_cuboids.size()>0 && frames_cuboids[0].size()>0;
	      if (has_detected_cuboid)  // prepare object measurement
	      {
		  cuboid* detected_cube = frames_cuboids[0][0];  // NOTE this is a simple dataset, only one landmark

		  g2o::cuboid cube_ground_value; //cuboid in the local ground frame.
		  Vector9d cube_pose;cube_pose<<detected_cube->pos(0),detected_cube->pos(1),detected_cube->pos(2),
			  0,0,detected_cube->rotY,detected_cube->scale(0),detected_cube->scale(1),detected_cube->scale(2);  // xyz roll pitch yaw scale
		  cube_ground_value.fromMinimalVector(cube_pose);
		  cube_local_meas = cube_ground_value.transform_to(curr_cam_pose_Twc); // measurement is in local camera frame

		  if (detect_cuboid_obj.whether_sample_cam_roll_pitch)  //if camera roll/pitch is sampled, transform to the correct camera frame.
		  {
		      Vector3d new_camera_eulers =  detect_cuboid_obj.cam_pose_raw.euler_angle;
		      new_camera_eulers(0) += detected_cube->camera_roll_delta; new_camera_eulers(1) += detected_cube->camera_pitch_delta;
		      Matrix3d rotation_new = euler_zyx_to_rot<double>(new_camera_eulers(0),new_camera_eulers(1),new_camera_eulers(2));
		      Vector3d trans = transToWolrd.col(3).head<3>();
		      g2o::SE3Quat curr_cam_pose_Twc_new(rotation_new,trans);
		      cube_local_meas = cube_ground_value.transform_to(curr_cam_pose_Twc_new);
		  }
		  proposal_error = detected_cube->normalized_error;
	      }
	  }
	  else
	  {
	      int cube_obs_frame_id = offline_pred_frame_objects(offline_cube_obs_row_id,0);
	      has_detected_cuboid = cube_obs_frame_id==frame_index;
	      if (has_detected_cuboid)  // prepare object measurement   not all frame has observation!!
	      {
		  VectorXd measure_data = offline_pred_frame_objects.row(offline_cube_obs_row_id);
		  g2o::cuboid cube_ground_value; 
		  Vector9d cube_pose;cube_pose<<measure_data(1),measure_data(2),measure_data(3),0,0,measure_data(4),
						measure_data(5),measure_data(6),measure_data(7);  // xyz roll pitch yaw scale
		  cube_ground_value.fromMinimalVector(cube_pose);
		  Eigen::VectorXd cam_pose_vec = init_frame_poses.row(frame_index);
		  g2o::SE3Quat cam_val_Twc(cam_pose_vec.segment<7>(1)); // time x y z qx qy qz qw
		  cube_local_meas = cube_ground_value.transform_to(cam_val_Twc); // measurement is in local camera frame
		  proposal_error = measure_data(8);
		  
		  // read offline saved 2d image
		  std::string detected_cube_2d_img_name = base_folder+"pred_3d_obj_overview/" + std::string(frame_index_c) + "_best_objects.jpg";
		  currframe->cuboids_2d_img = cv::imread(detected_cube_2d_img_name, 1);
		  
		  offline_cube_obs_row_id++; // switch to next row  NOTE at most one object one frame in this data
	      }
	  }
	  
	  if (has_detected_cuboid)
	  {
	      object_landmark* localcuboid = new object_landmark();
	      
	      localcuboid->cube_meas = cube_local_meas;
	      localcuboid->meas_quality = (1-proposal_error+0.5)/2;  // initial error 0-1, higher worse,  now change to [0.5,1] higher, better
	      currframe->observed_cuboids.push_back(localcuboid);
	  }
	  
	  
	  // set up g2o cube vertex. only one in this dataset
	  if (frame_index==0)
	  {
	      g2o::cuboid init_cuboid_global_pose = cube_local_meas.transform_from(curr_cam_pose_Twc);
	      vCube = new g2o::VertexCuboid();
	      vCube->setEstimate(init_cuboid_global_pose);
	      vCube->setId(0);
	      vCube->setFixed(false);
	      graph.addVertex(vCube);
	  }
	  
	  
	  // set up g2o camera vertex
	  g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
	  currframe->pose_vertex = vSE3;
	  vSE3->setId(frame_index+1);
	  graph.addVertex(vSE3);
	  
	  vSE3->setEstimate(curr_cam_pose_Twc.inverse()); //g2o vertex usually stores world to camera pose.
	  vSE3->setFixed(frame_index==0);
	  
	  // add g2o camera-object measurement edges, if there is
	  if (currframe->observed_cuboids.size()>0)
	  {
		object_landmark* cube_landmark_meas = all_frames[frame_index]->observed_cuboids[0];
		g2o::EdgeSE3Cuboid* e = new g2o::EdgeSE3Cuboid();
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vSE3 ));
		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>( vCube ));
		e->setMeasurement(cube_landmark_meas->cube_meas);
		e->setId(frame_index);
		Vector9d inv_sigma;inv_sigma<<1,1,1,1,1,1,1,1,1;
		inv_sigma = inv_sigma*2.0*cube_landmark_meas->meas_quality;
		Matrix9d info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
		e->setInformation(info);
		graph.addEdge(e);
	  }
	  
	  // camera vertex, add cam-cam odometry edges
	  if (frame_index>0)
	  {
		g2o::EdgeSE3Expmap* e = new g2o::EdgeSE3Expmap();
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>( all_frames[frame_index-1]->pose_vertex ));
		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>( all_frames[frame_index]->pose_vertex ));
		e->setMeasurement(odom_val);

		e->setId(total_frame_number+frame_index);
		Vector6d inv_sigma;inv_sigma<<1,1,1,1,1,1;
		inv_sigma = inv_sigma*1.0;
		Matrix6d info = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
		e->setInformation(info);
		graph.addEdge(e);
	  }
	  graph.initializeOptimization();
	  graph.optimize(5); // do optimization!
	  
	  
	  // retrieve the optimization result, for debug visualization
	  for (int j=0;j<=frame_index;j++)
	  {
		all_frames[j]->cam_pose_Tcw = all_frames[j]->pose_vertex->estimate();
		all_frames[j]->cam_pose_Twc = all_frames[j]->cam_pose_Tcw.inverse();
	  }
	  object_landmark* current_landmark = new object_landmark();  current_landmark->cube_vertex = new g2o::VertexCuboid();
	  current_landmark->cube_vertex->setEstimate(vCube->estimate());
	  cube_pose_opti_history[frame_index] = current_landmark;
	  
	  if (all_frames[frame_index]->observed_cuboids.size()>0)
	  {
	      object_landmark* cube_landmark_meas = all_frames[frame_index]->observed_cuboids[0];
	      g2o::cuboid local_cube = cube_landmark_meas->cube_meas;
	      
	      g2o::cuboid global_cube = local_cube.transform_from(all_frames[frame_index]->cam_pose_Twc);
	      object_landmark* tempcuboids2 = new object_landmark();	  tempcuboids2->cube_vertex = new g2o::VertexCuboid();
	      tempcuboids2->cube_vertex->setEstimate(global_cube);
	      cube_pose_raw_detected_history[frame_index] = tempcuboids2;
	  }
	  else
	      cube_pose_raw_detected_history[frame_index] = nullptr;
    }
    
    cout<<"Finish all optimization! Begin visualization."<<endl;
    
    publish_all_poses(all_frames, cube_pose_opti_history,cube_pose_raw_detected_history,truth_frame_poses);      
}



int main(int argc,char* argv[])
{
    ros::init(argc, argv, "object_slam");
    
    ros::NodeHandle nh;

    nh.param ("/base_folder", base_folder, ros::package::getPath("object_slam")+"/data/");
    nh.param ("/online_detect_mode", online_detect_mode, true);
    nh.param ("/save_results_to_txt", save_results_to_txt, false);

    
    cout<<""<<endl;
    cout<<"base_folder   "<<base_folder<<endl;
    if (online_detect_mode)
	ROS_WARN_STREAM("Online detect object mode !!\n");
    else
	ROS_WARN_STREAM("Offline read object mode !!\n");

    //NOTE important
    // in online mode, pred_frame_objects and init_frame_poses are not used. only first frame of truth_frame_poses is used.
    
    std::string pred_objs_txt = base_folder+"detect_cuboids_saved.txt";  // saved cuboids in local ground frame.
    std::string init_camera_pose = base_folder+"pop_cam_poses_saved.txt"; // offline camera pose for cuboids detection (x y yaw=0, truth roll/pitch/height)
    std::string truth_camera_pose = base_folder+"truth_cam_poses.txt";
    Eigen::MatrixXd pred_frame_objects(100,10);  // 100 is some large row number, each row in txt has 10 numbers
    Eigen::MatrixXd init_frame_poses(100,8);
    Eigen::MatrixXd truth_frame_poses(100,8);

    if (!read_all_number_txt(pred_objs_txt,pred_frame_objects))
	return -1;
    if (!read_all_number_txt(init_camera_pose,init_frame_poses))
	return -1;
    if (!read_all_number_txt(truth_camera_pose,truth_frame_poses))
	return -1;
        
    std::cout<<"read data size:  "<<pred_frame_objects.rows()<<"  "<<init_frame_poses.rows()<<"  "<<truth_frame_poses.rows()<<std::endl;
    
    incremental_build_graph(pred_frame_objects,init_frame_poses,truth_frame_poses);    
    
    return 0;
}
