/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <ros/ros.h>
#include <ros/package.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>

#include "System.h"
#include "Parameters.h"
#include "tictoc_profiler/profiler.hpp"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System *pSLAM) : mpSLAM(pSLAM) {}

    void GrabStereo(const sensor_msgs::ImageConstPtr &msgleft, const sensor_msgs::ImageConstPtr &msgright);

    ORB_SLAM2::System *mpSLAM;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if (argc != 3)
    {
        cerr << endl
             << "Usage: rosrun ORB_SLAM2 RGBD path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }
    ros::NodeHandle nodeHandler;

    bool enable_loop_closing = true;
    nodeHandler.param<bool>("enable_loop_closing", enable_loop_closing, true);
    nodeHandler.param<bool>("whether_detect_object", ORB_SLAM2::whether_detect_object, false);
    nodeHandler.param<bool>("whether_read_offline_cuboidtxt", ORB_SLAM2::whether_read_offline_cuboidtxt, false);
    nodeHandler.param<bool>("enable_viewer", ORB_SLAM2::enable_viewer, true);
    nodeHandler.param<bool>("enable_viewmap", ORB_SLAM2::enable_viewmap, true);
    nodeHandler.param<bool>("enable_viewimage", ORB_SLAM2::enable_viewimage, true);
    nodeHandler.param<bool>("mono_firstframe_truth_depth_init", ORB_SLAM2::mono_firstframe_truth_depth_init, false);
    nodeHandler.param<bool>("mono_firstframe_Obj_depth_init", ORB_SLAM2::mono_firstframe_Obj_depth_init, false);
    nodeHandler.param<bool>("mono_allframe_Obj_depth_init", ORB_SLAM2::mono_allframe_Obj_depth_init, false);
    nodeHandler.param<bool>("associate_point_with_object", ORB_SLAM2::associate_point_with_object, false);
    nodeHandler.param<bool>("parallel_mapping", ORB_SLAM2::parallel_mapping, true);
    std::string scene_name;
    ros::param::get("/scene_name", scene_name);
    ros::param::get("/base_data_folder", ORB_SLAM2::base_data_folder);

    if (scene_name.compare(std::string("kitti")) == 0)
        ORB_SLAM2::scene_unique_id = ORB_SLAM2::kitti;

    if (!enable_loop_closing)
        ROS_WARN_STREAM("Turn off global loop closing!!");
    else
        ROS_WARN_STREAM("Turn on global loop closing!!");

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::STEREO, enable_loop_closing);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> rgb_left_sub(nh, "/stereo/left/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> rgb_right_sub(nh, "/stereo/right/image_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_left_sub, rgb_right_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo, &igb, _1, _2));

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    std::string packagePath = ros::package::getPath("orb_slam2");
    SLAM.SaveKeyFrameTrajectoryTUM(packagePath + "/Outputs/KeyFrameTrajectory.txt"); // can also call saveKitti
    SLAM.SaveTrajectoryTUM(packagePath + "/Outputs/AllFrameTrajectory.txt");

    if (ORB_SLAM2::scene_unique_id == ORB_SLAM2::kitti)
        SLAM.SaveTrajectoryKITTI(packagePath + "/Outputs/AllFrameTrajectoryKITTI.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr &msgleft, const sensor_msgs::ImageConstPtr &msgright)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGBLeft;
    try
    {
        cv_ptrRGBLeft = cv_bridge::toCvShare(msgleft);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRGBRight;
    try
    {
        cv_ptrRGBRight = cv_bridge::toCvShare(msgright);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    mpSLAM->TrackStereo(cv_ptrRGBLeft->image, cv_ptrRGBRight->image, cv_ptrRGBLeft->header.stamp.toSec());
}
