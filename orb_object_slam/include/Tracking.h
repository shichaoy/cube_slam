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

#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Frame.h"
#include <mutex>
#include "ORBVocabulary.h"

// #include "ros/ros.h"

#include "Eigen/Dense"
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "unordered_map"

class detect_3d_cuboid;

namespace ORB_SLAM2
{

// class Frame;  // this non-pointer use Frame, so cannot forward.
class KeyFrame;
class Viewer;
class FrameDrawer;
class MapDrawer;
class Map;
class MapPoint;
class MapObject;
class LocalMapping;
class LoopClosing;
class System;
class ORBextractor;
class KeyFrameDatabase;
class Initializer;

class Tracking
{

public:
    Tracking(){}; // for my post mapping...
    Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
             KeyFrameDatabase *pKFDB, const std::string &strSettingPath, const int sensor);

    ~Tracking();

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp);
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp, int msg_seq_id = -1);

    void SetLocalMapper(LocalMapping *pLocalMapper);
    void SetLoopClosing(LoopClosing *pLoopClosing);
    void SetViewer(Viewer *pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const std::string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

public:
    // by me
    Eigen::Matrix3d Kalib;
    Eigen::Matrix3f Kalib_f;
    Eigen::Matrix3d invKalib;
    Eigen::Matrix3f invKalib_f;
    cv::Mat InitToGround, GroundToInit;                     // orb's init camera frame to my ground
    Eigen::Matrix4f InitToGround_eigen, GroundToInit_eigen; // ground to init frame

    void DetectCuboid(KeyFrame *pKF);
    void AssociateCuboids(KeyFrame *pKF); // compare with keypoint feature inside box
    void MonoObjDepthInitialization();    // initilize mono SLAM based on depth map.
    void ReadAllObjecttxt();
    std::vector<Eigen::MatrixXd> all_offline_object_cubes; // each n*12 read all txt together so that don't need to read on the fly

    detect_3d_cuboid *detect_cuboid_obj;
    double obj_det_2d_thre;

    int start_msg_seq_id = -1;

    Eigen::MatrixXd kitti_sequence_img_to_object_detect_ind;
    std::string kitti_raw_sequence_name;

    bool whether_save_online_detected_cuboids;
    bool whether_save_final_optimized_cuboids;
    std::ofstream save_online_detected_cuboids;
    std::ofstream save_final_optimized_cuboids;
    void SaveOptimizedCuboidsToTxt();
    bool done_save_obj_to_txt = false;
    unsigned int final_object_record_frame_ind;

    // ground detection
    float nominal_ground_height;
    float filtered_ground_height;
    std::vector<float> height_esti_history;
    float ground_roi_middle; // 4 for middle 1/2  3 for middle 1/3
    float ground_roi_lower;
    int ground_inlier_pts;
    float ground_dist_ratio;
    int ground_everyKFs;
    unsigned int first_absolute_scale_frameid;
    unsigned int first_absolute_scale_framestamp;

    // by me
    // ros::NodeHandle n;

    bool use_truth_trackid; // for (dynamic) object assocition, testing.
    unordered_map<int, MapObject *> trackletid_to_landmark;
    bool triangulate_dynamic_pts;

    // Tracking states
    enum eTrackingState
    {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation   reference to current
    std::vector<cv::Mat> mlRelativeFramePoses; // for each frame, reference to current
    std::vector<KeyFrame *> mlpReferences;     // for each frame. not include mono initialization frames
    std::vector<double> mlFrameTimes;          // for each frame's time stamp
    std::vector<bool> mlbLost;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

protected:
    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular(); // create map keyframes (the two frames which can initialize) and keypoints

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame(); // frame to keyframe tracking. use ORB feature descriptors for matching (in 2D), instead of projection.
    void UpdateLastFrame();
    bool TrackWithMotionModel(); // track to map points seen in last frame (not keyframe)

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames(); // find reference keyframe and local covisible keyframes

    bool TrackLocalMap();     // frame to local map, more points involved compared to TrackWithMotionModel(). only one frame. map fixed.
    void SearchLocalPoints(); // search map points for current frame

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping *mpLocalMapper;
    LoopClosing *mpLoopClosing;

    //ORB
    ORBextractor *mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor *mpIniORBextractor; // just allow more feature points

    //BoW
    ORBVocabulary *mpORBVocabulary;
    KeyFrameDatabase *mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer *mpInitializer;

    //Local Map
    KeyFrame *mpReferenceKF;                   // the past keyframe which shares most MapPoints with this frame
    std::vector<KeyFrame *> mvpLocalKeyFrames; // local visibile or covisible keyframes../ not include just created KF
    std::vector<MapPoint *> mvpLocalMapPoints; // for track with local map

    // System
    System *mpSystem;

    //Drawers
    Viewer *mpViewer;
    FrameDrawer *mpFrameDrawer;
    MapDrawer *mpMapDrawer;

    //Map
    Map *mpMap;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    int mMinFrames; // by default is 0
    int mMaxFrames; // by default is fps

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    // Current matches in frame.   current frame's matched map point number
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame *mpLastKeyFrame;
    Frame mLastFrame;              // deep copy of current frame!.
    unsigned int mnLastKeyFrameId; // frame ID, not keyframeID
    unsigned int mnLastRelocFrameId;

    //Motion Model  last to current frame
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    std::list<MapPoint *> mlpTemporalPoints;
};

} // namespace ORB_SLAM2

#endif // TRACKING_H
