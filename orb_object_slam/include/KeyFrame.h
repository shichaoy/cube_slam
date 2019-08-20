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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"

#include <mutex>
#include "Eigen/Dense"
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;
class MapObject;
class ORBextractor;
class KeyFrame
{
public:
    // everything is deep copied independently. not shared memory
    KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB);

    // Pose functions
    void SetPose(const cv::Mat &Tcw); // set world to camera
    cv::Mat GetPose();                // get world to camera pose
    cv::Mat GetPoseInverse();         // get camera to world pose
    cv::Mat GetCameraCenter();        // Twc
    cv::Mat GetStereoCenter();
    cv::Mat GetRotation();
    cv::Mat GetTranslation(); // Tcw

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(KeyFrame *pKF, const int &weight);
    void EraseConnection(KeyFrame *pKF);
    void UpdateConnections(); // find KFs that also observe points of current frame. sort in order
    void UpdateBestCovisibles();
    std::set<KeyFrame *> GetConnectedKeyFrames();
    std::vector<KeyFrame *> GetVectorCovisibleKeyFrames();
    std::vector<KeyFrame *> GetBestCovisibilityKeyFrames(const int &N); // select the topN similar images, that have most point matches.
    std::vector<KeyFrame *> GetCovisiblesByWeight(const int &w);
    int GetWeight(KeyFrame *pKF);

    // Spanning tree functions
    void AddChild(KeyFrame *pKF);
    void EraseChild(KeyFrame *pKF);
    void ChangeParent(KeyFrame *pKF);
    std::set<KeyFrame *> GetChilds();
    KeyFrame *GetParent();
    bool hasChild(KeyFrame *pKF);

    // Loop Edges
    void AddLoopEdge(KeyFrame *pKF);
    std::set<KeyFrame *> GetLoopEdges();

    // by me. create simple map points, using 3D position.
    void SetupSimpleMapPoints(MapPoint *pNewMP, int point_ind);

    // MapPoint observation functions
    void AddMapPoint(MapPoint *pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseHarrisMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint *pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint *pMP);
    std::set<MapPoint *> GetMapPoints();          // only good associated map points
    std::vector<MapPoint *> GetMapPointMatches(); // same length as keypoints. if keypoint not matched, nullptr
    int TrackedMapPoints(const int &minObs);
    MapPoint *GetMapPoint(const size_t &idx);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r) const;
    cv::Mat UnprojectStereo(int i);             // return world point
    cv::Mat UnprojectDepth(int i, float depth); // return world point

    // Image
    bool IsInImage(const float &x, const float &y) const;

    // Enable/Disable bad flag changes
    void SetNotErase();
    void SetErase();

    // Set/check bad flag, will remove from map.
    void SetBadFlag();
    bool isBad();

    // Compute Scene Depth (q=2 median). Used in monocular depth initialization. sort the depth...
    float ComputeSceneMedianDepth(const int q);

    static bool weightComp(int a, int b)
    {
        return a > b;
    }

    static bool lId(KeyFrame *pKF1, KeyFrame *pKF2)
    {
        return pKF1->mnId < pKF2->mnId;
    }

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:
    // by me, detect_3d_cuboid needs canny edge.
    cv::Mat raw_img;

    // NOTE the object_landmark vector need to push back, not pre-allocated vector
    // landmarks are copied from local cubes, not new-created

    // generated 3d cuboid, might be shorted than yolo 2d boxes. because some 2d box might not generate 3d object.
    std::vector<MapObject *> local_cuboids;       // actual local generated cuboid from this frame. not associated yet. measurement is important
    std::vector<MapObject *> cuboids_landmark;    // check if exist or bad before use. associated SLAM map landmark, might be shorter than local_cuboids. copied from local_cuboids pointer. push when need
    std::vector<int> keypoint_associate_objectID; // same length as keypoints. point-object associations  -1: no associated object, 0,1...  associated object ID in local cubes.  one keypoint uniquely associate a object.
    std::vector<cv::Rect> object_2d_rectangles;   // all local_cuboids's 2d rectangles  mainly utility/debug use

    // dynamic
    std::vector<cv::KeyPoint> mvKeysHarris;     // only, extra, dynamic keypoints   Harris corner, not orb features.
    std::vector<MapPoint *> mvpMapPointsHarris; // only, extra, dynamic mappoints.
    std::vector<int> keypoint_associate_objectID_harris;
    std::vector<MapPoint *> GetHarrisMapPointMatches();

    std::vector<bool> KeysStatic; // whether point is static

    cv::Mat UnprojectPixelDepth(cv::Point2f &pt, float depth);

    // compute depth in an image region.
    std::vector<std::vector<float>> allGridMedianDepth;
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY); //x horizontal   y vertical
    bool PosInGrid(int ptx, int pty, int &posX, int &posY);

    void EraseMapObjectMatch(const size_t &idx);
    void EraseMapObjectMatch(MapObject *pMP);

    bool frame_object_being_drawed = false;

    // for ground scaling methods.
    std::vector<int> ground_region_potential_pts; // inds of lower 1/3 and middle pts. so that don't need to check again.
    std::vector<bool> keypoint_inany_object;      // any 2d bbox, nothing to do with 3d, just want to remove them.
    long unsigned int mnGroundFittingForKF = 0;

    static long unsigned int nNextId;
    long unsigned int mnId;            // key frame id
    const long unsigned int mnFrameId; // frame id in all images

    const double mTimeStamp;

    int record_txtrow_id = -1; //final save to txt row id.

    // Grid (to speed up feature matching)
    const int mnGridCols;
    const int mnGridRows;
    const float mfGridElementWidthInv;
    const float mfGridElementHeightInv;

    // Variables used by the tracking
    long unsigned int mnTrackReferenceForFrame;
    long unsigned int mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBAFixedForKF;

    // Variables used by the keyframe database
    long unsigned int mnLoopQuery;
    int mnLoopWords;
    float mLoopScore;
    long unsigned int mnRelocQuery;
    int mnRelocWords;
    float mRelocScore;

    // Variables used by loop closing
    cv::Mat mTcwGBA;
    cv::Mat mTcwBefGBA;
    long unsigned int mnBAGlobalForKF;

    // Calibration parameters
    const float fx, fy, cx, cy, invfx, invfy, mbf, mb, mThDepth;

    // Number of KeyPoints
    const int N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight; // negative value for monocular points
    const std::vector<float> mvDepth;  // negative value for monocular points
    const cv::Mat mDescriptors;

    //BoW
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Scale
    const int mnScaleLevels;
    const float mfScaleFactor;
    const float mfLogScaleFactor;
    const std::vector<float> mvScaleFactors;
    const std::vector<float> mvLevelSigma2;
    const std::vector<float> mvInvLevelSigma2; // 1  0.69 0.48 0.33 ...

    // Image bounds and calibration
    const int mnMinX;
    const int mnMinY;
    const int mnMaxX;
    const int mnMaxY;
    const cv::Mat mK;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t>>> mGrid;

    // The following variables need to be accessed trough a mutex to be thread safe.
protected:
    // SE3 Pose and camera center
    cv::Mat Tcw; //world to camera
    cv::Mat Twc; //camera to world
    cv::Mat Ow;  //translation, camera to world

    cv::Mat Cw; // Stereo middel point. Only for visualization

    // MapPoints associated to keypoints, NULL pointer if no association. same length as keypoints.
    // for stereo/RGBD, some close keypoints are direactly associated with map points
    // for monocular, need to triangulate with old frame
    std::vector<MapPoint *> mvpMapPoints; // has nothing to do with Frame. mvpMapPoints after copying! private variable!!!! only add by AddMapPoint()

    // BoW
    KeyFrameDatabase *mpKeyFrameDB;
    ORBVocabulary *mpORBvocabulary;

    std::map<KeyFrame *, int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame *> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool mbFirstConnection;
    KeyFrame *mpParent;
    std::set<KeyFrame *> mspChildrens;
    std::set<KeyFrame *> mspLoopEdges;

    // Bad flags
    bool mbNotErase;
    bool mbToBeErased;
    bool mbBad;

    float mHalfBaseline; // Only for visualization

    Map *mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

} // namespace ORB_SLAM2

#endif // KEYFRAME_H
