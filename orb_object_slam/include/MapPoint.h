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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <set>
#include <map>
#include <mutex>
#include <opencv2/core/core.hpp>
#include "Eigen/Dense"

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;
class MapObject;

class MapPoint
{
public:
    MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map *pMap);
    MapPoint(const cv::Mat &Pos, Map *pMap, Frame *pFrame, const int &idxF);

    void SetWorldPos(const cv::Mat &Pos);
    cv::Mat GetWorldPos();
    Eigen::Vector3f GetWorldPosVec();
    cv::Mat GetWorldPosBA();

    cv::Mat GetNormal();
    KeyFrame *GetReferenceKeyFrame();

    std::map<KeyFrame *, size_t> GetObservations();
    int Observations();

    // corresponding to the idx keypoint of keyframe pKF, not the absoluate coordinate
    void AddObservation(KeyFrame *pKF, size_t idx);
    void EraseObservation(KeyFrame *pKF); // called in Keyframe when set bad, and in optimizer when removing outliers

    int GetIndexInKeyFrame(KeyFrame *pKF); // -1 if not exist
    bool IsInKeyFrame(KeyFrame *pKF);      // whether kF is added to observations

    void SetBadFlag();
    bool isBad();

    void Replace(MapPoint *pMP); //usually means replace close point, with similar decriptor. called in matcher.Fuse(), in localmapping
    MapPoint *GetReplaced();

    void IncreaseVisible(int n = 1);
    void IncreaseFound(int n = 1);
    float GetFoundRatio();
    inline int GetFound()
    {
        return mnFound;
    }

    void ComputeDistinctiveDescriptors(); // select/update map point's decriptors based on frame's keypoints' decriptors

    cv::Mat GetDescriptor();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, const float &logScaleFactor);

    // by me, for association with object
    void AddObjectObservation(MapObject *obj); //called by AddObservation
    void EraseObjectObservation(MapObject *obj);
    void FindBestObject(); //find which object observes this point most

    int GetBelongedObject(MapObject *&obj); // change obj, return observation times.
    MapObject *GetBelongedObject();         //return obj

public:
    long unsigned int mnId;
    static long unsigned int nNextId;
    long int mnFirstKFid;
    long int mnFirstFrame;
    int nObs;

    // Variables used by the tracking
    float mTrackProjX;
    float mTrackProjY;
    float mTrackProjXR;
    bool mbTrackInView;
    int mnTrackScaleLevel;
    float mTrackViewCos;
    long unsigned int mnTrackReferenceForFrame; // more like a marker
    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopPointForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    cv::Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;

    // for dynamic stuff
    bool is_dynamic = false;
    cv::Mat mWorldPos_latestKF;
    cv::Mat mWorldPos_latestFrame;
    cv::Mat PosToObj; // 3d point relative to object center. ideally will converge/fix after BA

    bool is_triangulated = false; //whether this point is triangulated or depth inited?
    bool is_optimized = false;

    MapObject *best_object;                        // one point can only belong to at most one object
    int max_object_vote;                           // sometimes point is wrongly associated to an object. need more frame observation
    std::set<MapObject *> LocalObjObservations;    // observed by local objects which hasn't become landmark at that time
    std::map<MapObject *, int> MapObjObservations; //object and observe times.
    std::mutex mMutexObject;

    bool already_bundled;

    bool ground_fitted_point = false;
    long unsigned int mnGroundFittingForKF;

    int record_txtrow_id = -1; // when finally record to txt, row id in txt

protected:
    // Position in absolute coordinates 3*1
    cv::Mat mWorldPos;

    // Keyframes observing the point and associated keypoint index in keyframe
    std::map<KeyFrame *, size_t> mObservations;

    // Mean viewing direction
    cv::Mat mNormalVector;

    // Best descriptor to fast matching
    cv::Mat mDescriptor;

    // Reference KeyFrame, first frame that observes this point
    KeyFrame *mpRefKF;

    // Tracking counters
    int mnVisible;
    int mnFound;

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;
    MapPoint *mpReplaced;

    // Scale invariance distances
    float mfMinDistance;
    float mfMaxDistance;

    Map *mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

} // namespace ORB_SLAM2

#endif // MAPPOINT_H
