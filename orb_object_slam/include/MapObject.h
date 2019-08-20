#pragma once

#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <g2o_Object.h>
#include <mutex>
#include <opencv2/core.hpp>
#include <unordered_map>

#include "KeyFrame.h"

namespace ORB_SLAM2
{
// class KeyFrame;
class MapPoint;
class Map;

struct cmpKeyframe
{ //sort frame based on ID
    bool operator()(const KeyFrame *a, const KeyFrame *b) const
    {
        return a->mnId < b->mnId;
    }
};

class MapObject
{
public:
    // whether update global unique map object ID. only used in keyframe creation.
    MapObject(Map *pMap, bool update_index = false);

    void SetWorldPos(const g2o::cuboid &Pos);
    g2o::cuboid GetWorldPos();     // get cuboid pose in world/init frame.
    g2o::SE3Quat GetWorldPosInv(); // get cuboid pose in world/init frame.
    g2o::cuboid GetWorldPosBA();   //get latest pose after BA

    void addObservation(KeyFrame *pKF, size_t idx); // object observed by frames
    void EraseObservation(KeyFrame *pKF);           // called in Keyframe when set bad, and in optimizer when removing outliers

    std::unordered_map<KeyFrame *, size_t> GetObservations();
    int Observations();
    std::vector<KeyFrame *> GetObserveFrames();
    std::vector<KeyFrame *> GetObserveFramesSequential();
    std::vector<KeyFrame *> observed_frames; // sequential push back  check bad before use

    long int mnId;           // unique id for this landmark
    static long int nNextId; // static member, automatically add 1 when new (true)
    static long int getIncrementedIndex();

    bool IsInKeyFrame(KeyFrame *pKF);      // whether observed by this kF
    int GetIndexInKeyFrame(KeyFrame *pKF); // get the local cuboids ID in pkf
    KeyFrame *GetReferenceKeyFrame();
    void SetReferenceKeyFrame(KeyFrame *refkf);
    KeyFrame *GetLatestKeyFrame();

    void SetBadFlag();
    bool isBad(); // whether definitly bad.
    bool isGood;  // whether definitely good.

    std::vector<MapPoint *> GetUniqueMapPoints();
    int NumUniqueMapPoints();
    void AddUniqueMapPoint(MapPoint *pMP, int obs_num); // called by pointAddobjectObservations
    void EraseUniqueMapPoint(MapPoint *pMP, int obs_num);

    std::vector<MapPoint *> GetPotentialMapPoints();
    void AddPotentialMapPoint(MapPoint *pMP);
    int largest_point_observations; // largest point observations. better use a heap....
    int pointOwnedThreshold;        // some threshold to whether use points in optimization.  set in optimizer.

    bool check_whether_valid_object(int own_point_thre = 30);

    void MergeIntoLandmark(MapObject *otherLocalObject); // merge the points here. this should already be a landmark. observation should be added elsewhere
    void SetAsLandmark();

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    bool obj_been_optimized = false;
    int point_object_BA_counter = -1;

    long unsigned int association_refid_in_tracking;

    std::vector<MapPoint *> used_points_in_BA;          // points with enough observation will be used in BA.
    std::vector<MapPoint *> used_points_in_BA_filtered; // actually BA used points, after removing some far away points

    static std::mutex mGlobalMutex;

    bool is_dynamic = false;

    Eigen::Vector2d velocityPlanar; //actually used, for kitti cars
    g2o::cuboid pose_Twc_latestKF;
    std::map<KeyFrame *, Eigen::Vector2d, cmpKeyframe> velocityhistory; // for offline analysis
    g2o::cuboid pose_Twc_afterba;                                       // latest pose after BA. might have some delay compared to pose_Twc_latestKF

    std::map<KeyFrame *, std::pair<g2o::cuboid, bool>, cmpKeyframe> allDynamicPoses; // poses/velocity in each keyframe due to movement.  poses/whether_BA
    std::unordered_map<KeyFrame *, int> bundle_vertex_ids;
    int truth_tracklet_id;

    Vector6d velocityTwist;                                    //general 6dof twist. for cars can assume no roll pitch   pose_Twc*exp(twist)=newpose
    g2o::SE3Quat getMovePose(KeyFrame *kf, double deltaT = 0); // deltaT relative to kf. works for short period where velocity doesn't change much

    //----------for local MapObject--------     no mutex needed, for local cuboid storage, not landmark
    int object_id_in_localKF;        // object id in reference keyframe's local objects.
    Eigen::Matrix2Xi box_corners_2d; // 2*8 on image  usually for local cuboids on reference frame.
    Eigen::MatrixXi edge_markers;    // in order to plot 2d cuboids with 8 corners.
    cv::Rect bbox_2d;                // (integer) yolo detected 2D bbox_2d x y w h
    Eigen::Vector4d bbox_vec;        // center, width, height
    cv::Rect bbox_2d_tight;          // tighted 2d object, used to find points association.
    double meas_quality;             // [0,1] the higher, the better
    g2o::cuboid cube_meas;           //local measurement in camera frame
    bool already_associated;
    bool become_candidate;
    MapObject *associated_landmark; // might be itself
    int left_right_to_car;          // on the left or right of a car. left=1 right=2 undecided=0   inititial=-1

    g2o::cuboid pose_noopti;

    int record_txtrow_id = -1;

protected:
    g2o::cuboid pose_Twc; // cuboid pose to the init/world. initialized as the position from first observe frame
    g2o::cuboid pose_Tcw; //inverse, not frequently used

    // Keyframes observing the object and associated localcuboid index in keyframe
    std::unordered_map<KeyFrame *, size_t> mObservations;
    KeyFrame *moRefKF;   // Reference KeyFrame  first frame see this.
    KeyFrame *mLatestKF; // latest frame see this.
    int nObs;            // num of frame observations

    // Bad flag (we do not currently erase MapPoint from memory)
    bool mbBad;

    Map *mpMap;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
    std::mutex mMutexParam;

    std::set<MapPoint *> mappoints_unique_own;    // uniquedly owned by this object.
    std::set<MapPoint *> mappoints_potential_own; // potentially owned
};

} // namespace ORB_SLAM2
