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

#include "KeyFrame.h"
#include "LocalMapping.h"
#include "Map.h"
#include "Tracking.h"
#include "MapPoint.h"

#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include "Parameters.h"
#include <mutex>

// by me
#include <ctime>
#include <algorithm> // std::random_shuffle
#include "tictoc_profiler/profiler.hpp"
#include "MapObject.h"

using namespace std;

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular) : mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
                                                                mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker = pTracker;
}

void LocalMapping::bundle_adjustment_caller()
{
    if (whether_detect_object && bundle_object_opti)
    {
        ca::Profiler::tictoc("BA Total Object");
        if (whether_dynamic_object)
            Optimizer::LocalBACameraPointObjectsDynamic(mpCurrentKeyFrame, &mbAbortBA, mpMap, false, false);
        else
            Optimizer::LocalBACameraPointObjects(mpCurrentKeyFrame, &mbAbortBA, mpMap, false, false);
        ca::Profiler::tictoc("BA Total Object");
    }
    else
    {
        ca::Profiler::tictoc("Normal BA");
        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);
        ca::Profiler::tictoc("Normal BA");
    }
}

bool LocalMapping::RunMappingIteration()
{
    //     Tracking will see that Local Mapping is busy   if BA processing KF now, don't accept new KF
    SetAcceptKeyFrames(false);

    //     Check if there are keyframes in the queue
    if (CheckNewKeyFrames())
    {
        // BoW conversion and insertion in Map
        ProcessNewKeyFrame();

        // Check recent MapPoints
        MapPointCulling();

        // Triangulate new MapPoints, will update some point matches for some older keyframes.
        CreateNewMapPoints();

        if (!CheckNewKeyFrames()) // until there is no more keyframes to process
        {
            // Find more matches in neighbor keyframes and fuse point duplications
            SearchInNeighbors();
        }

        mbAbortBA = false;

        if (!CheckNewKeyFrames() && !stopRequested()) // after all keyframes being inserted
        {
            // Local BA
            if (mpMap->KeyFramesInMap() > 2)
            {
                // don't immediately BA if first frame is depth intialized, only a few frames, baseline is small, will damage the map.
                if ((mbMonocular) && (mono_firstframe_truth_depth_init || mono_firstframe_Obj_depth_init) && (scene_unique_id != kitti))
                {
                    if (mpMap->KeyFramesInMap() > 5)
                    {
                        bundle_adjustment_caller();
                    }
                }
                else
                {
                    bundle_adjustment_caller();
                }
            }

            // Check redundant local Keyframes, could delete frame is many points in this frame are observed before
            KeyFrameCulling();
        }

        mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
    }
    else if (Stop())
    {
        // Safe area to stop
        while (isStopped() && !CheckFinish())
        {
            usleep(3000);
        }
        if (CheckFinish())
            return false;
    }

    ResetIfRequested();

    // Tracking will see that Local Mapping is idle
    SetAcceptKeyFrames(true);

    if (CheckFinish())
        return false;

    return true;
}

void LocalMapping::Run()
{
    mbFinished = false;
    while (1)
    {
        if (!RunMappingIteration())
            break;

        usleep(3000);
    }
    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA = true;
}

bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return (!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front(); // the oldest
        mlNewKeyFrames.pop_front();                 // remove it from list
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    int newaddobservation = 0;
    int newaddRecent = 0;

    for (size_t i = 0; i < vpMapPointMatches.size(); i++) //add frame observation for associated map points
    {
        MapPoint *pMP = vpMapPointMatches[i];
        // in mono/stereo/RGBD depth initialization, or stereo/RGBG create new KF,   new created map points already added to observation
        // in tracking, there is also matched points to previous frame, not yet add to observation

        if (pMP)
        {
            if (!pMP->isBad())
            {
                if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) // associated   not yet add to observation
                {
                    newaddobservation++;
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else
                {
                    if (pMP->is_dynamic) // don't need to cull dynamic points
                        continue;
                    newaddRecent++;
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }
    if (use_dynamic_klt_features)
    {
        const vector<MapPoint *> vpHarrisMapPointMatches = mpCurrentKeyFrame->GetHarrisMapPointMatches();
        for (size_t i = 0; i < vpHarrisMapPointMatches.size(); i++)
        {
            MapPoint *pMP = vpHarrisMapPointMatches[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                    {
                        pMP->AddObservation(mpCurrentKeyFrame, i);
                    }
                }
            }
        }
    }

    std::cout << "Mapping: recent buffer pts:  " << mlpRecentAddedMapPoints.size() << "   Frame created pts:  " << newaddRecent
              << "   Frame associated pts:   " << newaddobservation << std::endl;

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if (mbMonocular)
    {
        nThObs = 2;
        // my new added point from Monocular depth initilalization only has one observation, should not delete
        if (mono_allframe_Obj_depth_init)
            nThObs = 1;
    }
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    int delete_case_1 = 0;
    int delete_case_2 = 0;
    int delete_case_3 = 0;
    int delete_case_4 = 0;

    while (lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint *pMP = *lit;
        if (pMP->isBad() || pMP->is_dynamic)
        {
            delete_case_1++;
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (pMP->GetFoundRatio() < 0.25f)
        {
            delete_case_2++;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs)
        {
            delete_case_3++;
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
        {
            delete_case_4++;
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            lit++;
        }
    }
}

template <class BidiIter> //Fisher-Yates shuffle    only the first num_random of the output vector is shuffled
BidiIter random_unique(BidiIter begin, BidiIter end, size_t num_random)
{
    size_t left = std::distance(begin, end);
    while (num_random--)
    {
        BidiIter r = begin;
        std::advance(r, rand() % left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if (mbMonocular)
        nn = 20;
    const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6, false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation(); //world to camera
    cv::Mat Rwc1 = Rcw1.t();                         //camera to world
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    for (size_t i = 0; i < vpNeighKFs.size(); i++)
    {
        if (i > 0 && CheckNewKeyFrames())
            return;

        KeyFrame *pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2 - Ow1;
        const float baseline = cv::norm(vBaseline);
        if (!mbMonocular) // if not monocular, compare with fixed baseline.
        {
            if (baseline < pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline / medianDepthKF2;

            if (ratioBaselineDepth < 0.01)
                continue;
        }

        // Compute Fundamental Matrix, based on pose.
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t, size_t>> vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0, 3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for (int ikp = 0; ikp < nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            if (whether_dynamic_object) // deal with dynamic points separately
            {
                if (mpCurrentKeyFrame->KeysStatic.size() > 0 && !mpCurrentKeyFrame->KeysStatic[idx1])
                    continue;
                if (pKF2->KeysStatic.size() > 0 && !pKF2->KeysStatic[idx2])
                    continue;
            }

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur >= 0; // true for stereo/rgbd camera

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur >= 0; // true for stereo/rgbd camera

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2)); // parallax in keyframe 1 and keyframe 2

            float cosParallaxStereo = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if (bStereo1) // parallax in left/right image of Keyframe 1
                cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2, mpCurrentKeyFrame->mvDepth[idx1]));
            else if (bStereo2) // parallax in left/right image of keyframe 2
                cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

            cv::Mat x3D;
            if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<float>(3) == 0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
            }
            else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1); // create this keypoint from KF 1
            }
            else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2); // create this keypoint from KF 1
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
            if (z1 <= 0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
            if (z2 <= 0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
            const float invz1 = 1.0 / z1;

            if (!bStereo1)
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
            const float invz2 = 1.0 / z2;
            if (!bStereo2)
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D - Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D - Ow2;
            float dist2 = cv::norm(normal2);

            if (dist1 == 0 || dist2 == 0)
                continue;

            const float ratioDist = dist2 / dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

            pMP->AddObservation(mpCurrentKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP); // check the new created points!!

            nnew++;
        }
    }
    std::cout << "New Triangulated pt num:   " << nnew << std::endl;
    // for features that haven't triangulated, use object depth initiliaztion.
    // don't do initialization here for dynamic objects. do it in tracking.cc
    // NOTE doesn't change tracker->mCurrentFrame, not pointer!   Keyframe is shared.
    if (mono_allframe_Obj_depth_init && !whether_dynamic_object)
    {
        const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

        double total_feature_pts = vpMapPointMatches.size();
        double raw_depth_pts = 0; // point already have depth
        double object_initialized_pts = 0;
        std::vector<int> raw_pixels_no_depth_inds;

        if (1)
        {
            bool actually_use_obj_depth = false;
            if (mono_allframe_Obj_depth_init && whether_detect_object && associate_point_with_object)
                if (mpCurrentKeyFrame->keypoint_associate_objectID.size() > 0)
                    actually_use_obj_depth = true;

            if (actually_use_obj_depth)
            {
                std::vector<int> has_object_depth_pixel_inds; // points with no depth yet but with matching object
                for (size_t i = 0; i < vpMapPointMatches.size(); i++)
                {
                    MapPoint *pMP = vpMapPointMatches[i];
                    if (!pMP)
                    {
                        if (mpCurrentKeyFrame->keypoint_associate_objectID[i] > -1) // have associated object
                            has_object_depth_pixel_inds.push_back(i);
                    }
                    else
                        raw_depth_pts++;
                }
                bool whether_actually_object_init_pt = false;

                double depth_point_ration_now = raw_depth_pts / total_feature_pts;

                int max_initialize_pts = 0;
                if (depth_point_ration_now < 0.30) //0.3
                    whether_actually_object_init_pt = true;
                max_initialize_pts = std::min(int(total_feature_pts * 0.30) - int(raw_depth_pts), int(has_object_depth_pixel_inds.size()));
                int nPoints = 0;

                if (whether_actually_object_init_pt)
                {
                    srand(time(NULL));
                    random_unique(has_object_depth_pixel_inds.begin(), has_object_depth_pixel_inds.end(), max_initialize_pts);

                    int vector_counter = 0;
                    while ((nPoints < max_initialize_pts) && (vector_counter < (int)has_object_depth_pixel_inds.size()))
                    {
                        int pixel_ind = has_object_depth_pixel_inds[vector_counter];
                        float point_depth = -1;
                        cv::Mat x3D;
                        if ((point_depth < 0) && actually_use_obj_depth)
                        {
                            if (mpCurrentKeyFrame->keypoint_associate_objectID[pixel_ind] > -1)
                            {
                                point_depth = mpCurrentKeyFrame->local_cuboids[mpCurrentKeyFrame->keypoint_associate_objectID[pixel_ind]]->cube_meas.translation()[2]; // camera z
                                x3D = mpCurrentKeyFrame->UnprojectDepth(pixel_ind, point_depth);
                            }
                        }
                        if (point_depth > 0)
                        {
                            MapPoint *pNewMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);
                            mpCurrentKeyFrame->SetupSimpleMapPoints(pNewMP, pixel_ind); // add to frame observation, add to map.
                            nPoints++;
                            mlpRecentAddedMapPoints.push_back(pNewMP);
                            if (whether_dynamic_object && actually_use_obj_depth)
                            {
                                pNewMP->is_dynamic = true;
                                pNewMP->SetWorldPos(x3D); // compute dynamic point to object pose
                            }
                        }
                        vector_counter++;
                    }
                    object_initialized_pts = nPoints;
                    std::cout << "Online depth initilaized mappoints!!  " << nPoints << std::endl;
                }
            }
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if (mbMonocular)
        nn = 20;
    const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame *> vpTargetKFs;
    for (vector<KeyFrame *>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;
        if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame *> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for (vector<KeyFrame *>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
        {
            KeyFrame *pKFi2 = *vit2;
            if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId || pKFi2->mnId == mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (vector<KeyFrame *>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
    {
        KeyFrame *pKFi = *vit;

        matcher.Fuse(pKFi, vpMapPointMatches); //merge close points, with similar descriptor.
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint *> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

    for (vector<KeyFrame *>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
    {
        KeyFrame *pKFi = *vitKF;

        vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for (vector<MapPoint *>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
        {
            MapPoint *pMP = *vitMP;
            if (!pMP)
                continue;
            if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            if (pMP->is_dynamic)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    // fuse map point duplications in neighbor keyframes
    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP)
        {
            if (!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    return K1.t().inv() * t12x * R12 * K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if (mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if (mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames = flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if (flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
    {
        KeyFrame *pKF = *vit;
        if (pKF->mnId == 0)
            continue;
        const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

        int /*nObs = 2;
        if(mbMonocular)*/
            nObs = 3;
        const int thObs = nObs;
        int nRedundantObservations = 0;
        int nMPs = 0;
        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP)
            {
                if (!pMP->isBad())
                {
                    if (!mbMonocular)
                    {
                        if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
                            continue;
                    }

                    if (pMP->is_dynamic)
                        continue;

                    nMPs++;
                    if (pMP->Observations() > thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                        int nObs = 0;
                        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                        {
                            KeyFrame *pKFi = mit->first;
                            if (pKFi == pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if (scaleLeveli <= scaleLevel + 1)
                            {
                                nObs++;
                                if (nObs >= thObs)
                                    break;
                            }
                        }
                        if (nObs >= thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }

        if (nRedundantObservations > 0.9 * nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2), 0, -v.at<float>(0),
            -v.at<float>(1), v.at<float>(0), 0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while (1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if (!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if (mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested = false;
    }
} // namespace ORB_SLAM2

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} // namespace ORB_SLAM2
