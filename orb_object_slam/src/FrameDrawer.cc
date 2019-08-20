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

#include "FrameDrawer.h"
#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"
#include "KeyFrame.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <mutex>

#include "Parameters.h"
#include "Converter.h"
#include "detect_3d_cuboid/object_3d_util.h"
#include "MapObject.h"

using namespace std;

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map *pMap) : mpMap(pMap)
{
    mState = Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    box_colors.push_back(cv::Scalar(255, 255, 0));
    box_colors.push_back(cv::Scalar(255, 0, 255));
    box_colors.push_back(cv::Scalar(0, 255, 255));
    box_colors.push_back(cv::Scalar(145, 30, 180));
    box_colors.push_back(cv::Scalar(210, 245, 60));
    box_colors.push_back(cv::Scalar(128, 0, 0));

    whether_keyframe = false;
}

cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys;     // Initialization: KeyPoints in reference frame
    vector<int> vMatches;              // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap;          // Tracked MapPoints in current frame
    int state;                         // Tracking state

    //for debug visualization
    vector<cv::KeyPoint> vCurrentKeys_inlastframe;
    vector<cv::Point2f> vfeaturesklt_lastframe;
    vector<cv::Point2f> vfeaturesklt_thisframe;

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state = mState;
        if (mState == Tracking::SYSTEM_NOT_READY)
            mState = Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if (mState == Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vCurrentKeys_inlastframe = mvCurrentKeys_inlastframe;
            vfeaturesklt_lastframe = mvfeaturesklt_lastframe;
            vfeaturesklt_thisframe = mvfeaturesklt_thisframe;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if (mState == Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vCurrentKeys_inlastframe = mvCurrentKeys_inlastframe;
            vfeaturesklt_lastframe = mvfeaturesklt_lastframe;
            vfeaturesklt_thisframe = mvfeaturesklt_thisframe;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if (mState == Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
            vCurrentKeys_inlastframe = mvCurrentKeys_inlastframe;
            vfeaturesklt_lastframe = mvfeaturesklt_lastframe;
            vfeaturesklt_thisframe = mvfeaturesklt_thisframe;
        }
    } // destroy scoped mutex -> release mutex

    if (im.channels() < 3) //this should be always true
        cvtColor(im, im, CV_GRAY2BGR);

    //Draw
    if (state == Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for (unsigned int i = 0; i < vMatches.size(); i++)
        {
            if (vMatches[i] >= 0)
            {
                cv::line(im, vIniKeys[i].pt, vCurrentKeys[vMatches[i]].pt,
                         cv::Scalar(0, 255, 0));
            }
        }
    }
    else if (state == Tracking::OK) //TRACKING
    {
        mnTracked = 0;
        mnTrackedVO = 0;
        const float r = 5; // rectangle width
        for (int i = 0; i < N; i++)
        {
            if (vbVO[i] || vbMap[i]) // matched to map, VO point (rgbd/stereo)
            {
                cv::Point2f pt1, pt2;
                pt1.x = vCurrentKeys[i].pt.x - r;
                pt1.y = vCurrentKeys[i].pt.y - r;
                pt2.x = vCurrentKeys[i].pt.x + r;
                pt2.y = vCurrentKeys[i].pt.y + r;

                if (vbMap[i]) // This is a match to a MapPoint in the map    // green
                {
                    cv::circle(im, vCurrentKeys[i].pt, 3, cv::Scalar(0, 255, 0), -1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame    // blue
                {
                    cv::circle(im, vCurrentKeys[i].pt, 3, cv::Scalar(255, 0, 0), -1);
                    mnTrackedVO++;
                }

                if (associate_point_with_object && (point_Object_AssoID.size() > 0)) // red object points
                    if (point_Object_AssoID[i] > -1)
                        cv::circle(im, vCurrentKeys[i].pt, 4, box_colors[point_Object_AssoID[i] % box_colors.size()], -1);

                if (vCurrentKeys_inlastframe.size() > 0 && !(vCurrentKeys_inlastframe[i].pt.x == 0 && vCurrentKeys_inlastframe[i].pt.y == 0))
                    cv::line(im, vCurrentKeys[i].pt, vCurrentKeys_inlastframe[i].pt, cv::Scalar(0, 0, 255), 2);
            }
            else // if not matched to map points. discarded points.
            {
                // 		cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,255),-1);  //magenda
            }
        }

        for (size_t i = 0; i < vfeaturesklt_thisframe.size(); i++)
        {
            if (!(vfeaturesklt_thisframe[i].x == 0 && vfeaturesklt_thisframe[i].y == 0))
            {
                cv::Scalar LineColor = cv::Scalar(0, 0, 255);
                LineColor = cv::Scalar(rand() % (int)(255 + 1), rand() % (int)(255 + 1), rand() % (int)(255 + 1));
                cv::line(im, vfeaturesklt_thisframe[i], vfeaturesklt_lastframe[i], LineColor, 2);
            }
        }
    }

    if (whether_detect_object) //draw object box
    {                          // better to write some warning that if it is keyframe or not, because I only detect cuboid for keyframes.
        for (size_t i = 0; i < bbox_2ds.size(); i++)
        {
            cv::rectangle(im, bbox_2ds[i], box_colors[i % box_colors.size()], 2); // 2d bounding box.
            if ((scene_unique_id != kitti) && (box_corners_2ds[i].cols() > 0))    // for most offline read data, usually cannot read it, could use rviz.
            {
                plot_image_with_cuboid_edges(im, box_corners_2ds[i], edge_markers_2ds[i]); // eight corners.
            }
            if (truth2d_trackid.size() > 0 && 1) //draw truth id
            {
                int font = cv::FONT_HERSHEY_PLAIN;
                char seq_index_c[256];
                sprintf(seq_index_c, "%d", truth2d_trackid[i]);
                std::string show_strings2(seq_index_c);
                cv::putText(im, show_strings2, cv::Point(bbox_2ds[i].x + 10, bbox_2ds[i].y + 10), font, 2, cv::Scalar(0, 0, 255), 2, 8); // # bgr
            }
        }
    }

    // draw ground pts
    for (size_t i = 0; i < potential_ground_fit_inds.size(); i++)
    {
        if (vbVO[i] || vbMap[i])
        {
            cv::circle(im, vCurrentKeys[potential_ground_fit_inds[i]].pt, 2, cv::Scalar(0, 0, 255), -1);
        }
    }
    cv::Mat imWithInfo;
    DrawTextInfo(im, state, imWithInfo);

    return imWithInfo;
}

void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if (nState == Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if (nState == Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if (nState == Tracking::OK)
    {
        if (!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if (mnTrackedVO > 0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if (nState == Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if (nState == Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

    imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
    im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
    imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
    cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

void FrameDrawer::Update(Tracking *pTracker)
{
    if (!enable_viewimage)
        return;

    unique_lock<mutex> lock(mMutex);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    mvbVO = vector<bool>(N, false);
    mvbMap = vector<bool>(N, false);
    mbOnlyTracking = pTracker->mbOnlyTracking;

    potential_ground_fit_inds.clear();
    current_frame_id = int(pTracker->mCurrentFrame.mnId);

    // for visualization
    mvCurrentKeys_inlastframe = pTracker->mCurrentFrame.mvpMapPoints_inlastframe;
    mvfeaturesklt_lastframe = pTracker->mCurrentFrame.featuresklt_lastframe;
    mvfeaturesklt_thisframe = pTracker->mCurrentFrame.featuresklt_thisframe;

    if (pTracker->mLastProcessedState == Tracking::NOT_INITIALIZED)
    {
        mvIniKeys = pTracker->mInitialFrame.mvKeys; // deep copy... takes time
        mvIniMatches = pTracker->mvIniMatches;
    }
    else if (pTracker->mLastProcessedState == Tracking::OK)
    {
        for (int i = 0; i < N; i++)
        {
            MapPoint *pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if (pMP)
            {
                if (!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if (pMP->Observations() > 0) // if it is already added to map points
                        mvbMap[i] = true;
                    else
                        mvbVO[i] = true; // associated map points, but not yet add observation???
                }
            }
        }
    }
    mState = static_cast<int>(pTracker->mLastProcessedState);

    if (whether_detect_object) // copy some object data for visualization
    {
        cam_pose_cw = pTracker->mCurrentFrame.mTcw; // camera pose, world to cam
        Kalib = pTracker->Kalib;
        bbox_2ds.clear();
        truth2d_trackid.clear();
        box_corners_2ds.clear();
        edge_markers_2ds.clear();
        point_Object_AssoID.clear();
        if (pTracker->mCurrentFrame.mpReferenceKF != NULL)                                             // mCurrentFrame.mpReferenceKF
            if ((pTracker->mCurrentFrame.mnId - pTracker->mCurrentFrame.mpReferenceKF->mnFrameId) < 1) // if current frame is a keyframe
            {
                if (whether_detect_object)
                {
                    for (const MapObject *object : pTracker->mCurrentFrame.mpReferenceKF->local_cuboids)
                    {
                        bbox_2ds.push_back(object->bbox_2d);
                        box_corners_2ds.push_back(object->box_corners_2d);
                        edge_markers_2ds.push_back(object->edge_markers);
                        if (pTracker->use_truth_trackid)
                            truth2d_trackid.push_back(object->truth_tracklet_id);
                    }

                    if (associate_point_with_object)
                        point_Object_AssoID = pTracker->mCurrentFrame.mpReferenceKF->keypoint_associate_objectID;
                }
            }
    }

    if (enable_ground_height_scale)
    {
        if (pTracker->mCurrentFrame.mpReferenceKF != NULL)                                             // mCurrentFrame.mpReferenceKF
            if ((pTracker->mCurrentFrame.mnId - pTracker->mCurrentFrame.mpReferenceKF->mnFrameId) < 1) // if current frame is a keyframe
            {
                potential_ground_fit_inds = pTracker->mCurrentFrame.mpReferenceKF->ground_region_potential_pts;
            }
    }
}

} // namespace ORB_SLAM2
