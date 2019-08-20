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

#ifndef MAP_H
#define MAP_H

#include <set>
#include <mutex>
#include <vector>
#include <opencv/cv.h>

#include "Eigen/Dense"
#include <Eigen/Geometry>

namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class MapObject;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame *pKF);
    void AddMapPoint(MapPoint *pMP);
    void AddMapObject(MapObject *pMO);
    void EraseMapPoint(MapPoint *pMP);
    void EraseKeyFrame(KeyFrame *pKF); // called when pKF is set bad
    void EraseMapObject(MapObject *pMO);
    void SetReferenceMapPoints(const std::vector<MapPoint *> &vpMPs);

    std::vector<KeyFrame *> GetAllKeyFrames();           // not sequential.... as use set, some frame might delete due to mapping culling
    std::vector<KeyFrame *> GetAllKeyFramesSequential(); // sequential  check good before use
    std::vector<MapPoint *> GetAllMapPoints();
    std::vector<MapObject *> GetAllMapObjects(); // not sequential....
    std::vector<MapObject *> GetGoodMapObjects();
    std::vector<MapPoint *> GetReferenceMapPoints();

    long unsigned int MapPointsInMap();  // get number of points
    long unsigned int KeyFramesInMap();  // get number of keyframes.
    long unsigned int MapObjectsInMap(); // get number of objects.

    long unsigned int GetMaxKFid();
    KeyFrame *GetLatestKF();

    void clear();

    std::vector<KeyFrame *> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate; // used in track(), and Optimizer of localmapping

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    // by me, object map scale
    cv::Mat InitToGround, GroundToInit; // orb's init camera frame to my ground
    Eigen::Matrix4f InitToGround_eigen;
    Eigen::Matrix4d InitToGround_eigen_d, GroundToInit_eigen_d;
    Eigen::Matrix3f Kalib_f, invKalib_f;
    Eigen::Matrix3d Kalib, invKalib;

    int img_width, img_height;

    cv::Mat GroundToInit_opti;
    cv::Mat InitToGround_opti;
    cv::Mat RealGroundToMine_opti;
    cv::Mat MineGroundToReal_opti;

protected:
    std::set<MapPoint *> mspMapPoints;
    std::set<MapObject *> mspMapObjects;
    std::set<KeyFrame *> mspKeyFrames;       // not sequential
    std::vector<KeyFrame *> mspKeyFramesVec; // sequential. only add, no delete. check isbad() before use

    std::vector<MapPoint *> mvpReferenceMapPoints;

    long unsigned int mnMaxKFid;
    KeyFrame *mLatestKF;

    std::mutex mMutexMap;
};

} // namespace ORB_SLAM2

#endif // MAP_H
