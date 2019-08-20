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

#include "Map.h"

#include "MapPoint.h"
#include "KeyFrame.h"
#include "MapObject.h"

#include <mutex>

using namespace std;
namespace ORB_SLAM2
{

Map::Map() : mnMaxKFid(0), mLatestKF(nullptr)
{
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);

    if (pKF->mnId >= mnMaxKFid) // by me >  changes to >=    otherwise first keyframe is not reflected.
    {
        mnMaxKFid = pKF->mnId;
        mLatestKF = pKF;
    }

    if (mspKeyFramesVec.size() > 0)
        if (pKF->mnId == mspKeyFramesVec.back()->mnId)
            return;
    mspKeyFramesVec.push_back(pKF);
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::AddMapObject(MapObject *pMO)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapObjects.insert(pMO);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
    // This only erase the pointer, not deallocate
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF); // called in Keyframe.cc when pKF is set bad... map point usually also removed.
}

void Map::EraseMapObject(MapObject *pMO)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapObjects.erase(pMO);
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

vector<KeyFrame *> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
}

vector<KeyFrame *> Map::GetAllKeyFramesSequential()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFramesVec;
}

vector<MapPoint *> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
}

vector<MapObject *> Map::GetAllMapObjects()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapObject *>(mspMapObjects.begin(), mspMapObjects.end());
}

vector<MapObject *> Map::GetGoodMapObjects()
{
    vector<MapObject *> res;
    for (set<MapObject *>::iterator sit = mspMapObjects.begin(), send = mspMapObjects.end(); sit != send; sit++)
        if ((*sit)->isGood)
            res.push_back(*sit);
    return res;
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

long unsigned int Map::MapObjectsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapObjects.size();
}

vector<MapPoint *> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

KeyFrame *Map::GetLatestKF()
{
    unique_lock<mutex> lock(mMutexMap);
    return mLatestKF;
}

void Map::clear()
{
    for (set<MapPoint *>::iterator sit = mspMapPoints.begin(), send = mspMapPoints.end(); sit != send; sit++)
        delete *sit;

    for (set<MapObject *>::iterator sit = mspMapObjects.begin(), send = mspMapObjects.end(); sit != send; sit++)
        delete *sit;

    for (set<KeyFrame *>::iterator sit = mspKeyFrames.begin(), send = mspKeyFrames.end(); sit != send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspMapObjects.clear();
    mspKeyFrames.clear();
    mspKeyFramesVec.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

} // namespace ORB_SLAM2
