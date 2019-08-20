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
#include "MapObject.h"
#include "KeyFrame.h"
#include "Frame.h"

#include "Optimizer.h"
#include "Converter.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <mutex>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

// by me
#include "tictoc_profiler/profiler.hpp"
#include "ros/ros.h"
#include "Parameters.h"
#include "detect_3d_cuboid/matrix_utils.h"
#include <boost/filesystem.hpp>
#include <boost/graph/graph_concepts.hpp>

using namespace std;
using namespace Eigen;
namespace ORB_SLAM2
{

void Optimizer::GlobalBundleAdjustemnt(Map *pMap, int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
    BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
}

void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                 int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->mnId);
        vSE3->setFixed(pKF->mnId == 0);
        optimizer.addVertex(vSE3);
        if (pKF->mnId > maxKFid)
            maxKFid = pKF->mnId;
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);

    // Set MapPoint vertices
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        MapPoint *pMP = vpMP[i];
        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        const int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();

        int nEdges = 0;
        //SET EDGES
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
        {

            KeyFrame *pKF = mit->first;
            if (pKF->isBad() || pKF->mnId > maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

            if (pKF->mvuRight[mit->second] < 0)
            {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust)
                {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else
            {
                Eigen::Matrix<double, 3, 1> obs;
                const float kp_ur = pKF->mvuRight[mit->second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust)
                {
                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if (nEdges == 0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
        else
        {
            vbNotIncludedMP[i] = false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        if (nLoopKF == 0)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            pKF->mTcwGBA.create(4, 4, CV_32F);
            Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            pKF->mnBAGlobalForKF = nLoopKF;
        }
    }

    //Points
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint *pMP = vpMP[i];

        if (pMP->isBad())
            continue;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

        if (nLoopKF == 0)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            pMP->mPosGBA.create(3, 1, CV_32F);
            Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
            pMP->mnBAGlobalForKF = nLoopKF;
        }
    }
}

int Optimizer::PoseOptimization(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences = 0;

    // Set Frame vertex
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    {
        unique_lock<mutex> lock(MapPoint::mGlobalMutex);

        for (int i = 0; i < N; i++)
        {
            MapPoint *pMP = pFrame->mvpMapPoints[i];
            if (pMP)
            {
                if (whether_dynamic_object && pMP->is_dynamic)
                    continue;

                // Monocular observation
                if (pFrame->mvuRight[i] < 0)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    if (pFrame->KeysStatic.size() > 0 && !pFrame->KeysStatic[i])
                    {
                        ROS_ERROR_STREAM("Found static map point, dynamic 2d feature");
                        continue;
                    }

                    Eigen::Matrix<double, 2, 1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose(); // create many unary edges

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else // Stereo observation
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    //SET EDGE
                    Eigen::Matrix<double, 3, 1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    const float &kp_ur = pFrame->mvuRight[i];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaStereo);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    e->bf = pFrame->mbf;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesStereo.push_back(e);
                    vnIndexEdgeStereo.push_back(i);
                }
            }
        }
    }

    if (nInitialCorrespondences < 3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
    const int its[4] = {10, 10, 10, 10};

    int nBad = 0;
    for (size_t it = 0; it < 4; it++)
    {

        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Mono[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx] = false;
                e->setLevel(0);
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if (pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if (chi2 > chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx] = true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                pFrame->mvbOutlier[idx] = false;
            }

            if (it == 2)
                e->setRobustKernel(0);
        }

        if (optimizer.edges().size() < 10)
            break;
    }

    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);

    pFrame->SetPose(pose);

    return nInitialCorrespondences - nBad;
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
{
    // Local KeyFrames to optimize: First Breath Search from Current Keyframe
    list<KeyFrame *> lLocalKeyFrames; // local KFs which share map points with current frame.

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames(); // directly get local keyframes.
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame *pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames. some points might not be seen by currentKF
    list<MapPoint *> lLocalMapPoints;
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
        for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId) // mnBALocalForKF  mnBAFixedForKF are marker
                    {
                        if (whether_dynamic_object && pMP->is_dynamic)
                            continue;
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame *> lFixedCameras;
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) // not been added to lLocalKeyFrames, and lFixedCameras!
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;

    // Set Local KeyFrame vertices
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == 0); // if firsr keyframe frame, set pose fixed.
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono; // a camera + map point
    vpEdgesMono.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo; // left + right camera + map point
    vpEdgesStereo.reserve(nExpectedSize);

    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    // for each map points, add the observal frame edges.
    int obs_greater_one_points = 0;
    int obs_one_points = 0;

    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;

        if (pMP->Observations() == 1) // by me, skip observation one point, which is just initialized by my depth! won't affect previous mono.
            continue;

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

        int id = pMP->mnId + maxKFid + 1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
        if (observations.size() == 1)
            obs_one_points++;
        if (observations.size() > 1)
            obs_greater_one_points++;

        // Set edges
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                if (pKFi->KeysStatic.size() > 0 && !pKFi->KeysStatic[mit->second])
                {
                    continue;
                }

                // Monocular observation
                if (pKFi->mvuRight[mit->second] < 0)
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ(); // camera point edge.  there is no camera-camera odometry edge.

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId))); //get vertex based on Id.
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore)
    {
        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
        }

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex
    if (parallel_mapping)
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty())
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes
    for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFrame = *lit;
        pKFrame->mnBALocalForKF = 0; // by me reset it
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFrame->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKFrame->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        pMP->mnBALocalForKF = 0;
        if (pMP->Observations() == 1)
            continue;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame *pKFrame = *lit;
        pKFrame->mnBAFixedForKF = 0; // reset them
        pKFrame->mnBALocalForKF = 0;
    }
}

// similar to localBA, add objects
void Optimizer::LocalBACameraPointObjects(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, bool fixCamera, bool fixPoint)
{
    // Local KeyFrames to optimize: First Breath Search from Current Keyframe
    vector<KeyFrame *> lLocalKeyFrames; // local KFs which share map points with current frame.

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames(); // directly get local keyframes.
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame *pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames. some points might not be seen by currentKF
    vector<MapPoint *> lLocalMapPoints;
    for (vector<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
        for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                    if (pMP->mnBALocalForKF != pKF->mnId) // mnBALocalForKF  mnBAFixedForKF are marker
                    {
                        if (whether_dynamic_object)
                            if (pMP->is_dynamic)
                                continue;
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    vector<MapObject *> lLocalMapObjects;
    for (vector<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        vector<MapObject *> vpMOs = (*lit)->cuboids_landmark;
        for (vector<MapObject *>::iterator vit = vpMOs.begin(), vend = vpMOs.end(); vit != vend; vit++)
        {
            MapObject *pMO = *vit;
            if (pMO)
                if (!pMO->isBad())
                    if (pMO->mnBALocalForKF != pKF->mnId) // mnBALocalForKF  mnBAFixedForKF are marker
                    {
                        lLocalMapObjects.push_back(pMO);
                        pMO->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    vector<KeyFrame *> lFixedCameras;
    for (vector<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) // not been added to lLocalKeyFrames, and lFixedCameras!
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // More Fixed Keyframes. Keyframes that see Local MapObjects but that are not Local Keyframes  改变的地方
    for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
    {
        unordered_map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
        for (unordered_map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) // not been added to lLocalKeyFrames, and lFixedCameras!
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    std::cout << "LocalBAPointObjects num of local points / objects / frames:  " << lLocalMapPoints.size() << "  " << lLocalMapObjects.size() << "  " << lLocalKeyFrames.size() << std::endl;

    // estimate object camera edges.
    int estimated_cam_obj_edges = 0;
    for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
    {
        MapObject *pMObject = *lit;
        estimated_cam_obj_edges += pMObject->Observations();
    }

#define ObjectFixScale // use when scale is provided and fixed. such as KITTI

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
#ifdef ObjectFixScale
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
#else
    g2o::BlockSolverX::LinearSolverType *linearSolver; // BlockSolverX instead of BlockSolver63
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
#endif
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;
    // Set Local KeyFrame vertices
    for (vector<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == 0); // if firsr keyframe frame, set pose fixed.
        if (fixCamera)
            vSE3->setFixed(true);
        // 	vSE3->fix_rotation = true;
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for (vector<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

#ifdef ObjectFixScale
    typedef g2o::VertexCuboidFixScale g2o_object_vertex;
    typedef g2o::EdgeSE3CuboidFixScaleProj g2o_camera_obj_2d_edge;
#else
    typedef g2o::VertexCuboid g2o_object_vertex;
    typedef g2o::EdgeSE3CuboidProj g2o_camera_obj_2d_edge;
#endif

    // Set MapObject vertices
    long int maxObjectid = 0;
    for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
    {
        MapObject *pMObject = *lit;
        g2o::cuboid cube_pose = pMObject->GetWorldPos();

        g2o_object_vertex *vObject = new g2o_object_vertex();

#ifdef ObjectFixScale
        if (scene_unique_id == kitti)
            vObject->fixedscale = Eigen::Vector3d(1.9420, 0.8143, 0.7631);
        else
            ROS_ERROR_STREAM("Please see cuboid scale!!!, otherwise use VertexCuboid()");

        // set the roll=M_PI/2, pitch=0, yaw., later did this in tracking.
        // initialize object absolute height?  or set based on camera height?
        if (scene_unique_id == kitti)
        {
            if (!build_worldframe_on_ground)
            {
                float cam_height = pKF->GetCameraCenter().at<float>(1); // in init frame, y downward for KITTI
                                                                        // 	cube_pose.setTranslation(Vector3d(cube_pose.translation()(0),1.7-0.7631,cube_pose.translation()(2)));
                cube_pose.setTranslation(Vector3d(cube_pose.translation()(0), cam_height + 1.0, cube_pose.translation()(2)));
            }
            else
            {
                float cam_height = pKF->GetCameraCenter().at<float>(2);
                cube_pose.setTranslation(Vector3d(cube_pose.translation()(0), cube_pose.translation()(1), cam_height - 1.0));
            }
            if (vObject->fixedscale(0) > 0) // even later didn't actually optimize, change the scale. just for visualization.
                cube_pose.setScale(vObject->fixedscale);
        }
#endif
        vObject->setEstimate(cube_pose);
        vObject->whether_fixrollpitch = true; // only rotate along object z axis.
        vObject->whether_fixheight = false;   //may make camera height estimation bad

        int id = pMObject->mnId + maxKFid + 1;
        vObject->setId(id);
        vObject->setFixed(false);
        optimizer.addVertex(vObject);
        if (pMObject->mnId > maxObjectid)
            maxObjectid = pMObject->mnId;
    }

    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();
    vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono; // a camera + map point
    vpEdgesMono.reserve(nExpectedSize);
    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);
    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo; // left + right camera + map point
    vpEdgesStereo.reserve(nExpectedSize);
    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    // set up map point and camera-point edges
    int obs_greater_one_points = 0;
    int obs_one_points = 0;
    for (vector<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;

        if (pMP->Observations() == 1) // HACK by me, skip observation one point, which is initialized by my depth! won't affect previous mono.
            continue;

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

        int id = pMP->mnId + maxKFid + maxObjectid + 2;
        vPoint->setId(id);
        if (fixPoint)
            vPoint->setFixed(true);
        else
            vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
        if (observations.size() == 1)
            obs_one_points++;
        if (observations.size() > 1)
            obs_greater_one_points++;

        //Set edges
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if (pKFi->mvuRight[mit->second] < 0)
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ(); // camera point edge.  there is no camera-camera odometry edge.

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId))); //get vertex based on Id.
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    // set up point-object 3d association
    if (1)
    {
        // get object point association.can we do this when validate cuboids?? then don't need to do it every time?
        int point_object_threshold = 2;
        vector<vector<Vector3d>> all_object_ba_points(lLocalMapObjects.size());
        for (size_t i = 0; i < lLocalMapObjects.size(); i++)
        {
            MapObject *pMO = lLocalMapObjects[i];
            pMO->point_object_BA_counter++;
            pMO->used_points_in_BA.clear();
            pMO->used_points_in_BA_filtered.clear();

            const std::vector<MapPoint *> &UniquePoints = pMO->GetUniqueMapPoints();
            int largest_point_obs_num = pMO->largest_point_observations;
            point_object_threshold = std::max(int(largest_point_obs_num * 0.4), 2); // whether use adaptive threshold or fixed.
            pMO->pointOwnedThreshold = point_object_threshold;

            for (size_t j = 0; j < UniquePoints.size(); j++)
                if (UniquePoints[j])
                    if (!UniquePoints[j]->isBad())
                        if (UniquePoints[j]->MapObjObservations[pMO] > point_object_threshold)
                        {
                            pMO->used_points_in_BA.push_back(UniquePoints[j]);
                            all_object_ba_points[i].push_back(Converter::toVector3d(UniquePoints[j]->GetWorldPos()));
                        }
        }

        double coarse_threshold = 4;
        double fine_threshold = 3;
        if (scene_unique_id == kitti)
        {
            coarse_threshold = 4;
            fine_threshold = 3;
        }

        for (size_t i = 0; i < lLocalMapObjects.size(); i++)
        {
            MapObject *pMObj = lLocalMapObjects[i];
            // compute the mean, eliminate outlier points.
            Eigen::Vector3d mean_point;
            mean_point.setZero();
            for (size_t j = 0; j < all_object_ba_points[i].size(); j++)
                mean_point += all_object_ba_points[i][j];
            mean_point /= (double)(all_object_ba_points[i].size());
            Eigen::Vector3d mean_point_final;
            mean_point_final.setZero();
            //NOTE  filtering of points!!!  remove outlier points
            Eigen::Vector3d mean_point_2;
            mean_point_2.setZero();
            int valid_point_num = 0;
            for (size_t j = 0; j < all_object_ba_points[i].size(); j++)
                if ((mean_point - all_object_ba_points[i][j]).norm() < coarse_threshold)
                {
                    mean_point_2 += all_object_ba_points[i][j];
                    valid_point_num++;
                }
            mean_point_2 /= (double)valid_point_num;
            std::vector<Eigen::Vector3d> good_points; // for car, if points are 4 meters away from center, usually outlier.
            for (size_t j = 0; j < all_object_ba_points[i].size(); j++)
            {
                if ((mean_point_2 - all_object_ba_points[i][j]).norm() < fine_threshold)
                {
                    mean_point_final += all_object_ba_points[i][j];
                    good_points.push_back(all_object_ba_points[i][j]);
                    pMObj->used_points_in_BA_filtered.push_back(pMObj->used_points_in_BA[j]);
                }
                // else  remove observation.
            }
            mean_point_final /= (double)(good_points.size());
            all_object_ba_points[i].clear();
            all_object_ba_points[i] = good_points;

            if ((all_object_ba_points[i].size() > 5) && 1) // whether want to initialize object position to be center of points
            {
                g2o_object_vertex *vObject = static_cast<g2o_object_vertex *>(optimizer.vertex(pMObj->mnId + maxKFid + 1));
                g2o::cuboid tempcube = vObject->estimate();
                tempcube.setTranslation(mean_point_final);
                vObject->setEstimate(tempcube);
            }
        }

        // point - object 3d measurement. set use fixed point or to optimize point
        for (size_t i = 0; i < lLocalMapObjects.size(); i++) // no need to optimize all objects...., use local KF's map objects?
        {
            MapObject *pMO = lLocalMapObjects[i];

            if (1) // an object connected to many fixed points. optimize only object
            {
#ifdef ObjectFixScale
                g2o::EdgePointCuboidOnlyObjectFixScale *e = new g2o::EdgePointCuboidOnlyObjectFixScale();
#else
                g2o::EdgePointCuboidOnlyObject *e = new g2o::EdgePointCuboidOnlyObject();
#endif
                for (size_t j = 0; j < all_object_ba_points[i].size(); j++)
                    e->object_points.push_back(all_object_ba_points[i][j]);

                if (e->object_points.size() > 10)
                {
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMO->mnId + maxKFid + 1)));
                    Eigen::Matrix3d info;
                    info.setIdentity();
                    e->setInformation(info);
                    e->max_outside_margin_ratio = 1;
                    if (scene_unique_id == kitti)
                    {
                        e->max_outside_margin_ratio = 2;
                        e->prior_object_half_size = Eigen::Vector3d(1.9420, 0.8143, 0.7631);
                    }
                    optimizer.addEdge(e);
                }
            }
            if (0) // each object is connect to one point. optimize both point and object
            {
                if (pMO->used_points_in_BA_filtered.size() > 10)
                    for (size_t j = 0; j < pMO->used_points_in_BA_filtered.size(); j++)
                    {
                        g2o::EdgePointCuboid *e = new g2o::EdgePointCuboid();
                        g2o::OptimizableGraph::Vertex *pointvertex = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMO->used_points_in_BA_filtered[j]->mnId + maxKFid + maxObjectid + 2));
                        if (pointvertex != nullptr)
                        {
                            e->setVertex(0, pointvertex);
                            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMO->mnId + maxKFid + 1)));
                            Eigen::Matrix3d info;
                            info.setIdentity();
                            e->setInformation(info * 10);
                            e->max_outside_margin_ratio = 2;
                            optimizer.addEdge(e);
                        }
                    }
            }
            // 	  ROS_ERROR_STREAM("BA filter size/e objec point size   "<<pMO->used_points_in_BA_filtered.size()<<"   "<<all_object_ba_points[i].size());
        }
    }

    // add camera - object 2d measurement.
    vector<g2o_camera_obj_2d_edge *> vpEdgesCameraObject;
    if (1)
    {
        Eigen::Vector4d inv_sigma;
        inv_sigma.setOnes();
        inv_sigma = inv_sigma * camera_object_BA_weight; // point sigma<1, object error is usually large, no need to set large sigma...

        if (lLocalMapObjects.size() > 5)
            inv_sigma = inv_sigma / 2;

        int object_boundary_margin = 10;
        Eigen::Matrix4d camera_object_sigma = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
        const float thHuberObject = sqrt(900); // object reprojection error is usually large
        int total_left = 0;
        int total_right = 0;
        int total_middle = 0;
        vector<g2o_camera_obj_2d_edge *> vpEdgesCameraObjectLeft;
        vector<g2o_camera_obj_2d_edge *> vpEdgesCameraObjectRight;
        vector<g2o_camera_obj_2d_edge *> vpEdgesCameraObjectMiddle;
        bool whether_want_camera_obj = true;
        if (whether_want_camera_obj)
            for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
            {
                MapObject *pMObject = *lit;

                g2o_camera_obj_2d_edge *obj_edge;
                int obj_obs_num = 0;

                const unordered_map<KeyFrame *, size_t> observations = pMObject->GetObservations();
                for (unordered_map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                {
                    KeyFrame *pKFi = mit->first;

                    if (!pKFi->isBad())
                    {
                        const MapObject *local_object = pKFi->local_cuboids[mit->second];

                        g2o_camera_obj_2d_edge *e = new g2o_camera_obj_2d_edge();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));                   // camera
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMObject->mnId + maxKFid + 1))); // object

                        e->setMeasurement(local_object->bbox_vec);

                        cv::Rect bbox_2d = local_object->bbox_2d;
                        // object should be in FOV, otherwise bad for this edge
                        if ((bbox_2d.x > object_boundary_margin) && (bbox_2d.y > object_boundary_margin) && (bbox_2d.x + bbox_2d.width < pMap->img_width - object_boundary_margin) &&
                            (bbox_2d.y + bbox_2d.height < pMap->img_height - object_boundary_margin))
                        {
                            e->Kalib = pMap->Kalib;
                            e->setInformation(camera_object_sigma * pMObject->meas_quality * pMObject->meas_quality);

                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberObject);

                            optimizer.addEdge(e);
                            vpEdgesCameraObject.push_back(e);

                            obj_edge = e;
                            obj_obs_num++;

                            if (scene_unique_id == kitti)
                            {
                                if (local_object->left_right_to_car == 1)
                                {
                                    vpEdgesCameraObjectLeft.push_back(e);
                                    total_left += 1;
                                }
                                if (local_object->left_right_to_car == 2)
                                {
                                    vpEdgesCameraObjectRight.push_back(e);
                                    total_right += 1;
                                }
                                if (local_object->left_right_to_car == 0)
                                {
                                    vpEdgesCameraObjectMiddle.push_back(e);
                                    total_middle += 1;
                                }
                            }
                        }
                    }
                }
                if (obj_obs_num == 1) // if an object is only connected to one camera, don't optimize it! no need
                {
                    obj_edge->setLevel(1);
                }
            }

        if (scene_unique_id == kitti) // balance left-right cars.
        {
            if (total_left > 2 * (total_right + total_middle))
            {
                for (size_t i = 0; i < vpEdgesCameraObjectLeft.size(); i++)
                    vpEdgesCameraObjectLeft[i]->setInformation(vpEdgesCameraObjectLeft[i]->information() / 2.0);
            }
            if (total_right > 2 * (total_left + total_middle))
            {
                for (size_t i = 0; i < vpEdgesCameraObjectRight.size(); i++)
                    vpEdgesCameraObjectRight[i]->setInformation(vpEdgesCameraObjectRight[i]->information() / 2.0);
            }
        }
    }

    std::cout << "BA edges  point-cam  object-cam  " << vpMapPointEdgeMono.size() + vpMapPointEdgeStereo.size() << "  " << vpEdgesCameraObject.size() << std::endl;

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore)
    {
        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                e->setLevel(1); // don't optimize this edge.
            }

            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesCameraObject.size(); i < iend; i++)
        {
            g2o_camera_obj_2d_edge *e = vpEdgesCameraObject[i];
            if (e->error().norm() > 80)
            {
                e->setLevel(1);
            }
        }
        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex
    if (parallel_mapping)
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty())
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    //Keyframes
    for (vector<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFrame = *lit;
        pKFrame->mnBALocalForKF = 0;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFrame->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKFrame->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for (vector<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        pMP->mnBALocalForKF = 0;
        if (pMP->Observations() == 1)
            continue;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + maxObjectid + 2));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }

    for (vector<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame *pKFrame = *lit;
        pKFrame->mnBAFixedForKF = 0;
        pKFrame->mnBALocalForKF = 0;
    }

    // Objects
    for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
    {
        MapObject *pMObject = *lit;
        pMObject->mnBALocalForKF = 0;
        pMObject->obj_been_optimized = true;
        g2o_object_vertex *vObject = static_cast<g2o_object_vertex *>(optimizer.vertex(pMObject->mnId + maxKFid + 1));
        pMObject->SetWorldPos(vObject->estimate());
    }
}

// similar to localBA, add objects
void Optimizer::LocalBACameraPointObjectsDynamic(KeyFrame *pKF, bool *pbStopFlag, Map *pMap, bool fixCamera, bool fixPoint)
{
    // Local KeyFrames to optimize: First Breath Search from Current Keyframe
    vector<KeyFrame *> lLocalKeyFrames; // local KFs which share map points with current frame.

    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;

    const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames(); // directly get local keyframes.
    for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
    {
        KeyFrame *pKFi = vNeighKFs[i];
        pKFi->mnBALocalForKF = pKF->mnId;
        if (!pKFi->isBad())
            lLocalKeyFrames.push_back(pKFi);
    }

    // Local MapPoints seen in Local KeyFrames. some points might not be seen by currentKF
    vector<MapPoint *> lLocalMapPoints;
    int deleteoldsinglepts = 0;
    for (vector<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
        for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
        {
            MapPoint *pMP = *vit;
            if (pMP)
                if (!pMP->isBad())
                {
                    // NOTE delete old mappoint which is only observed once
                    if ((*lit) != pKF && pMP->is_dynamic && pMP->Observations() == 1)
                    {
                        pMP->SetBadFlag();
                        deleteoldsinglepts++;
                    }
                    if (pMP->mnBALocalForKF != pKF->mnId) // mnBALocalForKF  mnBAFixedForKF are marker
                    {
                        // 			if (pMP->is_dynamic)
                        // 			    continue;
                        lLocalMapPoints.push_back(pMP);
                        pMP->mnBALocalForKF = pKF->mnId;
                    }
                }
        }
    }
    int deleteoldsinglepts2 = 0;
    if (use_dynamic_klt_features)
    {
        for (vector<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            vector<MapPoint *> vpMPs = (*lit)->GetHarrisMapPointMatches();
            for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                MapPoint *pMP = *vit;
                if (pMP)
                    if (!pMP->isBad())
                    {
                        // NOTE delete old mappoint which is only observed once.
                        if ((*lit) != pKF && pMP->is_dynamic && pMP->Observations() == 1)
                        {
                            pMP->SetBadFlag();
                            deleteoldsinglepts2++;
                        }
                        if (pMP->mnBALocalForKF != pKF->mnId) // mnBALocalForKF  mnBAFixedForKF are marker
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
                    }
            }
        }
    }
    cout << "delete old single map points   " << deleteoldsinglepts << "   " << deleteoldsinglepts2 << endl;

    vector<MapObject *> lLocalMapObjects;
    for (vector<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        vector<MapObject *> vpMOs = (*lit)->cuboids_landmark;
        for (vector<MapObject *>::iterator vit = vpMOs.begin(), vend = vpMOs.end(); vit != vend; vit++)
        {
            MapObject *pMO = *vit;
            if (pMO)
                if (!pMO->isBad())
                    if (pMO->mnBALocalForKF != pKF->mnId) // mnBALocalForKF  mnBAFixedForKF are marker
                    {
                        lLocalMapObjects.push_back(pMO);
                        pMO->mnBALocalForKF = pKF->mnId;
                    }
        }
    }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    vector<KeyFrame *> lFixedCameras;
    for (vector<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;
            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) // not been added to lLocalKeyFrames, and lFixedCameras!
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    // More Fixed Keyframes. Keyframes that see Local MapObjects but that are not Local Keyframes  改变的地方
    for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
    {
        unordered_map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
        for (unordered_map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            // NOTE if frame too old, skip it. dynamic objects can be observed by many frames!!!
            if ((pKFi->mTimeStamp - pKF->mTimeStamp) > 8.0 && pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) // not been added to lLocalKeyFrames, and lFixedCameras!
            {
                pKFi->mnBAFixedForKF = pKF->mnId;
                if (!pKFi->isBad())
                    lFixedCameras.push_back(pKFi);
            }
        }
    }

    std::cout << "LocalBAPointObjects num of local points / objects / frames:  " << lLocalMapPoints.size() << "  " << lLocalMapObjects.size() << "  " << lLocalKeyFrames.size() << std::endl;

#define ObjectFixScale

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
#ifdef ObjectFixScale
    g2o::BlockSolverX::LinearSolverType *linearSolver;                              //HACK BlockSolverX  BlockSolver_6_3
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>(); // LinearSolverEigen   LinearSolverDense
    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
#else
    g2o::BlockSolverX::LinearSolverType *linearSolver; // BlockSolverX instead of BlockSolver63
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
#endif
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    unsigned long maxKFid = 0;
    // Set Local KeyFrame vertices
    for (vector<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId == 0);
        if (fixCamera)
            vSE3->setFixed(true);
        // 	vSE3->fix_rotation = true;
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for (vector<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame *pKFi = *lit;
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if (pKFi->mnId > maxKFid)
            maxKFid = pKFi->mnId;
    }

#ifdef ObjectFixScale
    typedef g2o::VertexCuboidFixScale g2o_object_vertex;
    typedef g2o::EdgeSE3CuboidFixScaleProj g2o_camera_obj_2d_edge;
#else
    typedef g2o::VertexCuboid g2o_object_vertex;
    typedef g2o::EdgeSE3CuboidProj g2o_camera_obj_2d_edge;
#endif

    // Set MapObject vertices
    long int maxObjectid = 0;
    Eigen::Vector3d objfixscale = Eigen::Vector3d(1.9420, 0.8143, 0.7631);
    long int maxIdTillObject = ++maxKFid;
    for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
    {
        MapObject *pMObject = *lit;
        pMObject->bundle_vertex_ids.clear();

        // for a dynamic object, create each vertex for each observed frame
        unordered_map<KeyFrame *, size_t> observations = pMObject->GetObservations();
        for (unordered_map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;
            if (pKFi->isBad())
                continue;

            if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId) //not used in BA
                continue;

            g2o::cuboid cube_pose;
            if (pMObject->allDynamicPoses.count(pKFi))
                cube_pose = pMObject->allDynamicPoses[pKFi].first;
            else
            {
                ROS_ERROR_STREAM("BA not found frame object pose!!!   "
                                 << "  objID  " << pMObject->mnId << "  frameid  " << pKFi->mnFrameId);
                exit(0); // should not happen
            }

            g2o_object_vertex *vObject = new g2o_object_vertex();

#ifdef ObjectFixScale
            if (scene_unique_id == kitti)
                vObject->fixedscale = Eigen::Vector3d(1.9420, 0.8143, 0.7631); // for kitti object, scale may don't need to set...
            else
                ROS_ERROR_STREAM("Please see cuboid scale!!!, otherwise use VertexCuboid()");

            if (scene_unique_id == kitti)
            {
                if (!build_worldframe_on_ground)
                {
                    float cam_height = pKFi->GetCameraCenter().at<float>(1); // in init frame, y downward for KITTI
                                                                             // 		  cube_pose.setTranslation(Vector3d(cube_pose.translation()(0),1.7-0.7631,cube_pose.translation()(2)));
                    cube_pose.setTranslation(Vector3d(cube_pose.translation()(0), cam_height + 1.0, cube_pose.translation()(2)));
                }
                else
                {
                    float cam_height = pKFi->GetCameraCenter().at<float>(2); // in init frame, y downward for KITTI
                }
                if (vObject->fixedscale(0) > 0) // even later didn't actually optimize, change the scale. just for visualization.
                    cube_pose.setScale(vObject->fixedscale);
            }
#endif

            vObject->setEstimate(cube_pose);
            vObject->whether_fixrotation = true;
            vObject->whether_fixheight = false;
            maxIdTillObject++;
            vObject->setId(maxIdTillObject);
            vObject->setFixed(false);
            pMObject->bundle_vertex_ids[pKFi] = maxIdTillObject;
            optimizer.addVertex(vObject);
        }
    }

    const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();
    vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono; // a camera + map point
    vpEdgesMono.reserve(nExpectedSize);
    vector<KeyFrame *> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);
    vector<MapPoint *> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);
    vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo; // left + right camera + map point
    vpEdgesStereo.reserve(nExpectedSize);
    vector<KeyFrame *> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);
    vector<MapPoint *> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    // set up map point and camera-point edges
    int obs_greater_one_points = 0;
    int obs_one_points = 0;
    int maxIdTillPoint = maxIdTillObject;
    for (vector<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;

        if (pMP->Observations() == 1) // HACK by me, skip observation one point, which is initialized by my depth! won't affect previous mono.
            continue;

        if (pMP->is_dynamic) // for dynamic point, treat separatedly, use object-point-camera edge.
            continue;

        g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

        int id = pMP->mnId + maxIdTillObject + 1;
        if (id > maxIdTillPoint)
            maxIdTillPoint = id;
        vPoint->setId(id);
        if (fixPoint)
            vPoint->setFixed(true);
        else
            vPoint->setMarginalized(true); // must be true, otherwise run error. Not sure why. from g2o paper, marginalize is just one way to solve...
        optimizer.addVertex(vPoint);

        const map<KeyFrame *, size_t> observations = pMP->GetObservations();
        if (observations.size() == 1)
            obs_one_points++;
        if (observations.size() > 1)
            obs_greater_one_points++;

        //Set edge to camera
        for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;

            if (!pKFi->isBad())
            {
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if (pKFi->mvuRight[mit->second] < 0)
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ(); // camera point edge.  there is no camera-camera odometry edge.

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId))); //get vertex based on Id.
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    vector<g2o::EdgeDynamicPointCuboidCamera *> vpEdgesCameraPointObject;
    vpEdgesCameraPointObject.reserve(nExpectedSize); // a camera + map point + object
    vector<g2o::UnaryLocalPoint *> vpEdgesPointObject;
    vpEdgesPointObject.reserve(nExpectedSize); // map point + object
    vector<MapPoint *> vpMapPointEdgeCamPtObj;
    vpMapPointEdgeCamPtObj.reserve(nExpectedSize);
    vector<KeyFrame *> vpKeyframeEdgeCamPtObj;
    vpKeyframeEdgeCamPtObj.reserve(nExpectedSize);

    // set dynamic point-object-camera edges.
    if (ba_dyna_pt_obj_cam)
    {
        for (vector<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            MapPoint *pMP = *lit;
            if (!pMP->is_dynamic) // static point already processed before.
                continue;

            if (pMP->Observations() < 4) //3 should have more frame observations... otherwise object depth inited map point is not accurate.
                continue;

            MapObject *belongedobj = pMP->GetBelongedObject();
            if (!belongedobj || belongedobj->mnBALocalForKF != pKF->mnId)
                continue;

            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->PosToObj)); // dynamic point use point-object pose

            int id = pMP->mnId + maxIdTillObject + 1;
            vPoint->setId(id);
            if (id > maxIdTillPoint)
                maxIdTillPoint = id;
            if (fixPoint)
                vPoint->setFixed(true);
            else
                vPoint->setMarginalized(true); // must be true, otherwise run error. Not sure why. from g2o paper, marginalize is just one way to solve...
            optimizer.addVertex(vPoint);

            g2o::UnaryLocalPoint *e2 = new g2o::UnaryLocalPoint();
            e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id))); // local point
            Eigen::Matrix3d info;
            info.setIdentity();
            e2->setInformation(info * 10);
            e2->objectscale = objfixscale;
            e2->max_outside_margin_ratio = 2;
            optimizer.addEdge(e2);
            vpEdgesPointObject.push_back(e2);

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            //Set edge to camera
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;
                if (!belongedobj->bundle_vertex_ids.count(pKFi))
                    continue;

                if (!pKFi->isBad())
                {
                    cv::KeyPoint kpUn;
                    if (use_dynamic_klt_features)
                        kpUn = pKFi->mvKeysHarris[mit->second];
                    else
                        kpUn = pKFi->mvKeysUn[mit->second];

                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    // point-object-camera
                    g2o::EdgeDynamicPointCuboidCamera *e = new g2o::EdgeDynamicPointCuboidCamera();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));                           //camera
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(belongedobj->bundle_vertex_ids[pKFi]))); //object
                    e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));                                   // point

                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave] * 1.0;
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->Kalib = pMap->Kalib;

                    optimizer.addEdge(e);

                    vpEdgesCameraPointObject.push_back(e);

                    vpMapPointEdgeCamPtObj.push_back(pMP);
                    vpKeyframeEdgeCamPtObj.push_back(pKFi);
                }
            }
        }
    }

    cout << "BA vpEdgesCameraPointObject size   " << vpEdgesCameraPointObject.size() << endl;

    // set up point-object 3d association
    if (1)
    {
        // get object point association.can we do this when validate cuboids?? then don't need to do it every time?
        int point_object_threshold = 2;
        vector<vector<Vector3d>> all_object_ba_points(lLocalMapObjects.size());
        for (size_t i = 0; i < lLocalMapObjects.size(); i++)
        {
            MapObject *pMO = lLocalMapObjects[i];
            pMO->point_object_BA_counter++;
            pMO->used_points_in_BA.clear();
            pMO->used_points_in_BA_filtered.clear();

            const std::vector<MapPoint *> &UniquePoints = pMO->GetUniqueMapPoints();
            int largest_point_obs_num = pMO->largest_point_observations;
            point_object_threshold = std::max(int(largest_point_obs_num * 0.4), 2); // whether use adaptive threshold or fixed.
            pMO->pointOwnedThreshold = point_object_threshold;

            for (size_t j = 0; j < UniquePoints.size(); j++)
                if (UniquePoints[j])
                    if (!UniquePoints[j]->isBad())
                        if (UniquePoints[j]->MapObjObservations[pMO] > point_object_threshold)
                        {
                            pMO->used_points_in_BA.push_back(UniquePoints[j]);
                            all_object_ba_points[i].push_back(Converter::toVector3d(UniquePoints[j]->GetWorldPos()));
                        }
        }

        double coarse_threshold = 1.5;
        double fine_threshold = 0.8;
        if (scene_unique_id == kitti)
        {
            coarse_threshold = 4;
            fine_threshold = 3;
        }

        for (size_t i = 0; i < lLocalMapObjects.size(); i++)
        {
            MapObject *pMObj = lLocalMapObjects[i];
            // compute the mean, eliminate outlier points.
            Eigen::Vector3d mean_point;
            mean_point.setZero();
            for (size_t j = 0; j < all_object_ba_points[i].size(); j++)
                mean_point += all_object_ba_points[i][j];
            mean_point /= (double)(all_object_ba_points[i].size());
            Eigen::Vector3d mean_point_final;
            mean_point_final.setZero();
            //NOTE  filtering of points!!!
            Eigen::Vector3d mean_point_2;
            mean_point_2.setZero();
            int valid_point_num = 0;
            for (size_t j = 0; j < all_object_ba_points[i].size(); j++)
                if ((mean_point - all_object_ba_points[i][j]).norm() < coarse_threshold)
                {
                    mean_point_2 += all_object_ba_points[i][j];
                    valid_point_num++;
                }
            mean_point_2 /= (double)valid_point_num;
            std::vector<Eigen::Vector3d> good_points; // for car, if points are 4 meters away from center, usually outlier.
            for (size_t j = 0; j < all_object_ba_points[i].size(); j++)
            {
                if ((mean_point_2 - all_object_ba_points[i][j]).norm() < fine_threshold)
                {
                    mean_point_final += all_object_ba_points[i][j];
                    good_points.push_back(all_object_ba_points[i][j]);
                    pMObj->used_points_in_BA_filtered.push_back(pMObj->used_points_in_BA[j]);
                }
                // else  remove observation.
            }
            mean_point_final /= (double)(good_points.size());
            all_object_ba_points[i].clear();
            all_object_ba_points[i] = good_points;

            if ((all_object_ba_points[i].size() > 5) && 1) // whether want to initialize object position to be center of points
            {
                g2o_object_vertex *vObject = static_cast<g2o_object_vertex *>(optimizer.vertex(pMObj->mnId + maxKFid + 1));
                g2o::cuboid tempcube = vObject->estimate();
                tempcube.setTranslation(mean_point_final);
                vObject->setEstimate(tempcube);
            }
        }

        // point - object 3d measurement. set use fixed point or to optimize point
        for (size_t i = 0; i < lLocalMapObjects.size(); i++) // no need to optimize all objects...., use local KF's map objects?
        {
            MapObject *pMO = lLocalMapObjects[i];

            if (1) // an object connected to many fixed points. optimize only object
            {
#ifdef ObjectFixScale
                g2o::EdgePointCuboidOnlyObjectFixScale *e = new g2o::EdgePointCuboidOnlyObjectFixScale();
#else
                g2o::EdgePointCuboidOnlyObject *e = new g2o::EdgePointCuboidOnlyObject();
#endif
                for (size_t j = 0; j < all_object_ba_points[i].size(); j++)
                    e->object_points.push_back(all_object_ba_points[i][j]);

                if (e->object_points.size() > 10)
                {
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMO->mnId + maxKFid + 1)));
                    Eigen::Matrix3d info;
                    info.setIdentity();
                    e->setInformation(info);
                    e->max_outside_margin_ratio = 1;
                    if (scene_unique_id == kitti)
                    {
                        e->max_outside_margin_ratio = 2;
                        e->prior_object_half_size = Eigen::Vector3d(1.9420, 0.8143, 0.7631);
                    }
                    optimizer.addEdge(e);
                }
            }

            if (0) // each object is connect to one point. optimize both point and object
            {
                if (pMO->used_points_in_BA_filtered.size() > 10)
                    for (size_t j = 0; j < pMO->used_points_in_BA_filtered.size(); j++)
                    {
                        g2o::EdgePointCuboid *e = new g2o::EdgePointCuboid();
                        g2o::OptimizableGraph::Vertex *pointvertex = dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMO->used_points_in_BA_filtered[j]->mnId + maxKFid + maxObjectid + 2));
                        if (pointvertex != nullptr)
                        {
                            e->setVertex(0, pointvertex);
                            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMO->mnId + maxKFid + 1)));
                            Eigen::Matrix3d info;
                            info.setIdentity();
                            e->setInformation(info * 10);
                            e->max_outside_margin_ratio = 2;
                            optimizer.addEdge(e);
                        }
                    }
            }
            // 	  ROS_ERROR_STREAM("BA filter size/e objec point size   "<<pMO->used_points_in_BA_filtered.size()<<"   "<<all_object_ba_points[i].size());
        }
    }

    // set up object-movement velocity constraints.
    vector<g2o::EdgeObjectMotion *> allmotionedges;
    if (ba_dyna_obj_velo)
    {
        Eigen::Vector3d inv_sigma;
        inv_sigma.setOnes();
        inv_sigma(2) = inv_sigma(2) * 5.0; //0.1 angle error is usually much smaller compared to position.
        inv_sigma = inv_sigma * object_velocity_BA_weight;
        Eigen::Matrix3d object_velocity_sigma = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
        const float thHuberObjectVeloc = sqrt(4);
        for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
        {
            MapObject *pMObject = *lit;

            if (pMObject->bundle_vertex_ids.size() < 4) //5 if only with few frames, don't optimize velocity. not accurate. meaningful for at least 3 objs
                continue;

            int id = pMObject->mnId + maxIdTillPoint + 1;
            g2o::VelocityPlanarVelocity *vVelocity = new g2o::VelocityPlanarVelocity();
            vVelocity->setEstimate(pMObject->velocityPlanar); // 0 very initially

            vVelocity->setId(id);
            vVelocity->setFixed(false);
            optimizer.addVertex(vVelocity);

            int valid_observed_frames = 0;

            //for a dynamic object, create each vertex for each observed frame
            vector<KeyFrame *> objkfs = pMObject->GetObserveFramesSequential();
            KeyFrame *firstgoodobsframe = NULL;
            KeyFrame *lastgoodobsframe = NULL;
            KeyFrame *prevgoodobsframe = NULL;
            for (size_t i = 0; i < objkfs.size(); i++)
            {
                KeyFrame *obskf = objkfs[i];
                if (!obskf->isBad() && pMObject->bundle_vertex_ids.count(obskf))
                {
                    if ((pKF->mTimeStamp - obskf->mTimeStamp) > 5.0) //HACK just use recent 10s frames.
                    {
                        continue;
                    }

                    valid_observed_frames++;
                    if (prevgoodobsframe == NULL)
                    {
                        prevgoodobsframe = obskf;
                        firstgoodobsframe = obskf;
                    }
                    else // add velocity edge
                    {
                        g2o::EdgeObjectMotion *e = new g2o::EdgeObjectMotion();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMObject->bundle_vertex_ids[prevgoodobsframe])));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMObject->bundle_vertex_ids[obskf])));
                        e->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vVelocity));
                        e->delta_t = obskf->mTimeStamp - prevgoodobsframe->mTimeStamp;
                        // 		  cout<<"add obj id for frames:   "<<pMObject->mnId<<"   "<<prevgoodobsframe->mnId<<"   "<<obskf->mnId<<"    delta "<<e->delta_t <<endl;
                        // 		  cout<<"obskf->mTimeStamp   "<<obskf->mTimeStamp<<"   "<<prevgoodobsframe->mTimeStamp<<endl;

                        e->setInformation(object_velocity_sigma);

                        optimizer.addEdge(e);

                        prevgoodobsframe = obskf;
                        lastgoodobsframe = obskf;
                        allmotionedges.push_back(e);
                    }
                }
            }

            //velocity Initialized
            if (pMObject->velocityPlanar(0) == 0 && pMObject->velocityPlanar(1) == 0 && firstgoodobsframe && lastgoodobsframe)
            {
                // compute linear velocity, angular velocity
                double time_range = lastgoodobsframe->mTimeStamp - firstgoodobsframe->mTimeStamp;
                g2o::cuboid init_pose = pMObject->allDynamicPoses[firstgoodobsframe].first;
                g2o::cuboid last_pose = pMObject->allDynamicPoses[lastgoodobsframe].first;
                double linear_esti = (last_pose.pose.translation() - init_pose.pose.translation()).norm() / time_range;
                double yaw_from = init_pose.pose.toXYZPRYVector()(5);
                double yaw_to = last_pose.pose.toXYZPRYVector()(5);
                double angle_error = g2o::normalize_theta(yaw_to - yaw_from);
                pMObject->velocityPlanar = Vector2d(linear_esti, 0);
                // 	     ROS_ERROR_STREAM("pMObject->velocityPlanar  estimate  "<<pMObject->velocityPlanar.transpose());
                vVelocity->setEstimate(pMObject->velocityPlanar);
            }
            // 	  ROS_ERROR_STREAM("object observed frames   "<<pMObject->mnId<<"   "<<valid_observed_frames);
        }
    }

    // add camera - object 2d measurement.
    vector<g2o_camera_obj_2d_edge *> vpEdgesCameraObject;
    if (ba_dyna_obj_cam) // for kitti, this works better as scale is given
    {
        // estimate object camera edges.
        int estimated_cam_obj_edges = 0;
        for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
        {
            MapObject *pMObject = *lit;
            estimated_cam_obj_edges += pMObject->Observations();
        }

        Eigen::Vector4d inv_sigma;
        inv_sigma.setOnes();
        inv_sigma = inv_sigma * camera_object_BA_weight; // point sigma<1, object error is usually large, no need to set large sigma...

        int object_boundary_margin = 10;
        Eigen::Matrix4d camera_object_sigma = inv_sigma.cwiseProduct(inv_sigma).asDiagonal();
        const float thHuberObject = sqrt(900); //900 object reprojection error is usually large
        int total_left = 0;
        int total_right = 0;
        int total_middle = 0;
        vector<g2o_camera_obj_2d_edge *> vpEdgesCameraObjectLeft;
        vector<g2o_camera_obj_2d_edge *> vpEdgesCameraObjectRight;
        vector<g2o_camera_obj_2d_edge *> vpEdgesCameraObjectMiddle;
        bool whether_want_camera_obj = true;
        if (whether_want_camera_obj)
            for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
            {
                MapObject *pMObject = *lit;

                g2o_camera_obj_2d_edge *obj_edge;
                int obj_obs_num = 0;

                const unordered_map<KeyFrame *, size_t> observations = pMObject->GetObservations();
                for (unordered_map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                {
                    KeyFrame *pKFi = mit->first;
                    if (!pMObject->bundle_vertex_ids.count(pKFi))
                        continue;

                    if (ba_dyna_obj_velo && (pKF->mTimeStamp - pKFi->mTimeStamp) > 5.0)
                    {
                        continue;
                    }

                    if (!pKFi->isBad())
                    {
                        const MapObject *local_object = pKFi->local_cuboids[mit->second];

                        g2o_camera_obj_2d_edge *e = new g2o_camera_obj_2d_edge();
                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));                        // camera
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pMObject->bundle_vertex_ids[pKFi]))); // object

                        e->setMeasurement(local_object->bbox_vec);

                        cv::Rect bbox_2d = local_object->bbox_2d;
                        // object should be in FOV, otherwise bad for this edge
                        if ((bbox_2d.x > object_boundary_margin) && (bbox_2d.y > object_boundary_margin) && (bbox_2d.x + bbox_2d.width < pMap->img_width - object_boundary_margin) &&
                            (bbox_2d.y + bbox_2d.height < pMap->img_height - object_boundary_margin))
                        {
                            e->Kalib = pMap->Kalib;
                            e->setInformation(camera_object_sigma * pMObject->meas_quality * pMObject->meas_quality);

                            g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(thHuberObject);

                            optimizer.addEdge(e);
                            vpEdgesCameraObject.push_back(e);

                            obj_edge = e;
                            obj_obs_num++;

                            if (scene_unique_id == kitti)
                            {
                                if (local_object->left_right_to_car == 1)
                                {
                                    vpEdgesCameraObjectLeft.push_back(e);
                                    total_left += 1;
                                }
                                if (local_object->left_right_to_car == 2)
                                {
                                    vpEdgesCameraObjectRight.push_back(e);
                                    total_right += 1;
                                }
                                if (local_object->left_right_to_car == 0)
                                {
                                    vpEdgesCameraObjectMiddle.push_back(e);
                                    total_middle += 1;
                                }
                            }
                        }
                    }
                }
                if (obj_obs_num == 1) // if an object is only connected to one camera, don't optimize it! no need
                {
                    obj_edge->setLevel(1);
                }
            }

        if (scene_unique_id == kitti)
        {
            if (total_left > 2 * (total_right + total_middle))
            {
                for (size_t i = 0; i < vpEdgesCameraObjectLeft.size(); i++)
                    vpEdgesCameraObjectLeft[i]->setInformation(vpEdgesCameraObjectLeft[i]->information() / 2.0);
            }
            if (total_right > 2 * (total_left + total_middle))
            {
                for (size_t i = 0; i < vpEdgesCameraObjectRight.size(); i++)
                    vpEdgesCameraObjectRight[i]->setInformation(vpEdgesCameraObjectRight[i]->information() / 2.0);
            }
        }
    }

    std::cout << "BA edges  point-cam: " << vpMapPointEdgeMono.size() + vpMapPointEdgeStereo.size() << "   object-cam: " << vpEdgesCameraObject.size()
              << "   pt-obj-cam: " << vpEdgesCameraPointObject.size() << "   pt-obj: " << vpEdgesPointObject.size() << "   obj-vel: " << allmotionedges.size() << std::endl;

    if (pbStopFlag)
        if (*pbStopFlag)
            return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore = true;

    if (pbStopFlag)
        if (*pbStopFlag)
            bDoMore = false;

    if (bDoMore)
    {
        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                e->setLevel(1); // don't optimize this edge.
            }
            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }
            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesCameraPointObject.size(); i < iend; i++)
        {
            g2o::EdgeDynamicPointCuboidCamera *e = vpEdgesCameraPointObject[i];
            if (e->chi2() > 8)
            {
                e->setLevel(1); // don't optimize this edge.
            }
            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesCameraObject.size(); i < iend; i++)
        {
            g2o_camera_obj_2d_edge *e = vpEdgesCameraObject[i];
            if (e->error().norm() > 80)
            {
                e->setLevel(1);
            }
        }
        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    vector<pair<KeyFrame *, MapPoint *>> vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
    {
        g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
        MapPoint *pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
        MapPoint *pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())
            continue;

        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame *pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi, pMP));
        }
    }

    // Get Map Mutex
    if (parallel_mapping)
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if (!vToErase.empty()) //remove observations
    {
        for (size_t i = 0; i < vToErase.size(); i++)
        {
            KeyFrame *pKFi = vToErase[i].first;
            MapPoint *pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data
    //Keyframes
    for (vector<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
    {
        KeyFrame *pKFrame = *lit;
        pKFrame->mnBALocalForKF = 0;
        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKFrame->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKFrame->SetPose(Converter::toCvMat(SE3quat));
    }

    //Points
    for (vector<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
    {
        MapPoint *pMP = *lit;
        pMP->mnBALocalForKF = 0;
        if (pMP->Observations() == 1)
            continue;
        if (pMP->is_dynamic)
            continue;
        g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxIdTillObject + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
    //fixed frames.
    for (vector<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
    {
        KeyFrame *pKFrame = *lit;
        pKFrame->mnBAFixedForKF = 0;
        pKFrame->mnBALocalForKF = 0;
    }

    // Objects
    for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
    {
        MapObject *pMObject = *lit;
        // 	ROS_ERROR_STREAM("Object ID   "<<pMObject->mnId<<"   obsNum   "<<pMObject->Observations());
        if (!whether_dynamic_object)
            pMObject->mnBALocalForKF = 0;
        pMObject->obj_been_optimized = true;

        int valid_observed_frames = 0;
        // 	pMObject->allDynamicPoses.clear();
        Vector6d velocity;
        const unordered_map<KeyFrame *, size_t> observations = pMObject->GetObservations();
        KeyFrame *latest_observed_goodframe = nullptr;
        for (unordered_map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;
            if (!pKFi->isBad() && pMObject->bundle_vertex_ids.count(pKFi))
            {
                valid_observed_frames++;
                g2o_object_vertex *vObject = dynamic_cast<g2o_object_vertex *>(optimizer.vertex(pMObject->bundle_vertex_ids[pKFi]));
                pMObject->allDynamicPoses[pKFi] = make_pair(vObject->estimate(), true);
                if (latest_observed_goodframe == nullptr || (pKFi->mnId > latest_observed_goodframe->mnId))
                    latest_observed_goodframe = pKFi;
            }
        }
        pMObject->pose_Twc_latestKF = pMObject->allDynamicPoses[latest_observed_goodframe].first;
        pMObject->SetWorldPos(pMObject->pose_Twc_latestKF);
        pMObject->pose_Twc_afterba = pMObject->pose_Twc_latestKF;

        int velocity_id = pMObject->mnId + maxIdTillPoint + 1;
        g2o::VelocityPlanarVelocity *vVelocity = dynamic_cast<g2o::VelocityPlanarVelocity *>(optimizer.vertex(velocity_id));
        if (vVelocity)
        {
            pMObject->velocityPlanar = vVelocity->estimate();
            pMObject->velocityhistory[pKF] = pMObject->velocityPlanar;
        }
    }

    // dynamic point
    if (ba_dyna_pt_obj_cam)
    {
        for (vector<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            MapPoint *pMP = *lit;
            pMP->mnBALocalForKF = 0;
            if (pMP->Observations() == 1)
                continue;
            if (!pMP->is_dynamic)
                continue;

            // 	    cout<<"dynamic point pose before BA    "<<pMP->PosToObj<<endl;
            g2o::VertexSBAPointXYZ *vPoint = dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxIdTillObject + 1));
            if (!vPoint)
                continue;

            pMP->PosToObj = Converter::toCvMat(vPoint->estimate());

            MapObject *belongedobj = pMP->GetBelongedObject();
            if (!belongedobj || belongedobj->mnBALocalForKF != pKF->mnId)
                continue;

            pMP->mWorldPos_latestKF = Converter::toCvMat(belongedobj->pose_Twc_latestKF.pose * (vPoint->estimate()));
            pMP->SetWorldPos(pMP->mWorldPos_latestKF);
            pMP->is_optimized = true;
        }
    }

    if (whether_dynamic_object)
        for (vector<MapObject *>::iterator lit = lLocalMapObjects.begin(), lend = lLocalMapObjects.end(); lit != lend; lit++)
        {
            MapObject *pMObject = *lit;
            pMObject->mnBALocalForKF = 0;
        }
}

void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *>> &LoopConnections, const bool &bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
        new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

    const unsigned int nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
    vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame *pKF = vpKFs[i];
        if (pKF->isBad())
            continue;
        g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->mnId;

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if (it != CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw, tcw, 1.0); // world to camera pose.
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if (pKF == pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi] = VSim3;
    }

    set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

    const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

    // Set Loop edges
    for (map<KeyFrame *, set<KeyFrame *>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
    {
        KeyFrame *pKF = mit->first;
        const long unsigned int nIDi = pKF->mnId;
        const set<KeyFrame *> &spConnections = mit->second;
        const g2o::Sim3 Siw = vScw[nIDi];    // world to i
        const g2o::Sim3 Swi = Siw.inverse(); // i to world

        for (set<KeyFrame *>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
        {
            const long unsigned int nIDj = (*sit)->mnId;
            if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi; // i to j

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji); // from vertex i (first one)  to vertex j (second one).

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
        }
    }

    // Set normal edges
    for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
    {
        KeyFrame *pKF = vpKFs[i];

        const int nIDi = pKF->mnId;

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if (iti != NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame *pParentKF = pKF->GetParent();

        // Spanning tree edge
        if (pParentKF)
        {
            int nIDj = pParentKF->mnId;

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if (itj != NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3 *e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
        for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
        {
            KeyFrame *pLKF = *sit;
            if (pLKF->mnId < pKF->mnId)
            {
                g2o::Sim3 Slw;

                LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if (itl != NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = vScw[pLKF->mnId];

                g2o::Sim3 Sli = Slw * Swi;
                g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                el->setMeasurement(Sli);
                el->information() = matLambda;
                optimizer.addEdge(el);
            }
        }

        // Covisibility graph edges
        const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
        {
            KeyFrame *pKFn = *vit;
            if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
            {
                if (!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                {
                    if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId))))
                        continue;

                    g2o::Sim3 Snw;

                    LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if (itn != NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = vScw[pKFn->mnId];

                    g2o::Sim3 Sni = Snw * Swi;

                    g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                    en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                    en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    en->setMeasurement(Sni);
                    en->information() = matLambda;
                    optimizer.addEdge(en);
                }
            }
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        KeyFrame *pKFi = vpKFs[i];

        const int nIDi = pKFi->mnId;

        g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw = VSim3->estimate();   // wolrd to i
        vCorrectedSwc[nIDi] = CorrectedSiw.inverse(); //  i to world
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *= (1. / s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

        pKFi->SetPose(Tiw); // world to i
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
    {
        MapPoint *pMP = vpMPs[i];

        if (pMP->isBad())
            continue;

        int nIDr;
        if (pMP->mnCorrectedByKF == pCurKF->mnId)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->mnId;
        }

        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
    vSim3->_fix_scale = bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0, 2);
    vSim3->_principle_point1[1] = K1.at<float>(1, 2);
    vSim3->_focal_length1[0] = K1.at<float>(0, 0);
    vSim3->_focal_length1[1] = K1.at<float>(1, 1);
    vSim3->_principle_point2[0] = K2.at<float>(0, 2);
    vSim3->_principle_point2[1] = K2.at<float>(1, 2);
    vSim3->_focal_length2[0] = K2.at<float>(0, 0);
    vSim3->_focal_length2[1] = K2.at<float>(1, 1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2 * N);
    vpEdges12.reserve(2 * N);
    vpEdges21.reserve(2 * N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for (int i = 0; i < N; i++)
    {
        if (!vpMatches1[i])
            continue;

        MapPoint *pMP1 = vpMapPoints1[i];
        MapPoint *pMP2 = vpMatches1[i];

        const int id1 = 2 * i + 1;
        const int id2 = 2 * (i + 1);

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if (pMP1 && pMP2)
        {
            if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0)
            {
                g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D1w = pMP1->GetWorldPos();
                cv::Mat P3D1c = R1w * P3D1w + t1w;
                vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                vPoint1->setId(id1);
                vPoint1->setFixed(true);
                optimizer.addVertex(vPoint1);

                g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                cv::Mat P3D2w = pMP2->GetWorldPos();
                cv::Mat P3D2c = R2w * P3D2w + t2w;
                vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                vPoint2->setId(id2);
                vPoint2->setFixed(true);
                optimizer.addVertex(vPoint2);
            }
            else
                continue;
        }
        else
            continue;

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double, 2, 1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ *e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

        g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double, 2, 1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ *e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

        g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(NULL);
            vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(NULL);
            nBad++;
        }
    }

    int nMoreIterations;
    if (nBad > 0)
        nMoreIterations = 10;
    else
        nMoreIterations = 5;

    if (nCorrespondences - nBad < 10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for (size_t i = 0; i < vpEdges12.size(); i++)
    {
        g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
        if (!e12 || !e21)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = static_cast<MapPoint *>(NULL);
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
    g2oS12 = vSim3_recov->estimate();

    return nIn;
}

} // namespace ORB_SLAM2
