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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"

#include <pangolin/pangolin.h>
#include <mutex>

// by me
#include "detect_3d_cuboid/object_3d_util.h"
#include "Parameters.h"
#include "Converter.h"
#include "MapObject.h"

using namespace std;
namespace ORB_SLAM2
{

MapDrawer::MapDrawer(Map *pMap, const string &strSettingPath) : mpMap(pMap)
{
	cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

	mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
	mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
	mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
	mPointSize = fSettings["Viewer.PointSize"];
	mCameraSize = fSettings["Viewer.CameraSize"];
	mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

	box_colors.push_back(Vector3f(230, 0, 0) / 255.0);	 // red  0
	box_colors.push_back(Vector3f(60, 180, 75) / 255.0);   // green  1
	box_colors.push_back(Vector3f(0, 0, 255) / 255.0);	 // blue  2
	box_colors.push_back(Vector3f(255, 0, 255) / 255.0);   // Magenta  3
	box_colors.push_back(Vector3f(255, 165, 0) / 255.0);   // orange 4
	box_colors.push_back(Vector3f(128, 0, 128) / 255.0);   //purple 5
	box_colors.push_back(Vector3f(0, 255, 255) / 255.0);   //cyan 6
	box_colors.push_back(Vector3f(210, 245, 60) / 255.0);  //lime  7
	box_colors.push_back(Vector3f(250, 190, 190) / 255.0); //pink  8
	box_colors.push_back(Vector3f(0, 128, 128) / 255.0);   //Teal  9

	all_edge_pt_ids.resize(8, 2); // draw 8 edges except front face
	all_edge_pt_ids << 2, 3, 3, 4, 4, 1, 3, 7, 4, 8, 6, 7, 7, 8, 8, 5;
	all_edge_pt_ids.array() -= 1;
	front_edge_pt_ids.resize(4, 2);
	front_edge_pt_ids << 1, 2, 2, 6, 6, 5, 5, 1;
	front_edge_pt_ids.array() -= 1;
}

void MapDrawer::DrawMapPoints()
{
	const vector<MapPoint *> &vpMPs = mpMap->GetAllMapPoints();			 // all map points
	const vector<MapPoint *> &vpRefMPs = mpMap->GetReferenceMapPoints(); // local covisible map points. a subset of vpMPs

	set<MapPoint *> spRefMPs(vpRefMPs.begin(), vpRefMPs.end()); // why use set? any duplication in vpRefMPs

	if (vpMPs.empty())
		return;

	// draw common map points
	int shift_truth_map = 0;
	float normal_point_size = 1; // raw: 1

	if (draw_nonlocal_mappoint)
	{
		glPointSize(mPointSize * normal_point_size);
		glBegin(GL_POINTS);
		if (whether_detect_object)
			glColor3f(0.5, 0.5, 0.5); // gray
		else
			glColor3f(0.0, 0.0, 1.0); // blue
		for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
		{
			if (vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i])) // not  local mappoint
				continue;
			cv::Mat pos = vpMPs[i]->GetWorldPos();
			glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2)); //+shift_truth_map
		}
		glEnd();
	}

	if (1) // a subset of mappoints  points BA in local map
	{
		glPointSize(mPointSize * normal_point_size);
		glBegin(GL_POINTS);
		glColor3f(0.5, 0.5, 0.5); // gray
		for (set<MapPoint *>::iterator sit = spRefMPs.begin(), send = spRefMPs.end(); sit != send; sit++)
		{
			if ((*sit)->isBad())
				continue;
			cv::Mat pos = (*sit)->GetWorldPos();
			glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2)); //+shift_truth_map
		}
		glEnd();
	}

	if (enable_ground_height_scale) // draw ground point, better to show gray points for above.
	{
		glPointSize(mPointSize * normal_point_size * 2);
		glBegin(GL_POINTS);
		glColor3f(0.0, 1.0, 0.0); // red
		for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
		{
			if (vpMPs[i]->isBad())
				continue;
			if (vpMPs[i]->ground_fitted_point)
			{
				cv::Mat pos = vpMPs[i]->GetWorldPos();
				glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2)); //+shift_truth_map
			}
		}
		glEnd();
	}

	// draw the object points
	if (whether_detect_object && associate_point_with_object)
	{
		glPointSize(mPointSize * 5);
		glBegin(GL_POINTS);
		const vector<MapObject *> all_Map_objs = mpMap->GetAllMapObjects();
		for (size_t object_id = 0; object_id < all_Map_objs.size(); object_id++)
		{
			MapObject *obj_landmark = all_Map_objs[object_id];
			if (obj_landmark->isBad())
				continue;
			vector<MapPoint *> owned_mappoints;
			owned_mappoints = obj_landmark->used_points_in_BA_filtered; // points really used in BA
			if (whether_dynamic_object)
				owned_mappoints = obj_landmark->GetUniqueMapPoints();
			Vector3f box_color = box_colors[obj_landmark->mnId % box_colors.size()];
			glColor4f(box_color(0), box_color(1), box_color(2), 1.0f);
			for (size_t pt_id = 0; pt_id < owned_mappoints.size(); pt_id++)
			{
				MapPoint *mpt = owned_mappoints[pt_id];
				if (!mpt->isBad())
				{
					cv::Mat pos;
					if (bundle_object_opti)
					{
						if (obj_landmark->obj_been_optimized)
							pos = mpt->GetWorldPosBA();
						else
							continue;
					}
					else
					{
						pos = mpt->GetWorldPos();
					}
					if (pos.rows == 0)
						continue;

					if (mpt->is_dynamic)
					{
						if (mpt->Observations() < 2) //don't show just depth inited frame
							continue;
					}
					glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
				}
			}
		}
		glEnd();
	}
}

void MapDrawer::DrawMapCuboids() // ideally this should be draw map cuboids.
{
	// make sure final cuboid is in init world frame.
	// draw all map objects
	const vector<MapObject *> all_Map_objs = mpMap->GetAllMapObjects();
	Vector4d front_face_color(1.0, 0.0, 1.0, 1.0); // draw front face edges magenta

	for (size_t object_id = 0; object_id < all_Map_objs.size(); object_id++)
	{
		MapObject *obj_landmark = all_Map_objs[object_id];

		if (obj_landmark->isBad()) // some good, some bad, some not determined
			continue;

		// show objects that being optimized! for kitti fix scale, this will make map visualization better.
		if (bundle_object_opti)
		{
			if (!obj_landmark->obj_been_optimized)
			{
				continue;
			}
		}

		Eigen::MatrixXd cube_corners;
		if (bundle_object_opti && whether_dynamic_object)
			cube_corners = obj_landmark->pose_Twc_afterba.compute3D_BoxCorner(); // show pose after BA, will have some delay, but looks good
		else
			cube_corners = obj_landmark->GetWorldPos().compute3D_BoxCorner();

		if (obj_landmark->Observations() == 1)
		{
			glLineWidth(mGraphLineWidth * 2);
			glBegin(GL_LINES);
			front_face_color = Vector4d(0, 0, 128.0 / 255.0, 1.0);
		}
		else
		{
			glLineWidth(mGraphLineWidth * 4);
			glBegin(GL_LINES);
			front_face_color = Vector4d(1.0, 0.0, 1.0, 1.0);
		}
		// draw cuboid
		Vector3f box_color = box_colors[obj_landmark->mnId % box_colors.size()];
		glColor4f(box_color(0), box_color(1), box_color(2), 1.0f); // draw all edges  cyan
		for (int line_id = 0; line_id < all_edge_pt_ids.rows(); line_id++)
		{
			glVertex3f(cube_corners(0, all_edge_pt_ids(line_id, 0)), cube_corners(1, all_edge_pt_ids(line_id, 0)), cube_corners(2, all_edge_pt_ids(line_id, 0)));
			glVertex3f(cube_corners(0, all_edge_pt_ids(line_id, 1)), cube_corners(1, all_edge_pt_ids(line_id, 1)), cube_corners(2, all_edge_pt_ids(line_id, 1)));
		}
		for (int line_id = 0; line_id < front_edge_pt_ids.rows(); line_id++)
		{
			glVertex3f(cube_corners(0, front_edge_pt_ids(line_id, 0)), cube_corners(1, front_edge_pt_ids(line_id, 0)), cube_corners(2, front_edge_pt_ids(line_id, 0)));
			glVertex3f(cube_corners(0, front_edge_pt_ids(line_id, 1)), cube_corners(1, front_edge_pt_ids(line_id, 1)), cube_corners(2, front_edge_pt_ids(line_id, 1)));
		}
		glEnd();

		// draw dynamic object history path
		if (whether_dynamic_object && obj_landmark->is_dynamic && obj_landmark->allDynamicPoses.size() > 0)
		{
			glLineWidth(mGraphLineWidth * 2);
			glBegin(GL_LINE_STRIP);									   // line strip connects adjacent points
			glColor4f(box_color(0), box_color(1), box_color(2), 1.0f); // draw all edges  cyan
			for (auto it = obj_landmark->allDynamicPoses.begin(); it != obj_landmark->allDynamicPoses.end(); it++)
			{
				if (bundle_object_opti && !it->second.second) //only show optimized frame object pose
					continue;
				g2o::cuboid cubepose = it->second.first;
				glVertex3f(cubepose.pose.translation()(0), cubepose.pose.translation()(1), cubepose.pose.translation()(2));
			}
			glEnd();
		}
	}
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
	const float &w = mKeyFrameSize;
	const float h = w * 0.75;
	const float z = w * 0.6;

	const vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();

	if (bDrawKF)
	{
		for (size_t i = 0; i < vpKFs.size(); i++)
		{
			KeyFrame *pKF = vpKFs[i];
			cv::Mat Twc = pKF->GetPoseInverse().t();

			glPushMatrix();

			glMultMatrixf(Twc.ptr<GLfloat>(0));

			glLineWidth(mKeyFrameLineWidth);
			glColor3f(0.0f, 0.0f, 1.0f);
			glBegin(GL_LINES);
			glVertex3f(0, 0, 0);
			glVertex3f(w, h, z);
			glVertex3f(0, 0, 0);
			glVertex3f(w, -h, z);
			glVertex3f(0, 0, 0);
			glVertex3f(-w, -h, z);
			glVertex3f(0, 0, 0);
			glVertex3f(-w, h, z);

			glVertex3f(w, h, z);
			glVertex3f(w, -h, z);

			glVertex3f(-w, h, z);
			glVertex3f(-w, -h, z);

			glVertex3f(-w, h, z);
			glVertex3f(w, h, z);

			glVertex3f(-w, -h, z);
			glVertex3f(w, -h, z);
			glEnd();

			glPopMatrix();
		}
	}

	if (bDrawGraph)
	{
		glLineWidth(mGraphLineWidth);
		glColor4f(0.0f, 1.0f, 0.0f, 0.6f);
		glBegin(GL_LINES);

		for (size_t i = 0; i < vpKFs.size(); i++)
		{
			// Covisibility Graph
			const vector<KeyFrame *> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
			cv::Mat Ow = vpKFs[i]->GetCameraCenter();
			if (!vCovKFs.empty())
			{
				for (vector<KeyFrame *>::const_iterator vit = vCovKFs.begin(), vend = vCovKFs.end(); vit != vend; vit++)
				{
					if ((*vit)->mnId < vpKFs[i]->mnId)
						continue;
					cv::Mat Ow2 = (*vit)->GetCameraCenter();
					glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
					glVertex3f(Ow2.at<float>(0), Ow2.at<float>(1), Ow2.at<float>(2));
				}
			}

			// Spanning tree
			KeyFrame *pParent = vpKFs[i]->GetParent();
			if (pParent)
			{
				cv::Mat Owp = pParent->GetCameraCenter();
				glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
				glVertex3f(Owp.at<float>(0), Owp.at<float>(1), Owp.at<float>(2));
			}

			// Loops
			set<KeyFrame *> sLoopKFs = vpKFs[i]->GetLoopEdges();
			for (set<KeyFrame *>::iterator sit = sLoopKFs.begin(), send = sLoopKFs.end(); sit != send; sit++)
			{
				if ((*sit)->mnId < vpKFs[i]->mnId)
					continue;
				cv::Mat Owl = (*sit)->GetCameraCenter();
				glVertex3f(Ow.at<float>(0), Ow.at<float>(1), Ow.at<float>(2));
				glVertex3f(Owl.at<float>(0), Owl.at<float>(1), Owl.at<float>(2));
			}
		}

		glEnd();
	}
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
	const float &w = mCameraSize;
	const float h = w * 0.75;
	const float z = w * 0.6;

	glPushMatrix();

#ifdef HAVE_GLES
	glMultMatrixf(Twc.m);
#else
	glMultMatrixd(Twc.m);
#endif

	glLineWidth(mCameraLineWidth);
	glColor3f(0.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3f(w, h, z);
	glVertex3f(0, 0, 0);
	glVertex3f(w, -h, z);
	glVertex3f(0, 0, 0);
	glVertex3f(-w, -h, z);
	glVertex3f(0, 0, 0);
	glVertex3f(-w, h, z);

	glVertex3f(w, h, z);
	glVertex3f(w, -h, z);

	glVertex3f(-w, h, z);
	glVertex3f(-w, -h, z);

	glVertex3f(-w, h, z);
	glVertex3f(w, h, z);

	glVertex3f(-w, -h, z);
	glVertex3f(w, -h, z);
	glEnd();

	glPopMatrix();

	if (draw_map_truth_paths)
		if (truth_poses.rows() > 0)
		{
			glLineWidth(mGraphLineWidth * 2);
			glBegin(GL_LINE_STRIP); // line strip connects adjacent points
			glColor4f(0.0f, 1.0f, 1.0f, 1.0f);
			for (int pt_id = 0; pt_id < truth_poses.rows(); pt_id++)
			{
				glVertex3f(truth_poses(pt_id, 0), truth_poses(pt_id, 1), truth_poses(pt_id, 2));
			}
			glEnd();
		}

	// draw path connecting keyframes. no complicated graph.
	const vector<KeyFrame *> vpKFSequ = mpMap->GetAllKeyFramesSequential();
	glLineWidth(mGraphLineWidth * 2);
	glBegin(GL_LINE_STRIP);
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);

	if (vpKFSequ.size() > 0)
		for (size_t i = 0; i < vpKFSequ.size(); i++)
		{
			KeyFrame *currKf = vpKFSequ[i];
			if (currKf)
				if (!currKf->isBad())
				{
					cv::Mat currOw = currKf->GetCameraCenter();

					glVertex3f(currOw.at<float>(0), currOw.at<float>(1), currOw.at<float>(2));
				}
		}
	glEnd();
}

void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
	unique_lock<mutex> lock(mMutexCamera);
	mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
	if (!mCameraPose.empty())
	{
		cv::Mat Rwc(3, 3, CV_32F);
		cv::Mat twc(3, 1, CV_32F);
		{
			unique_lock<mutex> lock(mMutexCamera);
			Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
			twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
		}

		M.m[0] = Rwc.at<float>(0, 0);
		M.m[1] = Rwc.at<float>(1, 0);
		M.m[2] = Rwc.at<float>(2, 0);
		M.m[3] = 0.0;

		M.m[4] = Rwc.at<float>(0, 1);
		M.m[5] = Rwc.at<float>(1, 1);
		M.m[6] = Rwc.at<float>(2, 1);
		M.m[7] = 0.0;

		M.m[8] = Rwc.at<float>(0, 2);
		M.m[9] = Rwc.at<float>(1, 2);
		M.m[10] = Rwc.at<float>(2, 2);
		M.m[11] = 0.0;

		M.m[12] = twc.at<float>(0);
		M.m[13] = twc.at<float>(1);
		M.m[14] = twc.at<float>(2);
		M.m[15] = 1.0;
	}
	else
		M.SetIdentity();
}

} // namespace ORB_SLAM2
