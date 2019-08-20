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
#include "Parameters.h"

namespace ORB_SLAM2
{
bool enable_viewer = true;
bool enable_viewmap = true;
bool enable_viewimage = true;

bool parallel_mapping = true;

bool whether_detect_object = false;
bool whether_read_offline_cuboidtxt = false;
bool associate_point_with_object = false;

bool whether_dynamic_object = false;
bool remove_dynamic_features = false;
bool use_dynamic_klt_features = false;

bool mono_firstframe_truth_depth_init = false;
bool mono_firstframe_Obj_depth_init = false;
bool mono_allframe_Obj_depth_init = false;

bool enable_ground_height_scale = false;
bool build_worldframe_on_ground = false;

// for BA
bool bundle_object_opti = false;
double object_velocity_BA_weight = 1.0;
double camera_object_BA_weight = 1.0;

// for gui
bool draw_map_truth_paths = true;
bool draw_nonlocal_mappoint = true;

//dynamic debug
bool ba_dyna_pt_obj_cam = false;
bool ba_dyna_obj_velo = true;
bool ba_dyna_obj_cam = true;

std::string base_data_folder;

Scene_Name scene_unique_id;

} // namespace ORB_SLAM2
