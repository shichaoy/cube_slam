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
#pragma once

#ifndef PARAMETERS_H
#define PARAMETERS_H
#include <string>

namespace ORB_SLAM2
{
extern bool enable_viewer;
extern bool enable_viewmap;
extern bool enable_viewimage;

extern bool parallel_mapping;

extern bool whether_detect_object;
extern bool whether_read_offline_cuboidtxt;
extern bool associate_point_with_object;

extern bool whether_dynamic_object;
extern bool remove_dynamic_features;
extern bool use_dynamic_klt_features;

extern bool mono_firstframe_truth_depth_init;
extern bool mono_firstframe_Obj_depth_init;
extern bool mono_allframe_Obj_depth_init;

extern bool enable_ground_height_scale;
extern bool build_worldframe_on_ground;

// for BA
extern bool bundle_object_opti;
extern double object_velocity_BA_weight;
extern double camera_object_BA_weight;

//dynamic debug
extern bool ba_dyna_pt_obj_cam;
extern bool ba_dyna_obj_velo;
extern bool ba_dyna_obj_cam;

// for gui
extern bool draw_map_truth_paths;
extern bool draw_nonlocal_mappoint;

enum Scene_Name
{
    voidtype = 0,
    kitti
};
extern Scene_Name scene_unique_id;
extern std::string base_data_folder;

} // namespace ORB_SLAM2

#endif // PARAMETERS_H
