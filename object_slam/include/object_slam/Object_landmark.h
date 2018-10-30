#pragma once

#include <vector>

#include <object_slam/g2o_Object.h>

class object_landmark{
public:
  
    g2o::cuboid cube_meas;  //cube_value
    g2o::VertexCuboid* cube_vertex;
    double meas_quality;  // [0,1] the higher, the better    
    
};