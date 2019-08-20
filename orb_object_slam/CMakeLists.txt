cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
project(orb_object_slam)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
# 
# # Check C++11 or C++0x support
# include(CheckCXXCompilerFlag)
# CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
# CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
# if(COMPILER_SUPPORTS_CXX11)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#    add_definitions(-DCOMPILEDWITHC11)
#    message(STATUS "Using flag -std=c++11.")
# elseif(COMPILER_SUPPORTS_CXX0X)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#    add_definitions(-DCOMPILEDWITHC0X)
#    message(STATUS "Using flag -std=c++0x.")
# else()
#    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
# endif()
# 
# LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


find_package(catkin REQUIRED COMPONENTS 
  cmake_modules
  roscpp 
  sensor_msgs 
  nav_msgs 
  message_filters 
  image_transport 
  cv_bridge
  image_geometry
  pcl_ros
  pcl_conversions
  detect_3d_cuboid
  tictoc_profiler
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PythonLibs 2.7 REQUIRED) 
find_package(PCL REQUIRED COMPONENTS common)

catkin_package(
  INCLUDE_DIRS include 
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS cmake_modules roscpp nav_msgs sensor_msgs message_filters image_transport cv_bridge pcl_ros  pcl_conversions
    detect_3d_cuboid  tictoc_profiler
)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${catkin_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
${EIGEN_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PYTHON_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)

SET(CMAKE_BUILD_TYPE Release)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/Tracking_util.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/MapObject.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/Parameters.cpp
src/g2o_Object.cpp
) 

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${catkin_LIBRARIES}  # need detect_3d_cuboid
)

#build ros node, saved in workspace/devel.
#Node for monocular camera
add_executable(ros_mono
src/ros_mono.cc
)
target_link_libraries(ros_mono  ${PROJECT_NAME} ${catkin_LIBRARIES} )

# # Node for RGB-D camera
# add_executable(ros_rgbd
# src/ros_rgbd.cc
# )
# target_link_libraries(ros_rgbd  ${PROJECT_NAME} ${catkin_LIBRARIES} )
# 
# # Node for Stereo camera
# add_executable(ros_stereo
# src/ros_stereo.cc
# )
# target_link_libraries(ros_stereo  ${PROJECT_NAME} ${catkin_LIBRARIES} )



#build non-ros node, saved in current package Build examples
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
# add_executable(rgbd_tum
# Examples/RGB-D/rgbd_tum.cc)
# target_link_libraries(rgbd_tum ${PROJECT_NAME})
# 
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
# add_executable(rgbd_kinect_me
# Examples/RGB-D/rgbd_kinect_me.cc)
# target_link_libraries(rgbd_kinect_me ${PROJECT_NAME})
# 
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)
# add_executable(stereo_kitti
# Examples/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti ${PROJECT_NAME})
# 
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
# add_executable(mono_tum
# Examples/Monocular/mono_tum.cc)
# target_link_libraries(mono_tum ${PROJECT_NAME})
# 
# add_executable(mono_kitti
# Examples/Monocular/mono_kitti.cc)
# target_link_libraries(mono_kitti ${PROJECT_NAME})


#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Vocabulary)
#add_executable(bin_vocabulary
#Vocabulary/bin_vocabulary.cc)
#target_link_libraries(bin_vocabulary ${PROJECT_NAME})




message(STATUS ${OpenCV_INCLUDE_DIRS})
