cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
project(object_slam)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
  cv_bridge
  pcl_ros
  detect_3d_cuboid
  line_lbd
)

## System dependencies are found with CMake's conventions
#find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 #LIBRARIES object_slam_lib
 CATKIN_DEPENDS cmake_modules roscpp cv_bridge detect_3d_cuboid line_lbd
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

SET(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# C++11
# set_source_files_properties(${SRCS} PROPERTIES COMPILE_FLAGS -std=c++0x)
# set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,. -L${CMAKE_CURRENT_BINARY_DIR}")

## Declare a C++ library


## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(object_slam ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
add_executable(object_slam_node src/main_obj.cpp)

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(object_slam_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(object_slam_node
  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_object_slam.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
