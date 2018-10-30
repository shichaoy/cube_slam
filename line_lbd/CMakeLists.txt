cmake_minimum_required(VERSION 2.8.3)
project(line_lbd)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  roscpp
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)

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
  LIBRARIES line_lbd_lib
  CATKIN_DEPENDS cmake_modules roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########
include_directories(${OpenCV_INCLUDE_DIRS})
## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native  -std=c++0x"
) 

## Declare a C++ library
add_library(line_lbd_lib
  libs/binary_descriptor.cpp libs/binary_descriptor_matcher.cpp libs/LSDDetector.cpp  libs/lsd.cpp libs/bitarray.hpp  
  libs/bitops.hpp libs/draw.cpp libs/precomp.hpp libs/types.hpp  class/line_lbd_allclass.cpp
)
target_link_libraries(line_lbd_lib ${OpenCV_LIBRARIES} ${catkin_LIBRARIES}) 
## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(line_lbd ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable

add_executable(linelbd_detect_node src/detect_lines.cpp)
target_link_libraries(linelbd_detect_node  line_lbd_lib ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})


#add_executable(linelbd_descrip_node src/compute_descriptors.cpp)
#target_link_libraries(linelbd_descrip_node  line_lbd_lib ${OpenCV_LIBRARIES} ${catkin_LIBRARIES})


## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(line_lbd_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(line_lbd_node
#   ${catkin_LIBRARIES}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_line_lbd.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
