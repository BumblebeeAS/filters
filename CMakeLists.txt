cmake_minimum_required(VERSION 3.10.2)
project(bb_filters)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
  image_transport
  sensor_msgs
  tf2_ros
)

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS rospy cv_bridge image_transport sensor_msgs tf2_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_python_setup()
# catkin_install_python(PROGRAMS scripts/enu_ned_republisher_noetic.py
#     scripts/tiles_analysis_node.py
#     DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
