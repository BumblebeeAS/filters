cmake_minimum_required(VERSION 3.5)
project(bb_filters)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wnon-virtual-dtor -Woverloaded-virtual)
endif()

# Auto-find dependencies from package.xml
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(DEPENDENCIES
  bb_perception_msgs
  visualization_msgs
  geometry_msgs
  std_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  Eigen3
  rclcpp_components
)
include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)


foreach (FOLDER configs urdf)
  message("Merge install: Creating symbolic link to ${FOLDER} in ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}")
  file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME})
  file(
    CREATE_LINK
    ${CMAKE_CURRENT_SOURCE_DIR}/${FOLDER}
    ${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/${FOLDER}
    SYMBOLIC
  )
endforeach()

add_library(detected_object_3d_vis_component SHARED src/detected_object_3d_vis.cpp)
ament_target_dependencies(
  detected_object_3d_vis_component
  ${DEPENDENCIES}
)
target_link_libraries(
  detected_object_3d_vis_component
  yaml-cpp
)
rclcpp_components_register_node(detected_object_3d_vis_component PLUGIN "DetectedObject3DArrayVisNode" EXECUTABLE detected_object_3d_vis)


add_library(detected_object_2d_vis_component SHARED src/detected_object_2d_vis.cpp)
ament_target_dependencies(
  detected_object_2d_vis_component
  ${DEPENDENCIES}
)
target_link_libraries(
  detected_object_2d_vis_component
  yaml-cpp
)
rclcpp_components_register_node(detected_object_2d_vis_component PLUGIN "DetectedObject2DArrayVisNode" EXECUTABLE detected_object_2d_vis)

install(TARGETS
  detected_object_2d_vis_component
  detected_object_3d_vis_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/detected_object_3d_filter.py
  scripts/detected_object_3d_labelling.py
  scripts/detected_object_2d_filter_projection.py

  scripts/cluster/cluster_poses_node.py
  scripts/cluster/cluster_tf_action.py
  scripts/cluster/cluster_tf_action_server.py
  scripts/cluster/cluster_tf_service_server.py
  scripts/cluster/cluster_tf_multi_action.py
  scripts/cluster/cluster_tf_multi_action_server.py
  scripts/cluster/cluster_tf_multi_service_server.py
  scripts/cluster/cluster_slalom_tfs.py
  scripts/cluster/cluster_slalom_tfs_node.py

  DESTINATION lib/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_mypy_FOUND TRUE)
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()
