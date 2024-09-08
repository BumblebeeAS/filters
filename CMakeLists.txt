cmake_minimum_required(VERSION 3.8)
project(bb_filters)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(bb_msgs REQUIRED)
find_package(bb_perception_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)

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

install(PROGRAMS
  scripts/detected_object_3d_filter.py
  scripts/detected_object_3d_composite_filter.py
  scripts/detected_object_3d_labelling.py
  scripts/lidar_backproject_ml.py
  scripts/lidar_backproject.py
  scripts/UKF_Tracker.py
  scripts/lidar_detected_object_2d_filter.py
  scripts/zed_detector.py
  scripts/robotx_gate_detection.py
  scripts/detected_object_2d_array_vis.py
  scripts/detected_object_3d_array_vis.py
  scripts/robotx_sim_obstacles_converter.py
  scripts/detected_object_2d_filter_projection.py
  scripts/lidar_bbox_extractor.py
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
