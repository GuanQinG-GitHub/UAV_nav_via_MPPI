cmake_minimum_required(VERSION 3.8)
project(starling_local_world_model)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)

add_executable(local_cloud_cropper_node
  src/local_cloud_cropper.cpp
)

ament_target_dependencies(local_cloud_cropper_node
  rclcpp
  sensor_msgs
  px4_msgs
)

add_executable(local_obstacle_metrics_node
  src/local_obstacle_metrics.cpp
)

ament_target_dependencies(local_obstacle_metrics_node
  rclcpp
  sensor_msgs
  std_msgs
)

install(TARGETS
  local_cloud_cropper_node
  local_obstacle_metrics_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
