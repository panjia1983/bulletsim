# pcl_ros:
#include_directories("/opt/ros/fuerte/stacks/perception_pcl/pcl_ros/include")
#link_directories("/opt/ros/fuerte/stacks/perception_pcl/pcl_ros/lib")

include_directories("/opt/ros/fuerte/stacks/vision_opencv/cv_bridge/include")
include_directories("/opt/ros/fuerte/stacks/geometry/tf/include"
  "/opt/ros/fuerte/stacks/geometry/tf/msg_gen/cpp/include"
  "/opt/ros/fuerte/stacks/geometry/tf/srv_gen/cpp/include")

link_directories("/opt/ros/fuerte/stacks/geometry/tf/lib"
  "/opt/ros/fuerte/stacks/vision_opencv/cv_bridge/lib")

include_directories(
    ${PCL_INCLUDE_DIRS}
    ${BULLET_INCLUDE_DIR}
    ${JSON_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${BULLETSIM_SOURCE_DIR}/src
    ${BULLET_DIR}/src
)
link_directories(
  ${PCL_LIBRARY_DIRS}   
)

include_directories("${BULLETSIM_SOURCE_DIR}/bulletsim_msgs/build/gen/cpp/bulletsim_msgs")

add_definitions("-DEIGEN_DEFAULT_TO_ROW_MAJOR")

add_library(clouds geom.cpp get_table2.cpp get_chessboard_pose.cpp plane_finding.cpp utils_pcl.cpp cloud_ops.cpp utils_cv.cpp utils_ros.cpp table.cpp ros_robot.cpp
grabcut.cpp)
target_link_libraries(clouds utils ${PCL_LIBRARIES} ${OpenCV_LIBS} ${BULLET_LIBS} boost_signals)

add_executable(preprocessor_node preprocessor_node.cpp)
target_link_libraries(preprocessor_node clouds utils ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ROS_LIBRARIES} boost_signals tf cv_bridge)

add_executable(simple_preprocessor_node simple_preprocessor_node.cpp)
target_link_libraries(simple_preprocessor_node clouds utils ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ROS_LIBRARIES} boost_signals tf cv_bridge)

add_executable(preprocessor_color_node preprocessor_color_node.cpp)
target_link_libraries(preprocessor_color_node clouds utils ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ROS_LIBRARIES} tf cv_bridge)

add_executable(cloud_merger_node cloud_merger_node.cpp)
target_link_libraries(cloud_merger_node clouds utils ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ROS_LIBRARIES} boost_signals tf cv_bridge)

add_executable(save_cloud_node save_cloud_node.cpp)
target_link_libraries(save_cloud_node utils ${PCL_LIBRARIES} ${ROS_LIBRARIES} tf)

add_executable(viz_curves viz_curves.cpp)
target_link_libraries(viz_curves ${PCL_LIBRARIES} clouds boost_program_options utils)

add_executable(get_table_server get_table_server.cpp)
target_link_libraries(get_table_server clouds ${PCL_LIBRARIES} ${ROS_LIBRARIES} utils boost_program_options)

add_executable(cloud_adjuster_node sandbox/cloud_adjuster_node.cpp)
target_link_libraries(cloud_adjuster_node clouds utils ${PCL_LIBRARIES} ${ROS_LIBRARIES})

add_executable(robot_preprocessor_node robot_preprocessor_node.cpp)
target_link_libraries(robot_preprocessor_node clouds utils robots ${PCL_LIBRARIES} ${ROS_LIBRARIES} tf)

add_executable(robot_self_filter robot_self_filter.cpp)
target_link_libraries(robot_self_filter clouds utils robots ${PCL_LIBRARIES} ${ROS_LIBRARIES} boost_signals tf tracking)

add_executable(define_boundary define_boundary.cpp)
target_link_libraries(define_boundary clouds utils ${PCL_LIBRARIES} ${OpenCV_LIBS} ${ROS_LIBRARIES} tf cv_bridge)