cmake_minimum_required(VERSION 3.8)
project(kmeans_plusplus)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS 2 패키지 찾기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

# PCL과 Eigen3 패키지 찾기 (신규 추가)
find_package(Eigen3 REQUIRED)
find_package(PCL 1.12 REQUIRED) # 설치된 PCL 버전에 맞게 수정할 수 있습니다.

# PCL 라이브러리 사용을 위한 설정 (신규 추가)
include_directories(SYSTEM 
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 실행 파일 생성
add_executable(kmeans_pp_node src/kmeans_pp_node.cpp)
ament_target_dependencies(
  kmeans_pp_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  Eigen3
)

# PCL 라이브러리 링크 (신규 추가)
target_link_libraries(kmeans_pp_node 
  ${PCL_LIBRARIES}
)

# 설치 설정
install(TARGETS
  kmeans_pp_node
  DESTINATION lib/${PROJECT_NAME}
)

# 린터 및 테스트 설정
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()