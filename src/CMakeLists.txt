cmake_minimum_required(VERSION 3.10.2)
# set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(image_matching_btw_aerial_uav)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_test main.cpp keyPointDetector.cpp  descriptorComputer.cpp imageMatcher.cpp)
target_link_libraries(opencv_test ${OpenCV_LIBS})

