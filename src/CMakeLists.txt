cmake_minimum_required(VERSION 3.2)
project(yoloV8)

set(OPENCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
find_package(OpenCV REQUIRED)

set(ncnn_DIR "/home/yh/code/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($(PROJECT_SOURCE_DIR)/head)
add_executable(yolov8 yolov8main.cpp yoloV8.cpp)
add_executable(sort sort.cpp utils.cpp)
target_link_libraries(yolov8 ncnn)
target_link_libraries(yolov8 ${OpenCV_LIBS})
target_link_libraries(sort ncnn)
target_link_libraries(sort ${OpenCV_LIBS})