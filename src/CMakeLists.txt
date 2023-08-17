cmake_minimum_required(VERSION 3.2)
project(yoloV8)

set(OPENCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
find_package(OpenCV REQUIRED)

set(ncnn_DIR "/home/yh/code/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories($(PROJECT_SOURCE_DIR)/head)
add_executable(main yolov8main.cpp yoloV8.cpp)
target_link_libraries(main ncnn)
target_link_libraries(main ${OpenCV_LIBS})