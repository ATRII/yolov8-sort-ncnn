cmake_minimum_required(VERSION 3.2)

project(yoloV8)
message("PROJECT_NAME: " ${PROJECT_NAME})

set(CMAKE_BUILD_TYPE "Release")
message("CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

add_subdirectory(src bin)
add_subdirectory(test test)
