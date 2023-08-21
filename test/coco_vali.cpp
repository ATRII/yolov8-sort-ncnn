#include "../head/yoloV8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
using std::filesystem::__cxx11::directory_iterator;
YoloV8 yolov8;
int target_size = 640; // 416; //320;  must be divisible by 32.

int main(int argc, char **argv)
{
    std::string dir("../../../img/coco/");
    // walks through coco's directory
    for (auto &v : directory_iterator(dir))
    {
        std::string fileName = dir + v.path().filename().string();
        cv::Mat m = cv::imread(fileName.c_str(), 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", fileName.c_str()); // err: bad filepath
            return -1;
        }
        std::string modelpath = "../../../model/yolov8s";
        yolov8.load(target_size, std::string(modelpath)); // load model (once) see yoloyV8.cpp line 246

        std::vector<Object> objects;
        yolov8.detect(m, objects); // recognize the objects
        yolov8.draw(m, objects);   //  draw boxes

        cv::imshow("result", m); // show the outcome
        // cv::imwrite("../output/img/output.jpg", m); // save outputimg under ../output/img/
        cv::waitKey(1000);
    }
}