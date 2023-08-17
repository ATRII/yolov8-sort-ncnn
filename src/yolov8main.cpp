// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// modified 1-14-2023 Q-engineering

#include "../head/yoloV8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
YoloV8 yolov8;
int target_size = 640; // 416; //320;  must be divisible by 32.
/*
int main(int argc, char **argv)
{
    const char *imagepath = argv[1];

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    yolov8.load(target_size); // load model (once) see yoloyV8.cpp line 246

    std::vector<Object> objects;
    yolov8.detect(m, objects); // recognize the objects
    yolov8.draw(m, objects);   // show the outcome

    cv::imshow("RPi4 - 1.95 GHz - 2 GB ram", m);
    //    cv::imwrite("out.jpg",m);
    cv::waitKey(0);

    return 0;
}
*/
int main(int argc, char **argv)
{
    const char *type = argv[1];      // input type
    const char *filepath = argv[2];  // file path
    const char *modelpath = argv[3]; // model path
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s [img/video] [filepath] [modelpath]\n", argv[0]); // err: bad param number
        return -1;
    }
    std::string type_s(type);
    if (type_s == "img")
    {
        cv::Mat m = cv::imread(filepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", filepath); // err: bad filepath
            return -1;
        }

        yolov8.load(target_size, std::string(modelpath)); // load model (once) see yoloyV8.cpp line 246

        std::vector<Object> objects;
        yolov8.detect(m, objects); // recognize the objects
        yolov8.draw(m, objects);   //  draw boxes

        cv::imshow("result", m);                    // show the outcome
        cv::imwrite("../output/img/output.jpg", m); // save outputimg under ../output/img/
        cv::waitKey(0);
    }
    else if (type_s == "video")
    {
        cv::VideoCapture video = cv::VideoCapture(filepath);
        if (!video.isOpened())
        {
            printf("cv::VideoCapture %s failed\n", filepath); // err: bad file path
            return -1;
        }
        int w = video.get(cv::CAP_PROP_FRAME_WIDTH);      // frame width
        int h = video.get(cv::CAP_PROP_FRAME_HEIGHT);     // frame height
        int tf = video.get(cv::CAP_PROP_FRAME_COUNT);     // total frame
        int fps = video.get(cv::CAP_PROP_FPS);            // fps
        yolov8.load(target_size, std::string(modelpath)); // load model (once) see yoloyV8.cpp line 246
        std::string outputpath = "../output/video/outputvideo.mp4";
        cv::VideoWriter videoWriter(outputpath.c_str(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(w, h));
        cv::Mat m;
        while (true)
        {
            video >> m;
            if (m.empty())
                break;
            std::vector<Object> objects;
            yolov8.detect(m, objects);    // recognize the objects
            yolov8.draw(m, objects);      // show the outcome
            videoWriter.write(m.clone()); // save output video under ../output/video/
            cv::imshow("result", m);
            cv::waitKey(20);
        }
        videoWriter.release();
        video.release();
    }
    else
    {
        fprintf(stderr, "wrong input type\nUsage [img/video]\n"); // err: bad params
        return -1;
    }

    return 0;
}