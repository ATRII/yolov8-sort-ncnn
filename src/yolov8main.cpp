#include "../head/yoloV8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
YoloV8 yolov8;
int target_size = 640; // 416; //320;  must be divisible by 32.

int main(int argc, char **argv)
{
    const char *type = argv[1];     // input type
    const char *filepath = argv[2]; // file path
    // const char *modelpath = argv[3]; // model path
    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " img|video filepath [modelpath]\n"; // err: bad param number
        return -1;
    }
    std::string type_s(type);
#ifdef TICKCNT
    double yolo_time = 0.0;
#endif
    if (type_s == "img")
    {
        cv::Mat m = cv::imread(filepath, 1);
        if (m.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", filepath); // err: bad filepath
            return -1;
        }
        if (argc == 4)
        {
            const char *modelpath = argv[3];                  // model path
            yolov8.load(target_size, std::string(modelpath)); // load model (once) see yoloyV8.cpp line 246
        }
        else
            yolov8.load(target_size);
        std::vector<Object> objects;
#ifdef TICKCNT
        yolov8.detect(m, objects, yolo_time);
#else
        yolov8.detect(m, objects);     // recognize the objects
#endif
        auto filtered = yolov8.filter(objects, 7); // filter labels
        yolov8.draw(m, filtered);                  // draw boxes

        cv::imshow("result", m); // show the outcome
        std::string outputpath = getfilename(filepath);
        outputpath = "../output/img/" + outputpath;
        cv::imwrite(outputpath, m); // save outputimg under ../output/img/
        cv::waitKey(0);
    }
    else if (type_s == "video")
    {
        cv::VideoCapture video = cv::VideoCapture(filepath);
        if (!video.isOpened())
        {
            fprintf(stderr, "cv::VideoCapture %s failed\n", filepath); // err: bad file path
            return -1;
        }
        int w = video.get(cv::CAP_PROP_FRAME_WIDTH);   // frame width
        int h = video.get(cv::CAP_PROP_FRAME_HEIGHT);  // frame height
        int tf = video.get(cv::CAP_PROP_FRAME_COUNT);  // total frame
        int fps = video.get(cv::CAP_PROP_FPS);         // fps
        int ttf = video.get(cv::CAP_PROP_FRAME_COUNT); // total frame
        if (argc == 4)
        {
            const char *modelpath = argv[3];                  // model path
            yolov8.load(target_size, std::string(modelpath)); // load model (once) see yoloyV8.cpp line 246
        }
        else
            yolov8.load(target_size);
        std::string outputpath = getfilename(filepath);
        std::string videopath = "../output/video/" + outputpath;
        cv::VideoWriter videoWriter(videopath.c_str(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(w, h));
        cv::Mat m;
        std::string detfilename = "../../data/detection/" + changeext(outputpath, "txt");
        std::ofstream fileclean(detfilename, std::ios_base::out);
        std::ofstream fout(detfilename, std::ios::app);
        int cur_frame = 0;
        while (true)
        {
            cur_frame++;
            video >> m;
            if (m.empty())
                break;
            std::vector<Object> objects;
#ifdef TICKCNT
            yolov8.detect(m, objects, yolo_time);
#else
            yolov8.detect(m, objects); // recognize the objects
#endif
            auto filtered = yolov8.filter(objects, 7); // filter labels
            yolov8.draw(m, filtered);                  // show the outcome
            if (fout.is_open())
                for (auto obj : filtered)
                    fout << cur_frame << " " << obj.rect.x << " " << obj.rect.y << " " << obj.rect.width << " " << obj.rect.height << "\n";

            videoWriter.write(m.clone()); // save output video under ../output/video/
            cv::imshow("result", m);
            cv::waitKey(5);
        }
        videoWriter.release();
        video.release();
        fout.close();
    }
    else
    {

        fprintf(stderr, "wrong input type\nUsage: [img|video]\n"); // err: bad params
        return -1;
    }

    return 0;
}