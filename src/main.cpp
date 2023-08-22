#include <fstream>
#include "../head/utils.h"
#include "../head/yoloV8.h"

YoloV8 yolov8;
int target_size = 640; // 416; //320;  must be divisible by 32.
int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3)
    {
        std::cerr << "ERROR: wrong param number\nUsage: " << argv[0] << " filename [modelpath]\n";
        return -1;
    }
    const char *filename = argv[1];
    std::string filepath(filename);
    filepath = "../../data/video/" + filepath;
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
        rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    cv::VideoCapture video = cv::VideoCapture(filepath);
    if (!video.isOpened())
    {
        std::cerr << "ERROR: cv::VideoCapture " << filepath << " failed\n"; // err: bad file path
        return -1;
    }
    int w = video.get(cv::CAP_PROP_FRAME_WIDTH);   // frame width
    int h = video.get(cv::CAP_PROP_FRAME_HEIGHT);  // frame height
    int tf = video.get(cv::CAP_PROP_FRAME_COUNT);  // total frame
    int fps = video.get(cv::CAP_PROP_FPS);         // fps
    int ttf = video.get(cv::CAP_PROP_FRAME_COUNT); // total frame
    if (argc == 3)
    {
        const char *modelpath = argv[2];                  // model path
        yolov8.load(target_size, std::string(modelpath)); // load model (once) see yoloyV8.cpp line 246
    }
    else
        yolov8.load(target_size);
    std::string videopath = "../output/video/" + std::string(filename);
    cv::VideoWriter videoWriter(videopath.c_str(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(w, h));
    cv::Mat m;
    int cur_frame = 0;
    std::vector<TrackingBox> detFrameData;
    int total_frames_processed = 0;
    double total_time = 0.0;
    while (true)
    {
        detFrameData.resize(0);
        cur_frame++;
        video >> m;
        if (m.empty())
            break;
        std::vector<Object> objects;
        yolov8.detect(m, objects);                 // recognize the objects
        auto filtered = yolov8.filter(objects, 7); // filter labels
        // yolov8.draw(m, filtered);                  // show the outcome
        for (auto f : filtered)
            detFrameData.push_back(TrackingBox(cur_frame, -1, f.rect));
        auto frameTrackingResult = SORT(detFrameData, total_frames_processed, total_time);

        // videoWriter.write(m.clone()); // save output video under ../output/video/
        // cv::imshow("result", m);
        // cv::waitKey(1);
        for (auto tb : frameTrackingResult)
        {
            cv::rectangle(m, tb.box, randColor[tb.id % CNUM], 2, 8, 0);
            std::string text = getid(tb.id);
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
            int x = tb.box.x;
            int y = tb.box.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > m.cols)
                x = m.cols - label_size.width;
            cv::rectangle(m, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          randColor[tb.id % CNUM], -1);
            cv::putText(m, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0, 0), 1);
        }
        cv::imshow(filename, m);
        videoWriter.write(m.clone());
        cv::waitKey(20);
    }
    videoWriter.release();
    video.release();
    return 0;
}
