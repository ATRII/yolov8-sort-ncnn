#define YOLOV8_H
#define TICKCNT

#include <opencv2/core/core.hpp>
#include <net.h>
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

class YoloV8
{
public:
    YoloV8();
    int load(int target_size, std::string model_path = "../../model/yolov8s");
#ifdef TICKCNT
    int detect(const cv::Mat &rgb, std::vector<Object> &objects, double &time, float prob_threshold = 0.4f, float nms_threshold = 0.5f);
#else
    int detect(const cv::Mat &rgb, std::vector<Object> &objects, double prob_threshold = 0.4f, float nms_threshold = 0.5f);
#endif
    int draw(cv::Mat &rgb, const std::vector<Object> &objects);
    std::vector<Object> filter(const std::vector<Object> &objects, int labelthreshold);

private:
    ncnn::Net yolo;
    int target_size;
    float norm_vals[3];
};
std::string getfilename(std::string path);
std::string changeext(std::string filename, const char *ext);
