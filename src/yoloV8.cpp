#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../head/yoloV8.h"

const char *class_names[] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"};
// colors corresponding to label
const cv::Scalar colors[] = {
    cv::Scalar(127, 63, 255, 0.6),
    cv::Scalar(63, 255, 127, 0.6),
    cv::Scalar(255, 127, 63, 0.6),
    cv::Scalar(63, 127, 127, 0.6),
    cv::Scalar(127, 63, 127, 0.6),
    cv::Scalar(127, 127, 63, 0.6),
    cv::Scalar(127, 127, 127, 0.6),
    cv::Scalar(0, 0, 0, 0.6)};
const cv::Scalar WHITE(255, 255, 255);
static float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}
static float intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<GridAndStride> &grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}
static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat &pred, float prob_threshold, std::vector<Object> &objects)
{
    const int num_points = grid_strides.size();
    const int num_class = 80;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float *scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void *)pred.row(i));
            {
                ncnn::Layer *softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float *dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;

            objects.push_back(obj);
        }
    }
}

YoloV8::YoloV8() {}
// load model
int YoloV8::load(int _target_size, std::string _model_path)
{
    yolo.clear();

    yolo.opt = ncnn::Option();
    // yolo.opt.num_threads = 4;
    std::string model_path_param = _model_path + std::string(".param");
    std::string model_path_bin = _model_path + std::string(".bin");
    yolo.load_param(model_path_param.c_str());
    yolo.load_model(model_path_bin.c_str());

    target_size = _target_size;

    norm_vals[0] = 1.0 / 255.0f;
    norm_vals[1] = 1.0 / 255.0f;
    norm_vals[2] = 1.0 / 255.0f;

    return 0;
}
#ifdef TICKCNT
int YoloV8::detect(const cv::Mat &rgb, std::vector<Object> &objects, double &time, float prob_threshold, float nms_threshold)
#else
int YoloV8::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold, float nms_threshold)
#endif
{
#ifdef TICKCNT
    int64 start_time = cv::getTickCount();
#endif
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    // resize
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    /*
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    */
    // CHANGEED: reserve bottom-right padding, while top-left padding removed(like yoloX)
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 0.f);
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);

    std::vector<Object> proposals;

    ncnn::Mat out;
    ex.extract("output", out);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        /*
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        */
        // CHANGED: cause top-left padding removed, no more additional coor calcu
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;
        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
    struct
    {
        bool operator()(const Object &a, const Object &b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
#ifdef TICKCNT
    int64 endtime = cv::getTickCount();
    time += (double)(endtime - start_time) / cv::getTickFrequency();
#endif
    return 0;
}

int YoloV8::draw(cv::Mat &rgb, const std::vector<Object> &objects)
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object &obj = objects[i];

        //         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        // cv::rectangle(rgb, obj.rect, cv::Scalar(255, 0, 0));
        cv::rectangle(rgb, obj.rect, colors[obj.label], 2);
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        // cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
        //               cv::Scalar(255, 255, 255), -1);
        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      colors[obj.label], -1);
        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1);
    }

    return 0;
}
// filter objects via labels
std::vector<Object> YoloV8::filter(const std::vector<Object> &objects, int labelthreshold)
{
    std::vector<Object> ans;
    for (auto obj : objects)
    {
        if (obj.label > labelthreshold)
            continue;
        ans.push_back(obj);
    }
    return ans;
}

// get file name form path
std::string getfilename(std::string path)
{
    int l = path.length();
    if (l == 0)
        return path;
    int i = l - 1;
    for (; i >= 0; i--)
        if (path[i] == '/')
            break;
    return path.substr(i + 1);
}
// change file's extension name
std::string changeext(std::string filename, const char *ext)
{
    int l = filename.length();
    if (l == 0)
        return filename;
    int i = l - 1;
    for (; i >= 0; i--)
        if (filename[i] == '.')
            break;
    return filename.substr(0, i + 1) + std::string(ext);
}