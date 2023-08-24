#include <vector>
#include <iostream>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#define POINT_F cv::Point_<float>
#define RECT_F cv::Rect_<float>
#define MAT_F cv::Mat_<float>
#define KALMAN_H 2
#define CNUM 8  // label types
#define TICKCNT // cnt time or not

class TrackingBox
{
public:
    TrackingBox(){};
    TrackingBox(int frame_, int id_, int label_, RECT_F box_) : frame(frame_), id(id_), label(label_), box(box_){};
    int frame;
    int id = -1;
    int label;
    RECT_F box;
};

class HungarianAlgorithm
{
public:
    HungarianAlgorithm();
    ~HungarianAlgorithm();
    double Solve(std::vector<std::vector<double>> &DistMatrix, std::vector<int> &Assignment);

private:
    void assignmentoptimal(int *assignment, double *cost, double *distMatrix, int nOfRows, int nOfColumns);
    void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns);
    void computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows);
    void step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
    void step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
};

class KalmanTracker
{
public:
    KalmanTracker();
    KalmanTracker(RECT_F initRect);
    KalmanTracker(RECT_F initRect, int label_);
    ~KalmanTracker();
    RECT_F predict();
    void update(RECT_F stateMat);
    RECT_F get_state();
    RECT_F get_rect_xysr(float cx, float cy, float s, float r);

    static int kf_count;
    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;
    int label;

private:
    void init_kf(RECT_F stateMat);
    cv::KalmanFilter kf;
    cv::Mat measurement;
    std::vector<RECT_F> m_history;
};

std::vector<TrackingBox> SORT(std::vector<TrackingBox> detFrameData, int &total_frames_processed, double &total_time);
double GetIOU(RECT_F bb_test, RECT_F bb_gt);
std::string getid(int id);