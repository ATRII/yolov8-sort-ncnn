#include <fstream>
#include <iomanip>  // to format image names using setw() and setfill()
#include <unistd.h> // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <set>
#include <algorithm>
#include "../head/utils.h"

#define CNUM 8
// global variables for counting
int total_frames_processed = 0;
double total_time = 0.0;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "ERROR!\nUsage: " << argv[0] << " filename\n";
        return -1;
    }
    TestSORT("test");
    std::cout << "Total Tracking took: " << total_time << " for " << total_frames_processed << " frames or " << ((double)total_frames_processed / (double)total_time) << " FPS\n";
    return 0;
}

void TestSORT(std::string dirName, bool display)
{
    std::cout << "Processing " << dirName << "..." << std::endl;
    // 0. randomly generate colors, only for display
    cv::RNG rng(0xFFFFFFFF);
    cv::Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
        rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
    // TODO: set custom path
    /*
    std::string imgPath = "D:/Data/Track/2DMOT2015/train/" + seqName + "/img1/";
    */
    std::string filepath = "../../data/video/" + dirName + ".mp4";
    if (display)
        if (access(filepath.c_str(), 0) == -1)
        {
            std::cerr << "Image path not found!" << std::endl;
            display = false;
        }

    // 1. read detection file
    std::ifstream detectionFile;
    std::string detFileName = "../../data/detection/" + dirName + ".txt";
    detectionFile.open(detFileName);

    if (!detectionFile.is_open())
    {
        std::cerr << "Error: can not find detection file " << detFileName << std::endl;
        return;
    }

    std::string detLine;
    std::istringstream ss;
    char ch;
    float tpx, tpy, tpw, tph;
    int total_frame = 0;
    std::vector<TrackingBox> tmpBox;
    std::vector<std::vector<TrackingBox>> detFrameData;
    while (getline(detectionFile, detLine))
    {
        TrackingBox tb;
        ss.str(detLine);
        ss >> tb.frame;
        ss >> tpx >> tpy >> tpw >> tph;
        ss.clear();

        tb.box = RECT_F(POINT_F(tpx, tpy), POINT_F(tpx + tpw, tpy + tph));
        if (tb.frame - 1 > detFrameData.size())
        {
            detFrameData.push_back(tmpBox);
            std::cout << detFrameData.size() << " " << tmpBox.size() << std::endl;
            tmpBox.resize(0);
        }
        tmpBox.push_back(tb);
        total_frame = total_frame > tb.frame ? total_frame : tb.frame;
    }
    if (tmpBox.size() > 0)
        detFrameData.push_back(tmpBox);
    detectionFile.close();

    // 2. group det_data by frame
    // NOTE: merged with step1

    // 3. update across frames
    int frame_count = 0;
    int max_age = 1;
    int min_hits = 3;
    double iouThreshold = 0.3;
    std::vector<KalmanTracker> trackers;
    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

    // variables used in the for-loop
    std::vector<RECT_F> predictedBoxes;
    std::vector<std::vector<double>> iouMatrix;
    std::vector<int> assignment;
    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    std::vector<cv::Point> matchedPairs;
    std::vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    double cycle_time = 0.0;
    int64 start_time = 0;

    // prepare result file.
    std::ofstream resultsFile;
    std::string resFileName = "../output/" + dirName + ".txt";
    resultsFile.open(resFileName);

    if (!resultsFile.is_open())
    {
        std::cerr << "Error: can not create file " << resFileName << std::endl;
        return;
    }
    cv::VideoCapture video = cv::VideoCapture(filepath);
    if (!video.isOpened())
    {
        std::cerr << "Error: cv::VideoCapture(" << filepath << ") failed\n"; // err: bad file path
        return;
    }
    //////////////////////////////////////////////
    // main loop
    for (int fi = 0; fi < total_frame; fi++)
    {
        std::cout << "fi: " << fi << std::endl;
        total_frames_processed++;
        frame_count++;
        // cout << frame_count << endl;

        // I used to count running time using clock(), but found it seems to conflict with cv::cvWaitkey(),
        // when they both exists, clock() can not get right result. Now I use cv::getTickCount() instead.
        start_time = cv::getTickCount();

        if (trackers.size() == 0) // the first frame met
        {
            // initialize kalman trackers using first detections.
            for (unsigned int i = 0; i < detFrameData[fi].size(); i++)
            {
                KalmanTracker trk = KalmanTracker(detFrameData[fi][i].box);
                trackers.push_back(trk);
            }
            // output the first frame detections
            for (unsigned int id = 0; id < detFrameData[fi].size(); id++)
            {
                TrackingBox tb = detFrameData[fi][id];
                resultsFile << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << std::endl;
            }
            continue;
        }

        ///////////////////////////////////////
        // 3.1. get predicted locations from existing trackers.
        predictedBoxes.clear();

        for (auto it = trackers.begin(); it != trackers.end();)
        {
            RECT_F pBox = (*it).predict();
            if (pBox.x >= 0 && pBox.y >= 0)
            {
                predictedBoxes.push_back(pBox);
                it++;
            }
            else
            {
                it = trackers.erase(it);
                // cerr << "Box invalid at frame: " << frame_count << endl;
            }
        }

        ///////////////////////////////////////
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
        // dets : detFrameData[fi]
        trkNum = predictedBoxes.size();
        detNum = detFrameData[fi].size();

        iouMatrix.clear();
        iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

        for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < detNum; j++)
            {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[fi][j].box);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm HungAlgo;
        assignment.clear();
        HungAlgo.Solve(iouMatrix, assignment);

        // find matches, unmatched_detections and unmatched_predictions
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum) //	there are unmatched detections
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           std::insert_iterator<std::set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (detNum < trkNum) // there are unmatched trajectory/predictions
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        }
        else
            ;

        // filter out matched with low IOU
        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1) // pass over invalid values
                continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
            {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
                matchedPairs.push_back(cv::Point(i, assignment[i]));
        }

        ///////////////////////////////////////
        // 3.3. updating trackers

        // update matched trackers with assigned detections.
        // each prediction is corresponding to a tracker
        int detIdx, trkIdx;
        for (unsigned int i = 0; i < matchedPairs.size(); i++)
        {
            trkIdx = matchedPairs[i].x;
            detIdx = matchedPairs[i].y;
            trackers[trkIdx].update(detFrameData[fi][detIdx].box);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatchedDetections)
        {
            KalmanTracker tracker = KalmanTracker(detFrameData[fi][umd].box);
            trackers.push_back(tracker);
        }

        // get trackers' output
        frameTrackingResult.clear();
        for (auto it = trackers.begin(); it != trackers.end();)
        {
            if (((*it).m_time_since_update < 1) &&
                ((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
            {
                TrackingBox res;
                res.box = (*it).get_state();
                res.id = (*it).m_id + 1;
                res.frame = frame_count;
                frameTrackingResult.push_back(res);
                it++;
            }
            else
                it++;

            // remove dead tracklet
            if (it != trackers.end() && (*it).m_time_since_update > max_age)
                it = trackers.erase(it);
        }

        cycle_time = (double)(cv::getTickCount() - start_time);
        total_time += cycle_time / cv::getTickFrequency();

        for (auto tb : frameTrackingResult)
            resultsFile << tb.frame << "," << tb.id << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << std::endl;
        std::cout << "checkpoint5" << std::endl;
        if (display) // read image, draw results and show them
        {
            // std::ostringstream oss;
            // oss << imgPath << std::setw(6) << std::setfill('0') << fi + 1;
            // cv::Mat img = cv::imread(oss.str() + ".jpg");
            int w = video.get(cv::CAP_PROP_FRAME_WIDTH);  // frame width
            int h = video.get(cv::CAP_PROP_FRAME_HEIGHT); // frame height
            int tf = video.get(cv::CAP_PROP_FRAME_COUNT); // total frame
            int fps = video.get(cv::CAP_PROP_FPS);        // fps
            cv::Mat m;
            video >> m;
            for (auto tb : frameTrackingResult)
                cv::rectangle(m, tb.box, randColor[tb.id % CNUM], 2, 8, 0);
            imshow(dirName, m);
            cv::waitKey(20);
        }
    }

    resultsFile.close();

    if (display)
        cv::destroyAllWindows();
}