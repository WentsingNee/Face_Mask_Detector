#ifndef FACE_MASK_DETECTOR_DATA_STRUCTURES_H_
#define FACE_MASK_DETECTOR_DATA_STRUCTURES_H_

#include <opencv2/opencv.hpp>
#include <vector>

struct FaceInfo {
    // bounding box coordinates
    float x1;
    float y1;
    float x2;
    float y2;

    // todo: refactor coordinates with cv::Point
    cv::Point2f bottomLeft;
    cv::Point2f topRight;

    // face credibility faceScore
    float faceScore;

    // mask credibility faceScore
    float maskScore;
    bool isWearingMask;
};

struct Image {
    cv::Mat frame;
    cv::Mat processed_frame;
    std::vector<FaceInfo> faceList;
};

struct MaskDetectorSetting {
    int normalisedWidth;
    int normalisedHeight;
    float scaleFactor;
    float scoreThreshold;
    std::string modelPath;

    inline int imageSize() const { return normalisedHeight * normalisedWidth; }
};

struct FaceDetectorSetting {

};

#endif //FACE_MASK_DETECTOR_DATA_STRUCTURES_H_
