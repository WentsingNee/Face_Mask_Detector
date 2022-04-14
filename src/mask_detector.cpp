#include "mask_detector.h"

MaskDetector::MaskDetector(const MaskDetectionSetting &mdSetting, const MobileConfig &mConfig) {

}

MaskDetector::~MaskDetector() {

}

cv::Mat MaskDetector::detect(cv::Mat &&frame, const std::vector<FaceInfo> &faceInfoList) {
    return frame;
}
