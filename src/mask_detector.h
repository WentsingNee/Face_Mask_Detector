#ifndef MASK_DETECTOR_H_
#define MASK_DETECTOR_H_

#include <opencv2/opencv.hpp>
#include <paddle_api.h>
#include <paddle_use_kernels.h>
#include <paddle_use_ops.h>

#include "data_structures.h"

using namespace paddle::lite_api;

class MaskDetector {
public:
    MaskDetector(const MaskDetectionSetting& mdSetting ,const MobileConfig& mConfig);
    ~MaskDetector();

    cv::Mat detect(cv::Mat&& frame, const std::vector<FaceInfo>& faceInfoList);

private:
    MobileConfig mobileConfig;
    MaskDetectionSetting maskDetectionSetting;
    std::unique_ptr<PaddlePredictor> predictor;
    std::unique_ptr<Tensor> inputTensor;
    std::unique_ptr<Tensor> outputTensor;
};

#endif // MASK_DETECTOR_H_