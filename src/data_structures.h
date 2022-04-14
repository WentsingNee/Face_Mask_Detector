//
// Created by Hanniko on 2022/4/11.
//

#ifndef FACE_MASK_DETECTOR_DATA_STRUCTURES_H_
#define FACE_MASK_DETECTOR_DATA_STRUCTURES_H_

struct FaceInfo {
    // bounding box coordinates
    float x1;
    float y1;
    float x2;
    float y2;

    // credibility score
    float score;
};

struct Image {

};

#endif //FACE_MASK_DETECTOR_DATA_STRUCTURES_H_
