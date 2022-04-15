set (CMAKE_CXX_STANDARD 11)

set (LIB_SRC
        bbox_tracker.cpp
        face_detector.cpp
        mask_detector.cpp
        data_structures.h)

# opencv packages
find_package (OpenCV REQUIRED)

# include Paddle Lite
set (PADDLE_LITE_DIR
        /Users/minhao/Downloads/inference_lite_lib)
include_directories (${PADDLE_LITE_DIR}/cxx/include)
link_directories (${PADDLE_LITE_DIR}/cxx/lib)

# build library for test
add_library (FaceMaskDetectionLib ${LIB_SRC})
target_link_libraries (FaceMaskDetectionLib ${OpenCV_LIBS} paddle_light_api_shared)

add_executable (FaceMaskDetector ${LIB_SRC} main.cpp)
target_link_libraries (FaceMaskDetector ${OpenCV_LIBS} paddle_light_api_shared)
