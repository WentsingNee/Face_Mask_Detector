set(CMAKE_CXX_STANDARD 11)
project(FaceMaskDetection)

set(LIB_SRC
        bbox_tracker.cpp
        face_detector.cpp
        mask_detector.cpp)

find_package(OpenCV REQUIRED)
# include Paddle Lite
include_directories()

add_executable(mask_detector main.cpp)
target_link_libraries(mask_detector ${LIB_SRC})
