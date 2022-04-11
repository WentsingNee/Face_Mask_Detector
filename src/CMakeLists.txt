set(CMAKE_CXX_STANDARD 11)
project(FaceMaskDetection)

set(LIB_SRC
        bbox_tracker.cpp
        face_detector.cpp
        mask_detector.cpp)

find_package(OpenCV REQUIRED)
IF(OpenCV_FOUND)
    MESSAGE(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
ELSE(OpenCV_FOUND)
    MESSAGE(STATUS "OpenCV_LIBS not found!")
ENDIF(OpenCV_FOUND)
# include Paddle Lite
#include_directories()

add_executable(FaceMaskDetection main.cpp)
target_link_libraries(FaceMaskDetection ${LIB_SRC})
target_link_libraries(FaceMaskDetection ${OpenCV_LIBS})
