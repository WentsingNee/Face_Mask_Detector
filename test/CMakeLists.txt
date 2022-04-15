find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
find_package (OpenCV REQUIRED)
add_definitions (-DBOOST_TEST_DYN_LINK)
set (PADDLE_LITE_DIR
        ${FaceMaskDetector_SOURCE_DIR}/lib/paddle-lite)
include_directories (${FaceMaskDetector_SOURCE_DIR}/src
        ${Boost_INCLUDE_DIRS}
        ${PADDLE_LITE_DIR}/cxx/include
)
link_directories (${PADDLE_LITE_DIR}/cxx/lib)

add_executable (FaceDetectorTest face_detector_tests.cpp)
target_link_libraries (FaceDetectorTest
        FaceMaskDetectionLib
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${OpenCV_LIBS}
        )

add_executable (MaskDetectorTest mask_detector_tests.cpp)
target_link_libraries(MaskDetectorTest
        FaceMaskDetectionLib
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        ${OpenCV_LIBS}
        )