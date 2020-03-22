CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( testProgram )

SET(CMAKE_CXX_COMPILER "g++")
SET(CMAKE_BUILD_TYPE Debug)

# 增加opencv的依赖
unset(OpenCV_CONFIG_PATH CACHE)
unset(OpenCV_DIR CACHE)
#FIND_PACKAGE(OpenCV REQUIRED)
find_package(OpenCV HINTS /usr/local/opencv-2.4.11/share/OpenCV)
ADD_EXECUTABLE(faceDetection faceDet.cpp)
TARGET_LINK_LIBRARIES(faceDetection ${OpenCV_LIBS})
unset(OpenCV_CONFIG_PATH CACHE)
unset(OpenCV_DIR CACHE)

