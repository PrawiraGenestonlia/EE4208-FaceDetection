// #define DEBUG 1

#include <iostream>
#include <fstream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <map>
#include "EE4208Utils.cpp"

using namespace std;
using namespace cv;

//function headers
int openWebCam();
cv::Mat detectAndDisplay(Mat frame);
void debugLog(std::string text);
std::string faceRecognition(cv::Mat greyScaleFace);
std::string ncc(cv::Mat weight, std::map<std::string, cv::Mat> trainedWeights);

//global variable
string face_cascade_name = "../data/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber = 0; // Number of file to be saved
string filename;

//
std::map<std::string, cv::Mat> trainedDataAvgFace;
std::map<std::string, cv::Mat> trainedDataEigenFace;
std::map<std::string, cv::Mat> trainedWeights;
std::map<std::string, cv::Mat> trainedEigenVectors;

int main()
{
    std::cout << "The program is running!" << std::endl;
    //

    ReadTrainedData("../data", trainedDataAvgFace, trainedDataEigenFace);
    ReadMapMatrix("../data", "/trainedWeights.dat", trainedWeights);
    ReadMapMatrix("../data", "/trainedEigenVectors.dat", trainedEigenVectors);
    // ShowImages(trainedDataAvgFace, trainedDataEigenFace);
    //
    openWebCam();
}

int openWebCam()
{

    VideoCapture capture(0);
    //-- 2. Read the video stream
    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }

    // Load the cascade
    if (!face_cascade.load(face_cascade_name))
    {
        printf("--(!)Error loading\n");
        return (-1);
    };

    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //
        putText(frame, to_string(capture.get(CAP_PROP_FPS)), Point2f(10, 10), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar_(255, 0, 0), 1, CV_8U);
        //-- 3. Apply the classifier to the frame
        cv::Mat face = detectAndDisplay(frame);
        debugLog("5");
        int pixel;
        if (face.elemSize())
        {
            pixel = face.at<uchar>(10, 20);
            std::cout << "Pixel coordinate: (10, 20)= " << pixel << std::endl;
        }
        else
        {
            pixel = 0;
        }

        debugLog("6");
        if (waitKey(10) == 27)
        {
            break; // escape
        }
    }
    return 0;
}

cv::Mat detectAndDisplay(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    stringstream sstm;

    // std::cout << "[cv::ColorConversionCodes::COLOR_BGR2GRAY] " << cv::ColorConversionCodes::COLOR_BGR2GRAY << std::endl;

    cvtColor(frame, frame_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | 2, Size(30, 30));

    // Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;

    size_t ic = 0; // ic is index of current element
    int ac = 0;    // ac is area of current element

    size_t ib = 0; // ib is index of biggest element
    int ab = 0;    // ab is area of biggest element

    debugLog("1");
    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);

        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);

        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }
        std::cout << "Face location:"
                  << "(" << roi_c.x << ", " << roi_c.y << ")" << std::endl;
        crop = frame(roi_b);
        resize(crop, res, Size(100, 100), 0, 0, cv::InterpolationFlags::INTER_LINEAR); // This will be needed later while saving images
        cvtColor(res, gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);                 // Convert cropped image to Grayscale

        //Face recognition
        std::string matchedName = faceRecognition(gray);

        // Form a filename
        filename = "";
        stringstream ssfn;
        ssfn << filenumber << ".png";
        filename = ssfn.str();

        imwrite(filename, gray);

        Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
        putText(frame, matchedName, pt1, cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar_(255, 0, 0), 1, CV_8U);
    }
    debugLog("2");

    // Show image
    sstm << "Crop area size: " << roi_b.width << "x" << roi_b.height << " Filename: " << filename;
    text = sstm.str();
    // putText(frame, text, cvPoint(30, 30), cv::HersheyFonts::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
    imshow("original", frame);
    /*
    int currentFileNumber = 1;
    if (currentFileNumber <= filenumber)
    {
        filename = "";
        stringstream ssfn;
        ssfn << "../output/" << currentFileNumber << "_1.png";
        filename = ssfn.str();
        currentFileNumber++;
        imwrite(filename, frame);
    }

    debugLog("3");
    if (!crop.empty())
    {
        debugLog("4.1");
        imshow("detected", crop);
        debugLog("4.2");
    }
    else
    {
        // destroyWindow("detected");
    }
    */
    return gray;
}

void debugLog(std::string text)
{
#ifdef DEBUG
    std::cout << "[Pass here] " << text << std::endl;
#endif
}

std::string faceRecognition(cv::Mat greyScaleFace)
{
    cv::Mat inputImage(greyScaleFace);
    inputImage.convertTo(inputImage, CV_32F);
    cv::Mat C = inputImage.t() * inputImage;

    cv::Mat EigenValues, EigenVectors;
    cv::eigen(C, EigenValues, EigenVectors);

    cv::Mat n2bestEigenVectors = inputImage * EigenVectors;

    cv::Mat weight = calcWeights(inputImage, n2bestEigenVectors, 0);

    return ncc(weight, trainedWeights);
}

std::string ncc(cv::Mat weight, const std::map<std::string, cv::Mat> trainedWeights)
{
    std::string output = "unknown";
    float lowest_err = 1000000000000;
    for (const auto &member : trainedWeights)
    {
        float current_err = 0;
        cv::Mat savedWeights = member.second;

        for (int i = 0; i < savedWeights.rows; i++)
        {
            current_err += norm(weight - savedWeights.row(i).reshape(1, 100));
        };
        if (current_err < lowest_err)
        {
            output = member.first;
            lowest_err = current_err;
        }
        std::cout << "checking: " << member.first << " ===== error: " << current_err << std::endl;
    }
    std::cout << "Matched to: " << output
              << "\nWith error of: " << lowest_err
              << std::endl;
    return output;
}