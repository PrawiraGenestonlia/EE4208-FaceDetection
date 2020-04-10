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

//
int ERR_THRESHOLD = 100000; //4000

//function headers
int openWebCam();
cv::Mat detectAndDisplay(Mat frame);
void debugLog(std::string text);
std::string faceRecognition(cv::Mat greyScaleFace);
std::string ncc(cv::Mat weight, std::map<std::string, cv::Mat> trainedWeights);
std::string nccWithGrouping(cv::Mat weight, const std::map<std::string, std::map<std::string, cv::Mat>> trainedWeightsWithName);
void obtainRestructureWeights();

//global variable
string face_cascade_name = "./data/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
string window_name = "EE4208 - Face detection";
int filenumber = 0; // Number of file to be saved
string filename;

//
std::map<std::string, cv::Mat> trainedWeights;
std::map<std::string, cv::Mat> trainedDataAvgFace;
std::map<std::string, cv::Mat> trainedEigenVectors;
std::map<std::string, std::map<std::string, cv::Mat>> trainedWeightsWithName;
//

int main(int argc, char *argv[])
{
  std::cout << "Face Detection program is running!" << std::endl;

  //load error threshold
  for (int i = 0; i < argc; i++)
  {
    if (i == 1)
    {
      std::string in(argv[i]);

      ERR_THRESHOLD = std::stoi(in) ? std::stoi(in) : 3500;
    }
  }
  std::cout << "Error threshold: " << ERR_THRESHOLD << std::endl;

  // Load the cascade
  if (!face_cascade.load(face_cascade_name))
  {
    printf("--(!)Error loading\n");
    return (-1);
  };

  // Load all the trained files
  ReadMapMatrix("./data", "/trainedWeights.dat", trainedWeights);
  ReadMapMatrix("./data", "/trainedAverageFace.dat", trainedDataAvgFace);
  ReadMapMatrix("./data", "/trainedEigenVectors.dat", trainedEigenVectors);
  obtainRestructureWeights();
  // Main program
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

  Mat frame;
  while (capture.read(frame))
  {
    if (frame.empty())
    {
      cout << "--(!) No captured frame -- Break!\n";
      break;
    }
    //-- 3. Apply the classifier to the frame
    cv::Mat face = detectAndDisplay(frame);

    /*
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
    */

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

  cvtColor(frame, frame_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
  equalizeHist(frame_gray, frame_gray);

  // Detect faces
  face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | 2, Size(100, 100)); //min size (100,100)

  // Set Region of Interest
  cv::Rect roi_b;
  cv::Rect roi_c;

  size_t ic = 0; // ic is index of current element
  int ac = 0;    // ac is area of current element

  size_t ib = 0; // ib is index of biggest element
  int ab = 0;    // ab is area of biggest element

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
    // std::cout << "Face location:" << "(" << roi_c.x << ", " << roi_c.y << ")" << std::endl;
    crop = frame(roi_b);
    resize(crop, res, Size(100, 100), 0, 0, cv::InterpolationFlags::INTER_LINEAR); // This will be needed later while saving images
    cvtColor(res, gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);                 // Convert cropped image to Grayscale

    //Face recognition
    std::string matchedName = faceRecognition(gray);

    Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
    Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
    rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
    putText(frame, matchedName, pt1, cv::HersheyFonts::FONT_HERSHEY_DUPLEX, 1, Scalar_(0, 0, 255), 2, CV_8U);

    /*
    // Form a filename
    filename = "";
    stringstream ssfn;
    ssfn << filenumber << ".png";
    filename = ssfn.str();
    imwrite(filename, gray);
    */

    // Show image
    cv::imshow(matchedName, res);
    cv::waitKey(10);
  }
  cv::imshow(window_name, frame);
  cv::waitKey(10);
  return gray;
}

std::string faceRecognition(cv::Mat greyScaleFace)
{
  cv::Mat inputImage(greyScaleFace);
  inputImage.convertTo(inputImage, CV_32F);
  inputImage = inputImage - trainedDataAvgFace["trained"];
  cv::Mat weight = calcWeightsReduced(inputImage.reshape(1, 1), trainedEigenVectors["trained"], 1, 0);

  return nccWithGrouping(weight, trainedWeightsWithName);
}

void obtainRestructureWeights()
{
  for (const auto &member : trainedWeights)
  {
    trainedWeightsWithName[obtainName(member.first)][member.first] = member.second;
  }
}

std::string ncc(cv::Mat weight, const std::map<std::string, cv::Mat> trainedWeights)
{
  std::string output = "unknown";
  float lowest_err = 100000000;
  for (const auto &member : trainedWeights)
  {
    cv::Mat savedWeights = member.second;
    float current_err = norm(weight - savedWeights, NORM_L2);
    if (current_err < lowest_err)
    {
      output = member.first;
      lowest_err = current_err;
    }
    std::cout << "checking: " << member.first << " ===== error: " << current_err << std::endl;
  }
  std::cout << "----------------------------------------------------------------" << std::endl;
  std::cout << "Matched to: " << output << " With error of: " << lowest_err << std::endl;
  std::cout << "▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔" << std::endl;
  if (lowest_err <= ERR_THRESHOLD)
  {
    return splitString(output);
  }
  else
  {
    return "unknown";
  }
}

std::string nccWithGrouping(cv::Mat weight, const std::map<std::string, std::map<std::string, cv::Mat>> trainedWeightsWithName)
{
  std::string output = "unknown";
  float lowest_err = 100000000;
  for (const auto &member : trainedWeightsWithName)
  {
    float current_err = 0;
    int numOfImages = 0;
    for (const auto &member2 : member.second)
    {
      numOfImages++;
      cv::Mat savedWeights = member2.second;
      current_err += norm(weight - savedWeights, NORM_L2);
    }
    current_err = current_err / numOfImages;
    if (current_err < lowest_err)
    {
      output = member.first;
      lowest_err = current_err;
    }
    std::cout << "checking: " << member.first << " (" << numOfImages << ") => error: " << current_err << std::endl;
  }
  std::cout << "..." << std::endl;
  std::string details = "";
  float lowest_err_details = 100000000;
  for (const auto &member : trainedWeightsWithName)
  {
    if (member.first == output)
    {
      for (const auto &member2 : member.second)
      {
        cv::Mat savedWeights = member2.second;
        float current_err = norm(weight - savedWeights, NORM_L2);
        if (current_err < lowest_err_details)
        {
          details = member2.first;
          lowest_err_details = current_err;
        }
        std::cout << "checking: " << member2.first << " ===== error: " << current_err << std::endl;
      }
      std::cout << "----------------------------------------------------------------" << std::endl;
      std::cout << "Matched to: " << details << " With error of: " << lowest_err_details << std::endl;
      std::cout << "▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔" << std::endl;
    }
  }
  if (lowest_err_details <= ERR_THRESHOLD)
  {
    return splitString(details);
  }
  else
  {
    return "unknown";
  }
}
