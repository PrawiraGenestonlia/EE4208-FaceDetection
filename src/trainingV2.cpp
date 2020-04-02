#define DEBUG 1
#define NUM_EIGEN_FACES 5
#define MAX_SLIDER_VALUE 255

#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <experimental/filesystem>
#include <map>
#include <unistd.h>
#include <fstream>
#include "EE4208Utils.cpp"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

//function headers
cv::Mat detectAndImage(const std::string filePath);
void debugLog(std::string text);
cv::Mat createDataMatrix(const vector<Mat> &images);

//global variable
string face_cascade_name = "./data/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::map<std::string, cv::Mat> trainedDataAvgFace;
std::map<std::string, cv::Mat> trainedEigenVectors;
std::map<std::string, cv::Mat> trainedWeights;

int main(int argc, char *argv[])
{
  //
  std::cout << "EE4208 Dataset Training is running!" << std::endl;

  // Load the cascade
  if (!face_cascade.load(face_cascade_name))
  {
    std::cout << "--(!)Error loading!" << std::endl;
    return (-1);
  };

  //capture arguments
  bool isTest = false;
  for (int i = 0; i < argc; i++)
  {
    if (i == 1)
    {
      isTest = true;
    }
  }

  // Variable to store images
  std::map<std::string, std::string> trainingSet = read_directory_independent_image(isTest ? "./data/test" : "./data/faceDataIndependent");
  std::map<std::string, cv::Mat> images;

  // preprocess the images to find face
  for (const auto &name : trainingSet)
  {
    cv::Mat singleImage = detectAndImage(name.second);
    if (!singleImage.empty())
    {
      std::cout << "[" << name.first << "] " << name.second << std::endl;
      images[name.first] = singleImage;
      // usleep(2000000);
    }
  }

  //combine all images into a single matrix
  std::vector<cv::Mat> totalImages;
  for (const auto &member : images)
  {
    totalImages.push_back(member.second);
  }
  // Create data matrix for PCA.
  int numberOfImages = static_cast<int>(totalImages.size());
  cv::Mat data = createDataMatrix(totalImages);

  // Calculate PCA of the data matrix
  std::cout << "Calculating PCA for " << numberOfImages << " images" << std::endl;
  cv::PCA pca(data, Mat(), PCA::DATA_AS_COL);
  std::cout << "Completed calculating PCA..." << std::endl;
  // Extract mean vector and reshape it to obtain average face
  cv::Mat averageFace = pca.mean.reshape(1, 100);
  averageFace.convertTo(averageFace, CV_8U);
  cv::imshow("averageFace", averageFace);
  cv::waitKey(10);

  std::vector<cv::Mat> totalFaceZeroMean;
  cv::Mat mean_face(10000, numberOfImages, CV_32F);
  for (const auto &member : images)
  {
    static int i = 0;
    cv::Mat fi_fave = member.second.reshape(1, 1).t() - averageFace.reshape(1, 1).t();
    fi_fave.convertTo(fi_fave, CV_32F);
    fi_fave.copyTo(mean_face.col(i));
    //
    fi_fave = fi_fave.reshape(1, 100);
    fi_fave.convertTo(fi_fave, CV_8U);
    totalFaceZeroMean.push_back(fi_fave);
    i++;
  }
  mean_face = mean_face.t();

  cv::Mat data2 = createDataMatrix(totalFaceZeroMean);
  //[method 1] using in built PCA function to find eigenvectors
  cv::PCA pca2(data2, Mat(), PCA::DATA_AS_COL);
  cv::Mat EigenValues(pca2.eigenvalues);
  cv::Mat EigenVectors(pca2.eigenvectors);

  //[method 2] manual calculation of eigenvectors
  // cv::Mat Cov = mean_face * mean_face.t() / numberOfImages;
  // cv::eigen(Cov, EigenValues, EigenVectors);
  // EigenValues = mean_face.t() * EigenValues;
  // EigenVectors = mean_face.t() * EigenVectors;
  // EigenVectors = EigenVectors.t();
  // EigenVectors = norm(EigenVectors);

  std::cout
      << "[DEBUG] "
      << "eigen size: " << EigenValues.size() << std::endl;

  //calculate weight of the eigenvectors
  std::cout << "Calculating weight..." << std::endl;
  cv::Mat weight = calcWeightsReduced(mean_face, EigenVectors, numberOfImages, 1);
  std::cout << "Completed calculating weight..." << std::endl;

  //save data
  for (const auto &name : images)
  {
    static int i = 0;
    trainedWeights[name.first] = weight.col(i);
    i++;
  }
  trainedDataAvgFace["trained"] = averageFace.reshape(1, 100);
  trainedEigenVectors["trained"] = EigenVectors;
  std::cout << "Saving training sets..." << std::endl;
  SaveMapMatrix("./data", "/trainedWeights.dat", trainedWeights);
  SaveMapMatrix("./data", "/trainedEigenVectors.dat", trainedEigenVectors);
  SaveMapMatrix("./data", "/trainedAverageFace.dat", trainedDataAvgFace);

  return 1;
}

cv::Mat detectAndImage(const std::string filePath)
{
  Mat readImage;
  std::vector<Rect> faces;
  Mat frame_gray;
  Mat crop;
  Mat res;
  Mat gray;
  string text;
  stringstream sstm;

  readImage = imread(filePath, 1);

  cv::cvtColor(readImage, frame_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
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
    crop = readImage(roi_b);
    resize(crop, res, Size(100, 100), 0, 0, cv::InterpolationFlags::INTER_LINEAR); // This will be needed later while saving images
    cvtColor(res, gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);                 // Convert cropped image to Grayscale
    gray.convertTo(gray, CV_8U);
  }
  if (!crop.empty())
  {
    imshow("training", gray);
    cv::waitKey(10);
    return gray;
  }
  else
  {
    cv::Mat null;
    return null;
  }
}

cv::Mat createDataMatrix(const vector<Mat> &images)
{
  cv::Mat data(images[0].rows * images[0].cols, static_cast<int>(images.size()), CV_8U);
  // Turn an image into one row vector in the data matrix
  for (unsigned int i = 0; i < images.size(); i++)
  {
    // Extract image as one long vector of size w x h x 3
    cv::Mat image = images[i].reshape(1, 1).t();
    // Copy the long vector into one row of the destm
    image.copyTo(data.col(i));
  }
  return data;
}
