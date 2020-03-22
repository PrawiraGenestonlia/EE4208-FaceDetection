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
string face_cascade_name = "../data/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::map<std::string, cv::Mat> trainedDataAvgFace;
std::map<std::string, cv::Mat> trainedDataEigenFace;
std::map<std::string, cv::Mat> trainedWeights;
std::map<std::string, cv::Mat> trainedEigenVectors;

int main()
{
    //
    std::cout << "EE4208 Dataset Training is running!" << std::endl;

    // Load the cascade
    if (!face_cascade.load(face_cascade_name))
    {
        std::cout << "--(!)Error loading!" << std::endl;
        return (-1);
    };

    // Variable to store images
    std::map<std::string, std::vector<std::string>> trainingSet = read_directory("../data/faceData");
    std::map<std::string, std::vector<cv::Mat>> images;

    // preprocess the images
    for (const auto &name : trainingSet)
    {
        for (uint i = 0; i < name.second.size(); i++)
        {
            std::cout << "[" << name.first << "] " << name.second[i] << std::endl;
            //
            cv::Mat singleImage = detectAndImage(name.second[i]);
            if (!singleImage.empty())
            {
                images[name.first].push_back(singleImage);
            }
        }
    }

    //for each person, pca analysis
    for (const auto &member : images)
    {
        // Create data matrix for PCA.
        Size sz = member.second[0].size();
        int numberOfImages = static_cast<int>(member.second.size());
        cv::Mat data = createDataMatrix(member.second);

        // Calculate PCA of the data matrix
        std::cout << "Calculating PCA for " << member.first << " with " << numberOfImages << " images" << std::endl;
        cv::PCA pca(data, Mat(), PCA::DATA_AS_COL);

        // Extract mean vector and reshape it to obtain average face
        cv::Mat averageFace = pca.mean.reshape(1, sz.height);

        cv::Mat reshapedAvgFace(averageFace);
        reshapedAvgFace.convertTo(reshapedAvgFace, CV_32F);
        cv::Mat mean_face(sz.height * sz.height, numberOfImages, CV_32F);
        cv::Mat C;

        for (int i = 0; i < numberOfImages; i++)
        {
            cv::Mat temp(member.second[i]);
            temp.convertTo(temp, CV_32F);
            cv::Mat currentMean(reshapedAvgFace * 0);
            currentMean = temp - reshapedAvgFace;
            cv::Mat temp2 = currentMean.reshape(1, 1).t();
            temp2.copyTo(mean_face.col(i));
        }

        C = mean_face.t() * mean_face / numberOfImages;

        cv::Mat EigenValues, EigenVectors;
        cv::eigen(C, EigenValues, EigenVectors);

        // EigenVectors = norm(EigenVectors);

        cv::Mat n2bestEigenVectors = mean_face * EigenVectors;

        cv::Mat weight = calcWeights(mean_face, n2bestEigenVectors, 1);

        // cv::Mat output = Mat(sz.height, sz.height, CV_32F, Scalar(0));
        cv::Mat output = averageFace.clone();
        output.convertTo(output, CV_32F);

        for (int i = 0; i < numberOfImages; i++)
        {
            cv::Mat temp(member.second[i]);
            temp = temp.reshape(1, 1).t();
            temp.convertTo(temp, CV_32F);
            cv::Mat weights_i = n2bestEigenVectors.t() * temp;
            cv::Mat eigenFace = n2bestEigenVectors * weights_i;
            eigenFace = eigenFace.reshape(1, 100);
            imshow("eigenFace", eigenFace);
            output += eigenFace;
        }

        averageFace.convertTo(averageFace, CV_8U);
        output.convertTo(output, CV_8U);
        trainedDataAvgFace[member.first] = averageFace;
        trainedDataEigenFace[member.first] = output;
        trainedWeights[member.first] = weight;
        trainedEigenVectors[member.first] = n2bestEigenVectors;
    }
    SaveTrainedData("../data", trainedDataAvgFace, trainedDataEigenFace);
    SaveMapMatrix("../data", "/trainedWeights.dat", trainedWeights);
    SaveMapMatrix("../data", "/trainedEigenVectors.dat", trainedEigenVectors);
    ShowImages(trainedDataAvgFace, trainedDataEigenFace);
    //wait for escape
    while (true)
    {
        if (waitKey(10) == 27)
        {
            break; // escape
        }
    }
    return 1;
}

void debugLog(std::string text)
{
#ifdef DEBUG
    std::cout << "[DEBUG] " << text << std::endl;
#endif
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

    cvtColor(readImage, frame_gray, cv::ColorConversionCodes::COLOR_BGR2GRAY);
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
        // sleep(2);
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
