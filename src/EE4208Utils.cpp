#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <experimental/filesystem>
#include <map>
#include <unistd.h>
#include <fstream>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

// List of available functions
void ShowImages(std::map<std::string, cv::Mat> trainedDataAvgFace, std::map<std::string, cv::Mat> trainedDataEigenFace);
void SaveTrainedData(const std::string filePath, std::map<std::string, cv::Mat> trainedDataAvgFace, std::map<std::string, cv::Mat> trainedDataEigenFace);
std::map<std::string, std::vector<std::string>> read_directory(const std::string &path);
std::map<std::string, std::string> read_directory_independent_image(const std::string &path);
void ReadTrainedData(const std::string filePath, std::map<std::string, cv::Mat> &trainedDataAvgFace, std::map<std::string, cv::Mat> &trainedDataEigenFace);
cv::Mat calcWeightsReduced(cv::Mat mean_face, cv::Mat u, int numberOfImages, bool bIsTraining);
std::string splitString(std::string inputString);

void ShowImages(std::map<std::string, cv::Mat> trainedDataAvgFace, std::map<std::string, cv::Mat> trainedDataEigenFace)
{
  // destroyAllWindows();
  // DRAW
  cv::Mat win_mat(cv::Size(trainedDataAvgFace.size() * 100, 230), CV_8U, Scalar(0));
  // cv::Mat win_mat(cv::Size(2000, 230), CV_8U, Scalar(0));

  for (const auto &member : trainedDataAvgFace)
  {
    static int i = 0;
    cv::Mat image = member.second;
    image.copyTo(win_mat(cv::Rect(i * 100, 30, 100, 100)));
    putText(win_mat, member.first, Point2f(i * 100, 25), cv::HersheyFonts::FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1, CV_8U);
    i++;
  }
  for (const auto &member : trainedDataEigenFace)
  {
    static int i = 0;
    cv::Mat image = member.second;
    image.copyTo(win_mat(cv::Rect(100 * i, 130, 100, 100)));
    i++;
  }
  cv::imshow("Trained Data", win_mat);
  cv::waitKey(10);
}

void SaveTrainedData(const std::string filePath, std::map<std::string, cv::Mat> trainedDataAvgFace, std::map<std::string, cv::Mat> trainedDataEigenFace)
{
  std::string path1 = filePath;
  path1.append("/trainedDataAvgFace.dat");
  ofstream fout1(path1);
  if (!fout1)
  {
    cout << "File Not Opened" << endl;
    return;
  }

  for (const auto &member : trainedDataAvgFace)
  {

    cv::Mat image = member.second;
    image.convertTo(image, CV_8U);
    // image.reshape(1, 1);
    fout1 << member.first << " ";
    for (int i = 0; i < image.rows; i++)
    {
      for (int j = 0; j < image.cols; j++)
      {
        stringstream ss;
        ss << image.row(i).col(j);
        std::string temp = ss.str();
        temp.erase(0, 1);
        temp.erase(temp.size() - 1);
        fout1 << temp << " ";
      }
    }
    fout1 << endl;
  }
  fout1.close();

  std::string path2 = filePath;
  path2.append("/trainedDataEigenFace.dat");
  ofstream fout2(path2);
  if (!fout2)
  {
    cout << "File Not Opened" << endl;
    return;
  }

  for (const auto &member : trainedDataEigenFace)
  {
    cv::Mat image = member.second;
    // image.convertTo(image, CV_8U);
    // image.reshape(1, 1);
    fout2 << member.first << " ";
    for (int i = 0; i < image.rows; i++)
    {
      for (int j = 0; j < image.cols; j++)
      {
        stringstream ss;
        ss << image.row(i).col(j);
        std::string temp = ss.str();
        temp.erase(0, 1);
        temp.erase(temp.size() - 1);
        fout2 << temp << " ";
      }
    }
    fout2 << endl;
  }
  fout2.close();
}

void SaveMapMatrix(const std::string filePath, const std::string filename, std::map<std::string, cv::Mat> trainedWeights)
{

  std::string path = filePath;
  path.append(filename);
  std::cout << "Saving " << filename << "..." << std::endl;
  ofstream fout(path);
  if (!fout)
  {
    cout << "File Not Opened" << endl;
    return;
  }

  for (const auto &member : trainedWeights)
  {

    cv::Mat weight = member.second;
    fout << member.first << " ";
    fout << weight.rows << " ";
    fout << weight.cols << " ";
    for (int i = 0; i < weight.rows; i++)
    {
      for (int j = 0; j < weight.cols; j++)
      {
        stringstream ss;
        ss << weight.row(i).col(j);
        std::string temp = ss.str();
        temp.erase(0, 1);
        temp.erase(temp.size() - 1);
        fout << temp << " ";
      }
    }
    fout << endl;
  }
  fout.close();
}

std::map<std::string, std::vector<std::string>> read_directory(const std::string &path)
{
  std::map<std::string, std::vector<std::string>> results;
  for (const auto &name : fs::directory_iterator(path))
  {
    if (fs::is_directory(name))
    {
      for (const auto &trainingFile : fs::directory_iterator(name))
      {
        if (trainingFile.path().extension() == ".jpg" || trainingFile.path().extension() == ".png")
        {
          results[name.path().filename()].push_back(trainingFile.path());
        }
      }
    }
  }
  return results;
}

std::map<std::string, std::string> read_directory_independent_image(const std::string &path)
{
  std::map<std::string, std::string> results;
  for (const auto &name : fs::directory_iterator(path))
  {
    if (fs::is_directory(name))
    {
      for (const auto &trainingFile : fs::directory_iterator(name))
      {
        if (trainingFile.path().extension() == ".jpg" || trainingFile.path().extension() == ".png")
        {
          results[trainingFile.path().filename()] = (trainingFile.path());
        }
      }
    }
  }
  return results;
}

void ReadTrainedData(const std::string filePath, std::map<std::string, cv::Mat> &trainedDataAvgFace, std::map<std::string, cv::Mat> &trainedDataEigenFace)
{
  std::cout << "Reading trained data..." << std::endl;
  std::string trainedDataAvgFacePath = filePath;
  std::string trainedDataEigenFacePath = filePath;
  trainedDataAvgFacePath.append("/trainedDataAvgFace.dat");
  trainedDataEigenFacePath.append("/trainedDataEigenFace.dat");

  //
  std::ifstream trainedDataAvgFaceFile(trainedDataAvgFacePath.c_str());

  if (!trainedDataAvgFaceFile.is_open())
  {
    cout << "Path Wrong!!!!" << endl;
    exit(0);
    return;
  }

  std::string line, name;

  while (std::getline(trainedDataAvgFaceFile, line))
  {
    // Create a stringstream of the current line
    std::stringstream ss(line);
    float val;
    std::string sval;
    std::vector<float> readArray;

    if (ss >> sval)
    {
      name = sval;
      if (ss.peek() == ' ')
        ss.ignore();
    }
    // Extract each integer
    while (ss >> val)
    {
      readArray.push_back(val);
      if (ss.peek() == ' ')
        ss.ignore();
    }

    cv::Mat image(100, 100, CV_8U, Scalar(0));
    int index = 0;
    for (int i = 0; i < image.rows; i++)
    {
      for (int j = 0; j < image.cols; j++)
      {
        image.row(i).col(j) = readArray[index];
        index++;
      }
    }
    image.convertTo(image, CV_8U);
    trainedDataAvgFace[name] = image;
  }
  trainedDataAvgFaceFile.close();

  //
  std::ifstream trainedDataEigenFaceFile(trainedDataEigenFacePath.c_str());

  if (!trainedDataEigenFaceFile.is_open())
  {
    cout << "Path Wrong!!!!" << endl;
    exit(0);
    return;
  }

  while (std::getline(trainedDataEigenFaceFile, line))
  {
    // Create a stringstream of the current line
    std::stringstream ss(line);
    float val;
    std::string sval;
    std::vector<float> readArray;

    if (ss >> sval)
    {
      name = sval;
      if (ss.peek() == ' ')
        ss.ignore();
    }
    // Extract each integer
    while (ss >> val)
    {
      readArray.push_back(val);
      if (ss.peek() == ' ' || ss.peek() == ' ')
        ss.ignore();
    }

    cv::Mat image(100, 100, CV_8U, Scalar(0));
    int index = 0;
    for (int i = 0; i < image.rows; i++)
    {
      for (int j = 0; j < image.cols; j++)
      {
        image.row(i).col(j) = readArray[index];
        index++;
      }
    }
    trainedDataEigenFace[name] = image;
  }
  trainedDataEigenFaceFile.close();
}

void ReadMapMatrix(const std::string filePath, const std::string filename, std::map<std::string, cv::Mat> &trainedWeights)
{
  std::string path = filePath;
  path.append(filename);
  std::cout << "Reading " << filename << "..." << std::endl;
  //
  std::ifstream inputFile(path.c_str());

  if (!inputFile.is_open())
  {
    cout << "Path Wrong!!!!" << endl;
    exit(0);
    return;
  }

  std::string line, name;

  while (std::getline(inputFile, line))
  {
    // Create a stringstream of the current line
    std::stringstream ss(line);
    float val;
    int ival;
    std::string sval;
    std::vector<float> readArray;
    int inputRow, inputCol;
    if (ss >> sval)
    {
      name = sval;
      if (ss.peek() == ' ')
        ss.ignore();
    }
    if (ss >> ival)
    {
      inputRow = ival;
      if (ss.peek() == ' ')
        ss.ignore();
    }
    if (ss >> ival)
    {
      inputCol = ival;
      if (ss.peek() == ' ')
        ss.ignore();
    }
    // Extract each integer
    while (ss >> val)
    {
      readArray.push_back(val);
      if (ss.peek() == ' ')
        ss.ignore();
    }

    cv::Mat output(inputRow, inputCol, CV_32F, Scalar(0));
    int index = 0;
    for (int i = 0; i < inputRow; i++)
    {
      for (int j = 0; j < inputCol; j++)
      {
        output.row(i).col(j) = readArray[index];
        index++;
      }
    }
    trainedWeights[name] = output;
  }
  inputFile.close();
}

cv::Mat calcWeights(cv::Mat mean_face, cv::Mat u, bool bIsTraining)
{
  // cv::Mat output(10000, mean_face.cols, CV_32F);
  cv::Mat output(mean_face.cols, 10000, CV_32F);
  if (bIsTraining)
  {
    for (int i = 0; i < mean_face.cols; i++)
    {
      cv::Mat weights_i = u.t() * mean_face.col(i);
      weights_i.copyTo(output.col(i));
      //visualise
      if (0)
      {
        cv::Mat eigenFace = u * weights_i;
        eigenFace = eigenFace.reshape(1, 100);
        eigenFace.convertTo(eigenFace, CV_8U);
        imshow("training", eigenFace);
        cv::waitKey(10);
      }
    }
    return output;
  }
  else
  {
    cv::Mat input_face = mean_face;
    cv::Mat weights_i = u.t() * input_face;
    weights_i.copyTo(output);
    //visualise
    if (0)
    {
      cv::Mat eigenFace = u * weights_i;
      eigenFace = eigenFace.reshape(1, 100);
      eigenFace.convertTo(eigenFace, CV_8U);
      imshow("calculated weight", eigenFace);
      cv::waitKey(10);
    }
    return output;
  }
}

cv::Mat calcWeightsReduced(cv::Mat mean_face, cv::Mat u, int numberOfImages, bool bIsTraining)
{
  cv::Mat output(numberOfImages, numberOfImages, CV_32F);
  if (bIsTraining)
  {
    for (int i = 0; i < mean_face.rows; i++)
    {
      cv::Mat weights_i = u * mean_face.row(i).t();
      weights_i.copyTo(output.col(i));
      //visualise
      if (0)
      {
        cv::Mat eigenFace = u * weights_i;
        eigenFace = eigenFace.reshape(1, 100);
        eigenFace.convertTo(eigenFace, CV_8U);
        imshow("training", eigenFace);
        cv::waitKey(10);
      }
    }
    return output;
  }
  else
  {
    cv::Mat input_face = mean_face;

    cv::Mat weights_i = u * input_face.t();
    weights_i.copyTo(output);
    //visualise
    if (0)
    {
      cv::Mat eigenFace = u * weights_i;
      eigenFace = eigenFace.reshape(1, 100);
      eigenFace.convertTo(eigenFace, CV_8U);
      imshow("calculated weight", eigenFace);
      cv::waitKey(10);
    }
    return output;
  }
}

std::string splitString(std::string inputString)
{
  std::string ss = "";
  for (int i = 0; i < inputString.size(); i++)
  {
    if (inputString[i] == '.')
    {
      ss += "]";
      break;
    }
    else if (inputString[i] == '_')
    {
      ss += " [";
    }
    else
    {
      ss += inputString[i];
    }
  }
  return ss;
}

std::string obtainName(std::string inputString)
{
  std::string ss = "";
  for (int i = 0; i < inputString.size(); i++)
  {
    if (inputString[i] == '_')
    {
      return ss;
    }
    ss += inputString[i];
  }
  return ss;
}