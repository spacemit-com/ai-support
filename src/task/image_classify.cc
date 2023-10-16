#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept> // To use runtime_error

#include "image_classify.h"
#include "task_api_factory.h"

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

void ImageClassify::Init()
{
    instanceName_="image-classification-inference";
    modelFilepath_="/home/gexy5/Documents/BianbuAI/data/models/squeezenet1.1-7.onnx";
    imageFilepath_="/home/gexy5/Documents/BianbuAI/data/imgs/demo.jpeg";
    labelFilepath_="/home/gexy5/Documents/BianbuAI/data/labels/synset.txt";
    InitCheck();
    labels_ = readLabels(labelFilepath_);
    GetEngine()->Init(instanceName_, modelFilepath_);
}

void ImageClassify::Postprocess()
{
    postprocessor_.Postprocess(Infer(input_tensors_), labels_);
}

void ImageClassify::Classify()
{
    Init();
    Preprocess(imageFilepath_, input_tensors_);
    Postprocess();
}

void ImageClassify::InitCheck()
{
    if(!checkModelExtension(modelFilepath_)) {
        throw std::runtime_error("[ ERROR ] The ModelFilepath is not correct. Make sure you are setting the path to an onnx model file (.onnx)");
    }

    // Validate LabelFilePath
    if(!checkLabelFileExtension(labelFilepath_)) {
        throw std::runtime_error("[ ERROR ] The LabelFilepath is not set correctly and the labels file should end with extension .txt");
    }
}

bool ImageClassify::checkModelExtension(const std::string& filename)
{
    if(filename.empty())
    {
        throw std::runtime_error("[ ERROR ] The Model file path is empty");
    }
    size_t pos = filename.rfind('.');
    if (pos == std::string::npos)
        return false;
    std::string ext = filename.substr(pos+1);
    if (ext == "onnx")
        return true;
    return false;
}
