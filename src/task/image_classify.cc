#include "image_classify.h"


void ImageClassify::Init(std::string instanceName, std::string modelFilepath, cv::Mat &img_raw, std::string labelFilepath)
{
    instanceName_=instanceName;
    modelFilepath_=modelFilepath;
    img_raw_ = img_raw;
    labelFilepath_=labelFilepath;
    InitCheck();
    labels_ = readLabels(labelFilepath_);
    GetEngine()->Init(instanceName_, modelFilepath_);
}

void ImageClassify::Postprocess()
{
    postprocessor_.Postprocess(Infer(input_tensors_), labels_);
}

void ImageClassify::Classify(std::string instanceName, std::string modelFilepath, cv::Mat &img_raw, std::string labelFilepath)
{   
    Init(instanceName, modelFilepath, img_raw, labelFilepath);
    std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now();
    Preprocess(img_raw_, input_tensors_);
    std::chrono::steady_clock::time_point end =
    std::chrono::steady_clock::now();
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
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
