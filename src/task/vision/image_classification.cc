#include "src/task/vision/image_classification.h"

int imageClassification::Init(std::string modelFilepath, std::string labelFilepath)
{
    instanceName_="image-classification-inference";
    modelFilepath_=modelFilepath;
    labelFilepath_=labelFilepath;
    InitCheck();
    labels_ = readLabels(labelFilepath_);
    return GetEngine()->Init(instanceName_, modelFilepath_);
}

std::string imageClassification::Postprocess()
{
    return postprocessor_.Postprocess(Infer(input_tensors_), labels_);
}

std::string imageClassification::Classify(cv::Mat &img_raw)
{   
    img_raw_ = img_raw;
    std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now();
    Preprocess(input_tensors_, img_raw_);
    std::chrono::steady_clock::time_point end =
    std::chrono::steady_clock::now();
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
    return Postprocess();
}

void imageClassification::InitCheck()
{
    if(!checkModelExtension(modelFilepath_)) {
        throw std::runtime_error("[ ERROR ] The ModelFilepath is not correct. Make sure you are setting the path to an onnx model file (.onnx)");
    }

    // Validate LabelFilePath
    if(!checkLabelFileExtension(labelFilepath_)) {
        throw std::runtime_error("[ ERROR ] The LabelFilepath is not set correctly and the labels file should end with extension .txt");
    }
}

bool imageClassification::checkModelExtension(const std::string& filename)
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
