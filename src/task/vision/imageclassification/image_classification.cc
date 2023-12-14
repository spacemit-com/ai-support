#include "src/task/vision/imageclassification/image_classification.h"
#include "utils/time.h"

int imageClassification::Init(const std::string modelFilepath, const std::string labelFilepath)
{
    instanceName_="image-classification-inference";
    modelFilepath_=modelFilepath;
    labelFilepath_=labelFilepath;
    InitCheck();
    labels_ = readLabels(labelFilepath_);
    return GetEngine()->Init(instanceName_, modelFilepath_);
}

void imageClassification::Preprocess(std::vector<float> &input_tensors,
          const cv::Mat& img_raw)
{
    auto input_dims = GetInputShape();
    preprocessor_.Preprocess(img_raw_, input_dims, input_tensors_);
}

ImageClassificationResult imageClassification::Postprocess()
{
    return postprocessor_.Postprocess(Infer(input_tensors_), labels_);
}



ImageClassificationResult imageClassification::Classify(const cv::Mat &img_raw)
{   
    img_raw_ = img_raw;
    {
#ifdef DEBUG
        std::cout<<"|-- Preprocess"<<std::endl;
        TimeWatcher t("|--");
#endif
        Preprocess(input_tensors_, img_raw_);
    }
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