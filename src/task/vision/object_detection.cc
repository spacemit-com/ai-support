#include "src/task/vision/object_detection.h"

std::vector<Boxi> ObjectDetection::Detect(std::string &instanceName, std::string &modelFilepath, cv::Mat &raw_img)
{
    int flag = Init(instanceName, modelFilepath, raw_img);
    return Postprocess();
}

std::vector<Boxi> ObjectDetection::Postprocess()
{
    std::vector<int64_t> inputDims = GetEngine()->GetInputDims();
    postprocessor_.Postprocess(Infer(input_tensors_), 
                               result_boxes_, 
                               inputDims, 
                               img_height_, 
                               img_width_);
    return result_boxes_;
}

int ObjectDetection::Init(std::string &instanceName, std::string &modelFilepath, cv::Mat &raw_img)
{
    instanceName_= instanceName;
    modelFilepath_= modelFilepath;
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;
    int flag = GetEngine()->Init(instanceName_, modelFilepath_);
    auto inputDims = GetEngine()->GetInputDims();
    std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now();
    processor_.Preprocess(raw_img, inputDims, input_tensors_, CHW, img_height_, img_width_);
    std::chrono::steady_clock::time_point end =
    std::chrono::steady_clock::now();
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
    return flag;
}