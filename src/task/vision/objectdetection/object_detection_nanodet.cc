#include "src/task/vision/objectdetection/object_detection_nanodet.h"


std::vector<Boxi> ObjectDetectionNanodet::Postprocess()
{
    postprocessor_.Postprocess_NanoDet(Infer(input_tensors_),
                                        result_boxes_,
                                        inputDims_,
                                        img_height_, 
                                        img_width_);
    return result_boxes_;
}

int ObjectDetectionNanodet::Init(std::string &modelFilepath)
{
    instanceName_= "object-detection-inference";
    modelFilepath_= modelFilepath;
    int flag = GetEngine()->Init(instanceName_, modelFilepath_);
    inputDims_ = GetEngine()->GetInputDims();
    return flag;
}

void ObjectDetectionNanodet::Preprocess(std::vector<float> &input_tensors, cv::Mat& img_raw)
{
    preprocessor_.Preprocess_NanoDet(img_raw, inputDims_, input_tensors_);
}

std::vector<Boxi> ObjectDetectionNanodet::Detect(cv::Mat &raw_img)
{
    result_boxes_.clear();
    input_tensors_.clear();
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;

#ifdef DEBUG
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    Preprocess(input_tensors_, raw_img);

#ifdef DEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "|-- preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
#endif

    return Postprocess();
}

