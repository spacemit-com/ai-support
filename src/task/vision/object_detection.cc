#include "src/task/vision/object_detection.h"

std::vector<Boxi> ObjectDetection::Inference(cv::Mat &raw_img)
{
    result_boxes_.clear();
    input_tensors_.clear();
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;
    std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now();
    processor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);
    std::chrono::steady_clock::time_point end =
    std::chrono::steady_clock::now();
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
    return result_boxes_;
}

std::vector<Boxi> ObjectDetection::Detect_NanoDet(cv::Mat &raw_img)
{
    result_boxes_.clear();
    input_tensors_.clear();
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;
    std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now();
    processor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);
    std::chrono::steady_clock::time_point end =
    std::chrono::steady_clock::now();
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
    postprocessor_.Postprocess_NanoDet(Infer(input_tensors_),
        result_boxes_,
        inputDims_,
        img_height_, 
        img_width_);
    return result_boxes_;
}

std::vector<Boxi> ObjectDetection::Detect_Yolov6(cv::Mat &raw_img)
{
    result_boxes_.clear();
    input_tensors_.clear();
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;
    std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now();
    processor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);
    std::chrono::steady_clock::time_point end =
    std::chrono::steady_clock::now();
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;

    postprocessor_.Postprocess_Yolov6(Infer(input_tensors_),
            result_boxes_,
            inputDims_,
            img_height_, 
            img_width_);
    return result_boxes_;
}

std::vector<Boxi> ObjectDetection::Detect(cv::Mat &raw_img)
{
    result_boxes_.clear();
    input_tensors_.clear();
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    processor_.Preprocess(raw_img, inputDims_, input_tensors_, HWC);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
    postprocessor_.Postprocess(Infer(input_tensors_), 
                        result_boxes_, 
                        inputDims_, 
                        img_height_, 
                        img_width_);
    return result_boxes_;
}

std::vector<Boxi> ObjectDetection::Postprocess()
{
    postprocessor_.Postprocess(Infer(input_tensors_), 
                               result_boxes_, 
                               inputDims_, 
                               img_height_, 
                               img_width_);
    return result_boxes_;
}

int ObjectDetection::Init(std::string &modelFilepath)
{
    instanceName_= "object-detection-inference";
    modelFilepath_= modelFilepath;
    int flag = GetEngine()->Init(instanceName_, modelFilepath_);
    inputDims_ = GetEngine()->GetInputDims();
    return flag;
}
