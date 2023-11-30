#include "src/task/vision/objectdetection/object_detection.h"

std::vector<Boxi> ObjectDetection::Inference(cv::Mat &raw_img)
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
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
#endif              

    return result_boxes_;
}

void ObjectDetection::Preprocess(std::vector<float> &input_tensors, cv::Mat& img_raw)
{
    processor_.Preprocess(img_raw, inputDims_, input_tensors_, CHW);
}

std::vector<Boxi> ObjectDetection::Detect_NanoDet(cv::Mat &raw_img)
{
    result_boxes_.clear();
    input_tensors_.clear();
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;

#ifdef DEBUG
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    processor_.Preprocess_NanoDet(raw_img, inputDims_, input_tensors_);

#ifdef DEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "|-- preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
#endif

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

#ifdef DEBUG
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    processor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);

#ifdef DEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
#endif

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

#ifdef DEBUG
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    processor_.Preprocess(raw_img, inputDims_, input_tensors_, HWC);

#ifdef DEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "preprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
#endif
            
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
