#include "src/task/vision/objectdetection/object_detection.h"
#include "src/utils/json.hpp"
#include "utils/time.h"

#include <fstream>
using json = nlohmann::json;

void ObjectDetection::Preprocess(std::vector<float> &input_tensors, const cv::Mat& img_raw)
{
    processor_.Preprocess(img_raw, inputDims_, input_tensors_, CHW);
}

ObjectDetectionResult ObjectDetection::Detect_NanoDet(const cv::Mat &raw_img)
{
    result_boxes_.clear();
    input_tensors_.clear();
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;

    {
#ifdef DEBUG
        std::cout << "|-- Preprocess "<<std::endl;
        TimeWatcher t("|--");
#endif
        processor_.Preprocess_NanoDet(raw_img, inputDims_, input_tensors_);
    }

    postprocessor_.Postprocess_NanoDet(Infer(input_tensors_),
        result_boxes_,
        inputDims_,
        img_height_, 
        img_width_,
        labels_);

    result_.result_bboxes = result_boxes_;
    return result_;
}

ObjectDetectionResult ObjectDetection::Detect_Yolov6(const cv::Mat &raw_img)
{
    result_boxes_.clear();
    input_tensors_.clear();
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;
    {
#ifdef DEBUG
        std::cout << "|-- Preprocess"<<std::endl;
        TimeWatcher t("|--");
#endif
        processor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);
    }
    postprocessor_.Postprocess_Yolov6(Infer(input_tensors_),
            result_boxes_,
            inputDims_,
            img_height_, 
            img_width_,
            labels_);

    result_.result_bboxes = result_boxes_;
    return result_;
}

ObjectDetectionResult ObjectDetection::Detect(const cv::Mat &raw_img)
{
    result_boxes_.clear();
    input_tensors_.clear();
    img_height_ = raw_img.rows;
    img_width_ = raw_img.cols;
    {
#ifdef DEBUG
        std::cout << "|-- Preprocess"<<std::endl;
        TimeWatcher t("|--");
#endif
        processor_.Preprocess(raw_img, inputDims_, input_tensors_, HWC);
    }
    postprocessor_.Postprocess(Infer(input_tensors_), 
                        result_boxes_, 
                        inputDims_, 
                        img_height_, 
                        img_width_,
                        labels_);

    result_.result_bboxes = result_boxes_;
    return result_;
}
ObjectDetectionResult ObjectDetection::Postprocess()
{
    postprocessor_.Postprocess(Infer(input_tensors_), 
                               result_boxes_, 
                               inputDims_, 
                               img_height_, 
                               img_width_,
                               labels_);

    result_.result_bboxes = result_boxes_;
    return result_;
}

int ObjectDetection::Initfromcommand(std::string &modelFilepath, std::string &labelFilepath)
{
    instanceName_ = "object-detection-inference";
    modelFilepath_ = modelFilepath;
    labelFilepath_ = labelFilepath;
    int flag = GetEngine()->Init(instanceName_, modelFilepath_);
    inputDims_ = GetEngine()->GetInputDims();
    labels_ = readLabels(labelFilepath_);
    return flag;
}

int ObjectDetection::Initfromconfig(std::string &configFilepath)
{
    std::ifstream f(configFilepath);
    json config = json::parse(f);
    int flag = GetEngine()->Init(config);
    labelFilepath_ = config["label_path"];
    inputDims_ = GetEngine()->GetInputDims();
    labels_ = readLabels(labelFilepath_);
    return flag;
}