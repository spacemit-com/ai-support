#include "object_detection.h"

void ObjectDetection::Detect()
{
    Init();
    Postprocess();
}

void ObjectDetection::Postprocess()
{
    std::cout<<"postprocess"<<std::endl;
    std::vector<int64_t> inputDims = GetEngine()->GetInputDims();
    std::vector<int64_t> outputDims = GetEngine()->GetOutputDims();
    postprocessor_.Postprocess(scaleparams_, 
                               Infer(input_tensors_), 
                               detected_boxes_, 
                               inputDims, 
                               outputDims, 
                               img_height_, 
                               img_width_);
}

void ObjectDetection::Init()
{
    instanceName_="object-detection-inference";
    modelFilepath_="/home/gexy5/Documents/BianbuAI/data/models/yolox_tiny.onnx";

    imageFilepath_="/home/gexy5/Documents/BianbuAI/data/imgs/4.jpg";
    labelFilepath_="/home/gexy5/Documents/BianbuAI/data/labels/synset.txt";
    labels_ = readLabels(labelFilepath_);
    GetEngine()->Init(instanceName_, modelFilepath_);
    std::vector<int64_t> inputDims = GetEngine()->GetInputDims();
    processor_.Preprocess(imageFilepath_, inputDims, input_tensors_, scaleparams_, CHW, img_height_, img_width_);
}