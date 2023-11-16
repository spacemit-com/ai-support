#include "task/vision/image_classification_task.h"

int main()
{
    imageClassificationTask imageclassification;
    std::string instanceName = "image-classification-inference";
    std::string modelFilepath = "/home/gexy5/Documents/bianbu-support/data/models/squeezenet1.1-7.onnx";
    cv::Mat imgRaw = cv::imread("/home/gexy5/Documents/bianbu-support/data/imgs/dog.jpeg");
    std::string labelFilepath = "/home/gexy5/Documents/bianbu-support/data/labels/synset.txt";
    std::string label = imageclassification.Classify(instanceName, modelFilepath, imgRaw, labelFilepath);
    std::cout<<label<<std::endl;
    return 0;
}