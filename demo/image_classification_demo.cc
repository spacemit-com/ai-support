#include "task/vision/image_classification_task.h"

int main()
{
    imageClassificationTask imageclassification;
    std::string modelFilepath = "/home/gexy5/Documents/bianbu-support/data/models/squeezenet1.1-7.onnx";
    cv::Mat imgRaw = cv::imread("/home/gexy5/Documents/bianbu-support/data/imgs/pexels-photo-18852705.jpeg");
    std::string labelFilepath = "/home/gexy5/Documents/bianbu-support/data/labels/synset.txt";
    int flag = imageclassification.Init(modelFilepath, labelFilepath);
    ImageClassificationResult result = imageclassification.Classify(imgRaw);
    std::cout<<result.label_text<<std::endl;
    return 0;
}