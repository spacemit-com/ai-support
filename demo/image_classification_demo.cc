#include "task/vision/image_classification_task.h"
#include "utils/time.h"

int main()
{
#ifdef DEBUG
    std::cout<<"."<<std::endl;
#endif
    std::string filePath = "/home/gexy5/Documents/bianbu-support/data/models/squeezenet1.1-7.onnx";
    std::string labelFilepath = "/home/gexy5/Documents/bianbu-support/data/labels/synset.txt";
    cv::Mat imgRaw;
    std::unique_ptr<imageClassificationTask> imageclassification = std::unique_ptr<imageClassificationTask>(new imageClassificationTask(filePath, labelFilepath));
    std::string modelFilepath = "/home/gexy5/Documents/bianbu-support/data/models/squeezenet1.1-7.onnx";
    {
#ifdef DEBUG
        TimeWatcher t("|-- Load input data");
#endif
        imgRaw = cv::imread("/home/gexy5/Documents/bianbu-support/data/imgs/pexels-photo-18852705.jpeg");
    }
    ImageClassificationResult result = imageclassification->Classify(imgRaw);
    std::cout<<result.label_text<<std::endl;
    return 0;
}