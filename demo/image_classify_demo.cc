#include "task_api_factory.h"
#include "image_classify.h"

using namespace std;
int main()
{
    std::unique_ptr<ImageClassify> imageclassify = TaskAPIFactory::Create<ImageClassify>();
    std::string instanceName = "image-classification-inference";
    std::string modelFilepath = "/home/geduer/Documents/BianbuAI/data/models/squeezenet1.1-7.onnx";
    cv::Mat img_raw = cv::imread("/home/geduer/Documents/BianbuAI/data/imgs/demo.jpeg");
    std::string labelFilepath = "/home/geduer/Documents/BianbuAI/data/labels/synset.txt";
    imageclassify->Classify(instanceName, modelFilepath, img_raw, labelFilepath);
    return 0;
}