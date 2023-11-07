#include "task_api_factory.h"
#include "image_classify.h"

using namespace std;
int main()
{
    std::unique_ptr<ImageClassify> imageclassify = TaskAPIFactory::Create<ImageClassify>();
    std::string instanceName = "image-classification-inference";
    std::string modelFilepath = "/home/gexy5/Documents/bianbu-support/data/models/squeezenet1.1-7.onnx";
    cv::Mat img_raw = cv::imread("/home/gexy5/Documents/bianbu-support/data/imgs/demo.jpeg");
    std::string labelFilepath = "/home/gexy5/Documents/bianbu-support/data/labels/synset.txt";
    std::string label = imageclassify->Classify(instanceName, modelFilepath, img_raw, labelFilepath);
    std::cout<<label<<std::endl;
    return 0;
}