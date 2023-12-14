#include "task/vision/image_classification_task.h"
#include "utils/time.h"

int main(int argc, char* argv[])
{
    std::string filePath, labelFilepath, imageFilepath;
    if(argc == 4)
    {
        filePath = argv[1];
        labelFilepath = argv[2];
        imageFilepath = argv[3];
    }
    else{
        std::cout<<"run with "<<argv[0]<<" <modelFilepath> <labelFilepath> <imageFilepath>" <<std::endl;
        return 0;
    }
    cv::Mat imgRaw;
    std::unique_ptr<imageClassificationTask> imageclassification = std::unique_ptr<imageClassificationTask>(new imageClassificationTask(filePath, labelFilepath));
#ifdef DEBUG
    std::cout<<"."<<std::endl;
#endif
    {
#ifdef DEBUG
        TimeWatcher t("|-- Load input data");
#endif
        imgRaw = cv::imread(imageFilepath);
    }
    ImageClassificationResult result = imageclassification->Classify(imgRaw);
    std::cout<<result.label_text<<std::endl;
    return 0;
}