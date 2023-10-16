#include "classification_postprocessor.h"

float ClassificationPostprocessor::division(float num, float den)
{
   if (den == 0) {
      throw std::runtime_error("[ ERROR ] Math error: Attempted to divide by Zero\n");
   }
   return (num / den);
}

void ClassificationPostprocessor::Postprocess(std::vector<Ort::Value> output_tensors, std::vector<std::string> labels)
{
    int predId = 0;
    float activation = 0;
    float maxActivation = std::numeric_limits<float>::lowest();
    float expSum = 0;
    /* The inference result could be found in the buffer for the output tensors, 
    which are usually the buffer from std::vector instances. */
    for (int i = 0; i < labels.size(); i++) {
        activation = output_tensors[0].At<float>({0,i});
        expSum += std::exp(activation);
        if (activation > maxActivation)
        {
            predId = i;
            maxActivation = activation;
        }
    }
    std::cout << "Predicted Label ID: " << predId << std::endl;
    std::cout << "Predicted Label: " << labels.at(predId) << std::endl;
    float result;
    try {
      result = division(std::exp(maxActivation), expSum);
      std::cout << "Uncalibrated Confidence: " << result << std::endl;
    }
    catch (std::runtime_error& e) {
      std::cout << "Exception occurred" << std::endl << e.what();
    }
}