#include "src/processor/classification_postprocessor.h"

float ClassificationPostprocessor::division(float num, float den)
{
   if (den == 0) {
      throw std::runtime_error("[ ERROR ] Math error: Attempted to divide by Zero\n");
   }
   return (num / den);
}

ImageClassificationResult ClassificationPostprocessor::Postprocess(std::vector<Ort::Value> output_tensors, std::vector<std::string> labels)
{
#ifdef DEBUG
    std::chrono::steady_clock::time_point begin =
    std::chrono::steady_clock::now();
#endif
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
#ifdef DEBUG
    std::chrono::steady_clock::time_point end =
    std::chrono::steady_clock::now();
    std::cout << "postprocess Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
#endif
    ImageClassificationResult result;
    std::chrono::steady_clock::time_point tp = std::chrono::steady_clock::now();
    result.timestamp = tp.time_since_epoch().count();
    result.label = predId;
    result.label_text = labels.at(predId);
    result.score = std::exp(maxActivation)/expSum;
    return result;
}