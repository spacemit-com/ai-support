#include <memory>

#include "task_api_factory.h"
#include "object_detection.h"

using namespace std;
int main()
{
    std::unique_ptr<ObjectDetection> objectdetection = TaskAPIFactory::Create<ObjectDetection>();
    objectdetection->Detect();
    return 0;
    /*
    // Measure latency
    int numTests{100};
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    //Run: Running the session is done in the Run() method:
    for (int i = 0; i < numTests; i++) {
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                    inputTensors.data(), 1, outputNames.data(),
                    outputTensors.data(), 1);
    }
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << "Minimum Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / static_cast<float>(numTests)
              << " ms" << std::endl;
    return 0;
    */
}