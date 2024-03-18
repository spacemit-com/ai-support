#include <iomanip>  // for: setprecision
#include <iostream>

#include "object_detection.hpp"
#include "task/vision/object_detection_task.h"
#include "utils/time.h"
#include "utils/utils.h"

int main(int argc, char* argv[]) {
  std::vector<Boxi> bboxes;
  std::string image_file_path, save_img_path, config_file_path;
  ObjectDetectionOption option;
  std::unique_ptr<ObjectDetectionTask> objectdetectiontask;
  cv::Mat img_raw;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  if (argc == 4) {
    config_file_path = argv[1];
    image_file_path = argv[2];
    save_img_path = argv[3];
    objectdetectiontask = std::unique_ptr<ObjectDetectionTask>(
        new ObjectDetectionTask(config_file_path));
  } else if (argc == 5) {
    option.model_path = argv[1];
    option.label_path = argv[2];
    image_file_path = argv[3];
    save_img_path = argv[4];
    objectdetectiontask =
        std::unique_ptr<ObjectDetectionTask>(new ObjectDetectionTask(option));
  } else {
    std::cout << "Please run with " << argv[0]
              << " <model_file_path> <label_file_path> <image_file_path> "
                 "<save_img_path> or "
              << argv[0]
              << " <config_file_path> <image_file_path> <save_img_path>"
              << std::endl;
    return -1;
  }
  if (objectdetectiontask->getInitFlag() != 0) {
    return -1;
  }
  {
#ifdef DEBUG
    TimeWatcher t("|-- Load input data");
#endif
    img_raw = cv::imread(image_file_path);
  }
  if (img_raw.empty()) {
    std::cout << "[ ERROR ] Read image failed" << std::endl;
    return -1;
  }
  bboxes = objectdetectiontask->Detect(img_raw).result_bboxes;
  {
#ifdef DEBUG
    TimeWatcher t("|-- Output result");
#endif
    for (size_t i = 0; i < bboxes.size(); i++) {
      std::cout << "bbox[" << std::setw(2) << i << "]"
                << " "
                << "x1y1x2y2: "
                << "(" << std::setw(4) << bboxes[i].x1 << "," << std::setw(4)
                << bboxes[i].y1 << "," << std::setw(4) << bboxes[i].x2 << ","
                << std::setw(4) << bboxes[i].y2 << ")"
                << ", "
                << "score: " << std::fixed << std::setprecision(3)
                << std::setw(4) << bboxes[i].score << ", "
                << "label_text: " << bboxes[i].label_text << std::endl;
    }
  }
  {
#ifdef DEBUG
    TimeWatcher t("|-- Box drawing");
#endif
    draw_boxes_inplace(img_raw, bboxes);
  }
  try {
    cv::imwrite(save_img_path, img_raw);
  } catch (cv::Exception& e) {
    std::cout << "[ ERROR ] Write result image failed : " << e.what()
              << std::endl;
    return -1;
  }
  // cv::imshow("detected.jpg",img_raw);
  // cv::waitKey(0);
  return 0;
}
