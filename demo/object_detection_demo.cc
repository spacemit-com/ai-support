#include <iomanip>  // for: setprecision
#include <iostream>

#include "object_detection.hpp"
#include "task/vision/object_detection_task.h"
#include "utils/json_helper.hpp"
#ifdef DEBUG
#include "utils/time.h"
#endif

static void usage(const char* exe) {
  std::cout << "Usage: \n"
            << exe << " <model_path> <label_path> <image_path> <save_path>\n"
            << exe << " <config_path> <image_path> <save_path>\n";
}

int main(int argc, char* argv[]) {
  if (argc != 4 && argc != 5) {
    usage(argv[0]);
    return -1;
  }

  ObjectDetectionOption option;
  std::string image_file_path, save_img_path;
  if (argc == 4) {
    if (configToOption(argv[1], option) != 0) {
      return -1;
    }
    image_file_path = argv[2];
    save_img_path = argv[3];
  } else if (argc == 5) {
    option.model_path = argv[1];
    option.label_path = argv[2];
    image_file_path = argv[3];
    save_img_path = argv[4];
  }

  std::unique_ptr<ObjectDetectionTask> objectdetectiontask =
      std::unique_ptr<ObjectDetectionTask>(new ObjectDetectionTask(option));
  if (objectdetectiontask->getInitFlag() != 0) {
    return -1;
  }

  cv::Mat img_raw;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
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
  std::vector<Boxi> bboxes = objectdetectiontask->Detect(img_raw).result_bboxes;
  for (size_t i = 0; i < bboxes.size(); i++) {
    std::cout << "bbox[" << std::setw(2) << i << "] x1y1x2y2: (" << std::setw(4)
              << bboxes[i].x1 << "," << std::setw(4) << bboxes[i].y1 << ","
              << std::setw(4) << bboxes[i].x2 << "," << std::setw(4)
              << bboxes[i].y2 << "), score: " << std::fixed
              << std::setprecision(3) << std::setw(4) << bboxes[i].score
              << ", label_text: " << bboxes[i].label_text << std::endl;
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
