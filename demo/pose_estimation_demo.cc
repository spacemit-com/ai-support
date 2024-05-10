#include <iomanip>  // for: setprecision
#include <iostream>

#include "pose_estimation.hpp"
#include "task/vision/object_detection_task.h"
#include "task/vision/pose_estimation_task.h"
#include "utils/json_helper.hpp"

static void usage(const char* exe) {
  std::cout << "Usage: \n"
            << exe
            << " <detection_model_path> <detection_label_path> "
               "<pose_point_model_path> <image_path> <save_path>\n"
            << exe
            << " <detection_config_path> <pose_point_config_path> <image_path> "
               "<save_path>\n";
}

int main(int argc, char* argv[]) {
  if (argc != 5 && argc != 6) {
    usage(argv[0]);
    return -1;
  }

  ObjectDetectionOption detection_option;
  PoseEstimationOption estimation_option;
  std::string image_file_path, save_img_path;
  if (argc == 5) {
    if (configToOption(argv[1], detection_option) != 0 ||
        configToOption(argv[2], estimation_option) != 0) {
      return -1;
    }
    image_file_path = argv[3];
    save_img_path = argv[4];
  } else if (argc == 6) {
    detection_option.model_path = argv[1];
    detection_option.label_path = argv[2];
    estimation_option.model_path = argv[3];
    image_file_path = argv[4];
    save_img_path = argv[5];
  }

  std::unique_ptr<ObjectDetectionTask> objectdetectiontask =
      std::unique_ptr<ObjectDetectionTask>(
          new ObjectDetectionTask(detection_option));
  std::unique_ptr<PoseEstimationTask> poseestimationtask =
      std::unique_ptr<PoseEstimationTask>(
          new PoseEstimationTask(estimation_option));
  if (objectdetectiontask->getInitFlag() != 0 ||
      poseestimationtask->getInitFlag() != 0) {
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
    if (img_raw.empty()) {
      std::cout << "[ ERROR ] Read image failed" << std::endl;
      return -1;
    }
  }

  std::vector<Boxi> bboxes = objectdetectiontask->Detect(img_raw).result_bboxes;
  for (size_t i = 0; i < bboxes.size(); i++) {
    Boxi box = bboxes[i];
    if (box.label != 0) {
      continue;
    }
    std::vector<PosePoint> points =
        poseestimationtask->Estimate(img_raw, box).result_points;
    draw_points_inplace(img_raw, points);
  }

  try {
    cv::imwrite(save_img_path, img_raw);
  } catch (cv::Exception& e) {
    std::cout << "[ ERROR ] Write result image failed : " << e.what()
              << std::endl;
    return -1;
  }
  return 0;
}
