#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <memory>
#include <vector>

#include "task/vision/image_classification_task.h"
#include "task/vision/image_classification_types.h"
#include "task/vision/object_detection_task.h"
#include "task/vision/object_detection_types.h"
#include "task/vision/pose_estimation_task.h"
#include "task/vision/pose_estimation_types.h"
#include "utils.h"
#include "utils/utils.h"
namespace py = pybind11;

struct PYImageClassificationTask {
 public:
  PYImageClassificationTask(const ImageClassificationOption &option) {
    task_ = std::make_shared<ImageClassificationTask>(option);
  }
  int getInitFlag() { return task_->getInitFlag(); };
  void Classify(const py::array_t<unsigned char> &input) {
    cv::Mat img = numpy_uint8_3c_to_cv_mat(input);
    task_->Classify(img);
  }

 private:
  std::shared_ptr<ImageClassificationTask> task_;
};

struct PYObjectDetectionTask {
 public:
  PYObjectDetectionTask(const ObjectDetectionOption &option) {
    task_ = std::make_shared<ObjectDetectionTask>(option);
  }
  int getInitFlag() { return task_->getInitFlag(); };
  std::vector<Box> Detect(const py::array_t<unsigned char> &input) {
    cv::Mat img = numpy_uint8_3c_to_cv_mat(input);
    ObjectDetectionResult result = task_->Detect(img);
    std::vector<Box> result_boxes;
    for (size_t i = 0; i < result.result_bboxes.size(); i++) {
      Box box;
      box.x1 = result.result_bboxes[i].x1;
      box.y1 = result.result_bboxes[i].y1;
      box.x2 = result.result_bboxes[i].x2;
      box.y2 = result.result_bboxes[i].y2;
      box.label_text = result.result_bboxes[i].label_text;
      box.label = result.result_bboxes[i].label;
      result_boxes.push_back(box);
    }
    return result_boxes;
  }

 private:
  std::shared_ptr<ObjectDetectionTask> task_;
};

struct PYPoseEstimationTask {
 public:
  PYPoseEstimationTask(const PoseEstimationOption &option) {
    task_ = std::make_shared<PoseEstimationTask>(option);
  }
  int getInitFlag() { return task_->getInitFlag(); };
  std::vector<PosePoint> Estimate(const py::array_t<unsigned char> &input,
                                  const Box &box) {
    cv::Mat img = numpy_uint8_3c_to_cv_mat(input);
    Boxi boxi;
    boxi.x1 = box.x1;
    boxi.y1 = box.y1;
    boxi.x2 = box.x2;
    boxi.y2 = box.y2;
    boxi.label_text = box.label_text;
    boxi.label = box.label;
    boxi.score = box.score;
    PoseEstimationResult result = task_->Estimate(img, boxi);
    return result.result_points;
  }

 private:
  std::shared_ptr<PoseEstimationTask> task_;
};

PYBIND11_MODULE(bianbuai_pybind11_state, m) {
  py::class_<ImageClassificationOption>(m, "ImageClassificationOption")
      .def(py::init())
      .def(py::init<const std::string &, const std::string &, const int &,
                    const int &>())
      .def_readwrite("model_path", &ImageClassificationOption::model_path)
      .def_readwrite("label_path", &ImageClassificationOption::label_path)
      .def_readwrite("intra_threads_num",
                     &ImageClassificationOption::intra_threads_num)
      .def_readwrite("inter_threads_num",
                     &ImageClassificationOption::inter_threads_num);
  py::class_<PYImageClassificationTask>(m, "ImageClassificationTask")
      .def(py::init<const ImageClassificationOption &>())
      .def("getInitFlag", &PYImageClassificationTask::getInitFlag)
      .def("Classify", &PYImageClassificationTask::Classify);
  py::class_<ObjectDetectionOption>(m, "ObjectDetectionOption")
      .def(py::init())
      .def(py::init<const std::string &, const std::string &, const int &,
                    const int &, const int, const int, const std::vector<int> &,
                    const std::vector<int> &>())
      .def_readwrite("model_path", &ObjectDetectionOption::model_path)
      .def_readwrite("label_path", &ObjectDetectionOption::label_path)
      .def_readwrite("intra_threads_num",
                     &ObjectDetectionOption::intra_threads_num)
      .def_readwrite("inter_threads_num",
                     &ObjectDetectionOption::inter_threads_num)
      .def_readwrite("score_threshold", &ObjectDetectionOption::score_threshold)
      .def_readwrite("nms_threshold", &ObjectDetectionOption::nms_threshold)
      .def_readwrite("class_name_whitelist",
                     &ObjectDetectionOption::class_name_whitelist)
      .def_readwrite("class_name_blacklist",
                     &ObjectDetectionOption::class_name_blacklist);
  py::class_<Box>(m, "Box")
      .def_readwrite("x1", &Box::x1)
      .def_readwrite("y1", &Box::y1)
      .def_readwrite("x2", &Box::x2)
      .def_readwrite("y2", &Box::y2)
      .def_readwrite("score", &Box::score)
      .def_readwrite("label_text", &Box::label_text)
      .def_readwrite("label", &Box::label);
  py::class_<PYObjectDetectionTask>(m, "ObjectDetectionTask")
      .def(py::init<const ObjectDetectionOption &>())
      .def("getInitFlag", &PYObjectDetectionTask::getInitFlag)
      .def("Detect", &PYObjectDetectionTask::Detect);
  py::class_<PoseEstimationOption>(m, "PoseEstimationOption")
      .def(py::init())
      .def(py::init<const std::string &, const int &, const int &>())
      .def_readwrite("model_path", &PoseEstimationOption::model_path)
      .def_readwrite("intra_threads_num",
                     &PoseEstimationOption::intra_threads_num)
      .def_readwrite("inter_threads_num",
                     &PoseEstimationOption::inter_threads_num);
  py::class_<PYPoseEstimationTask>(m, "PoseEstimationTask")
      .def(py::init<PoseEstimationOption &>())
      .def("getInitFlag", &PYPoseEstimationTask::getInitFlag)
      .def("Estimate", &PYPoseEstimationTask::Estimate);
  py::class_<PosePoint>(m, "PosePoint")
      .def_readwrite("x", &PosePoint::x)
      .def_readwrite("y", &PosePoint::y)
      .def_readwrite("score", &PosePoint::score);
}