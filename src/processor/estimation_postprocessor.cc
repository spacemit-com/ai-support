#include "estimation_postprocessor.h"

#include "utils/time.h"

void EstimationPostprocessor::Postprocess(
    std::vector<Ort::Value> output_tensors,
    std::pair<cv::Mat, cv::Mat> crop_result_pair,
    std::vector<PosePoint> &result_points) {
#ifdef DEBUG
  TimeWatcher t("|-- Postprocess");
#endif
  std::vector<int64_t> simcc_x_dims =
      output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
  std::vector<int64_t> simcc_y_dims =
      output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

  assert(simcc_x_dims.size() == 3 && simcc_y_dims.size() == 3);

  // int batch_size = simcc_x_dims[0] == simcc_y_dims[0] ? simcc_x_dims[0] : 0;
  int joint_num = simcc_x_dims[1] == simcc_y_dims[1] ? simcc_x_dims[1] : 0;
  int extend_width = simcc_x_dims[2];
  int extend_height = simcc_y_dims[2];

  const float *simcc_x_result = output_tensors[0].GetTensorData<float>();
  const float *simcc_y_result = output_tensors[1].GetTensorData<float>();

  for (int i = 0; i < joint_num; ++i) {
    // find the maximum and maximum indexes in the value of each Extend_width
    // length
    auto x_biggest_iter =
        std::max_element(simcc_x_result + i * extend_width,
                         simcc_x_result + i * extend_width + extend_width);
    int max_x_pos =
        std::distance(simcc_x_result + i * extend_width, x_biggest_iter);
    int pose_x = max_x_pos / 2;
    float score_x = *x_biggest_iter;

    // find the maximum and maximum indexes in the value of each exten_height
    // length
    auto y_biggest_iter =
        std::max_element(simcc_y_result + i * extend_height,
                         simcc_y_result + i * extend_height + extend_height);
    int max_y_pos =
        std::distance(simcc_y_result + i * extend_height, y_biggest_iter);
    int pose_y = max_y_pos / 2;
    float score_y = *y_biggest_iter;

    // float score = (score_x + score_y) / 2;
    float score = std::max(score_x, score_y);

    PosePoint temp_point;
    temp_point.x = static_cast<int>(pose_x);
    temp_point.y = static_cast<int>(pose_y);
    temp_point.score = score;
    result_points.emplace_back(temp_point);
    // char index[8];
    // sprintf(index, "%d", i);
    // cv::putText(input_mat_copy_rgb, index, cv::Point(pose_result[i].x,
    // pose_result[i].y), 1, 1, cv::Scalar{ 0, 0, 255 }, 1);
  }
  // cv::imwrite("pose.jpg", input_mat_copy_rgb);
  // anti affine transformation to obtain the coordinates on the original
  // picture
  cv::Mat affine_transform_reverse = crop_result_pair.second;
  for (size_t i = 0; i < result_points.size(); ++i) {
    cv::Mat origin_point_Mat = cv::Mat::ones(3, 1, CV_64FC1);
    origin_point_Mat.at<double>(0, 0) = result_points[i].x;
    origin_point_Mat.at<double>(1, 0) = result_points[i].y;
    cv::Mat temp_result_mat = affine_transform_reverse * origin_point_Mat;
    result_points[i].x = temp_result_mat.at<double>(0, 0);
    result_points[i].y = temp_result_mat.at<double>(1, 0);
  }
}
