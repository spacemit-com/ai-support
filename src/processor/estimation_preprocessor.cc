#include "src/processor/estimation_preprocessor.h"

#include "src/utils/cv2_utils.h"
#include "src/utils/nms_utils.h"
#include "utils/time.h"
#include "utils/utils.h"

void EstimationPreprocessor::Preprocess(
    const cv::Mat& mat, const Boxi& box,
    std::vector<std::vector<float>>& input_tensor_values,
    std::pair<cv::Mat, cv::Mat>& crop_result_pair, unsigned int data_format) {
  if (mat.empty()) return;
  if (data_format == 0) {
    // const int input_height = input_node_dims[0][2];
    // const int input_width = input_node_dims[0][3];
    crop_result_pair = CropImageByDetectBox(mat, box);

    cv::Mat crop_matBGR = crop_result_pair.first;
    cv::Mat affine_transform_reverse = crop_result_pair.second;
    cv::Mat crop_mat, crop_matRGB;

    {
#ifdef DEBUG
      TimeWatcher t("| |-- Convert to RGB");
#endif
      // step 3: Convert the image to HWC RGB UINT8 format.
      cv::cvtColor(crop_matBGR, crop_matRGB, cv::COLOR_BGR2RGB);
    }
    {
#ifdef DEBUG
      TimeWatcher t("| |-- Convert to fp32");
#endif
      crop_matRGB.convertTo(crop_mat, CV_32F, 1.0 / 255);
    }
    {
#ifdef DEBUG
      TimeWatcher t("| |-- Normalize");
#endif
      cv::Mat channels[3];
      cv::split(crop_mat, channels);
      int channel = 3;
      std::vector<float> input_tensor_value;
      for (int i = 0; i < channel; i++) {
        // channels[i] = (channels[i] - mean_vals[i]) * scale_vals[i];
        std::vector<float> data = std::vector<float>(channels[i].reshape(1, 1));
        input_tensor_value.insert(input_tensor_value.end(), data.begin(),
                                  data.end());
      }
      input_tensor_values.push_back(input_tensor_value);
    }
  }
}

std::pair<cv::Mat, cv::Mat> EstimationPreprocessor::CropImageByDetectBox(
    const cv::Mat& input_image, const Boxi& box) {
  // auto time_start = std::chrono::steady_clock::now();
  std::pair<cv::Mat, cv::Mat> result_pair;

  if (!input_image.data) {
    return result_pair;
  }

  if (!box.flag) {
    return result_pair;
  }

  // deep copy
  cv::Mat input_mat_copy;
  input_image.copyTo(input_mat_copy);

  // calculate the width, height and center points of the human detection box
  int box_width = box.x2 - box.x1;
  int box_height = box.y2 - box.y1;
  int box_center_x = box.x1 + box_width / 2;
  int box_center_y = box.y1 + box_height / 2;

  float aspect_ratio = 192.0 / 256.0;

  // adjust the width and height ratio of the size of the picture in the RTMPOSE
  // input
  if (box_width > (aspect_ratio * box_height)) {
    box_height = box_width / aspect_ratio;
  } else if (box_width < (aspect_ratio * box_height)) {
    box_width = box_height * aspect_ratio;
  }

  float scale_image_width = box_width * 1.3;
  float scale_image_height = box_height * 1.3;

  // get the affine matrix
  cv::Mat affine_transform =
      GetAffineTransform(box_center_x, box_center_y, scale_image_width,
                         scale_image_height, 192, 256);

  cv::Mat affine_transform_reverse =
      GetAffineTransform(box_center_x, box_center_y, scale_image_width,
                         scale_image_height, 192, 256, true);

  // affine transform
  cv::Mat affine_image;
  cv::warpAffine(input_mat_copy, affine_image, affine_transform,
                 cv::Size(192, 256), cv::INTER_LINEAR);

  result_pair = std::make_pair(affine_image, affine_transform_reverse);
  return result_pair;
}
