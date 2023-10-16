#include "detection_postprocessor.h"

void DetectionPostprocessor::Postprocess(const YoloXScaleParams &scale_params,
                                         std::vector<Ort::Value> output_tensors,
                                         std::vector<Boxf> &detected_boxes,
                                         std::vector<int64_t>& input_dims,
                                         std::vector<int64_t>& output_dims,
                                         int img_height,
                                         int img_width,
                                         float score_threshold, 
                                         float iou_threshold, 
                                         unsigned int topk, 
                                         unsigned int nms_type)
{
  std::vector<Boxf> bbox_collection;
  const unsigned int num_anchors = output_dims.at(1); // n = ?
  const unsigned int num_classes = output_dims.at(2) - 5;
  const float input_height = static_cast<float>(input_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_dims.at(3)); // e.g 640

  std::vector<YoloXAnchor> anchors;
  std::vector<int> strides = {8, 16, 32}; // might have stride=64
  generate_anchors(input_height, input_width, strides, anchors);

  float r_ = scale_params.r;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    float obj_conf = output_tensors[0].At<float>({0, i, 4});
    if (obj_conf < score_threshold) continue; // filter first.

    float cls_conf = output_tensors[0].At<float>({0, i, 5});
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = output_tensors[0].At<float>({0, i, j + 5});
      if (tmp_conf > cls_conf)
      {
        cls_conf = tmp_conf;
        label = j;
      }
    } // argmax
    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    const int grid0 = anchors.at(i).grid0;
    const int grid1 = anchors.at(i).grid1;
    const int stride = anchors.at(i).stride;

    float dx = output_tensors[0].At<float>({0, i, 0});
    float dy = output_tensors[0].At<float>({0, i, 1});
    float dw = output_tensors[0].At<float>({0, i, 2});
    float dh = output_tensors[0].At<float>({0, i, 3});

    float cx = (dx + (float) grid0) * (float) stride;
    float cy = (dy + (float) grid1) * (float) stride;
    float w = std::exp(dw) * (float) stride;
    float h = std::exp(dh) * (float) stride;
    float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
    float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
    float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
    float y2 = ((cy + h / 2.f) - (float) dh_) / r_;

    Boxf box;
    box.x1 = std::max(0.f, x1);
    box.y1 = std::max(0.f, y1);
    box.x2 = std::min(x2, (float) img_width - 1.f);
    box.y2 = std::min(y2, (float) img_height - 1.f);
    box.score = conf;
    box.label = label;
    box.label_text = class_names[label];
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
  // 4. hard|blend|offset nms with topk.
  nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
  std::cout << "detected_bboxes num: " << detected_boxes[0].label_text << "\n";
}

void DetectionPostprocessor::generate_anchors(const int target_height,
                             const int target_width,
                             std::vector<int> &strides,
                             std::vector<YoloXAnchor> &anchors)
{
  for (auto stride: strides)
  {
    int num_grid_w = target_width / stride;
    int num_grid_h = target_height / stride;
    for (int g1 = 0; g1 < num_grid_h; ++g1)
    {
      for (int g0 = 0; g0 < num_grid_w; ++g0)
      {
#ifdef LITE_WIN32
        YoloXAnchor anchor;
        anchor.grid0 = g0;
        anchor.grid1 = g1;
        anchor.stride = stride;
        anchors.push_back(anchor);
#else
        anchors.push_back((YoloXAnchor) {g0, g1, stride});
#endif
      }
    }
  }
}


void DetectionPostprocessor::nms(std::vector<Boxf>& input, std::vector<Boxf>& output,
                float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == BLEND) blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == OFFSET) offset_nms(input, output, iou_threshold, topk);
  else hard_nms(input, output, iou_threshold, topk);
}