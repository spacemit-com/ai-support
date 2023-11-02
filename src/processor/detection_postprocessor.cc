#include "detection_postprocessor.h"
#include <iostream>

void DetectionPostprocessor::Postprocess(std::vector<Ort::Value> output_tensors,
                                         std::vector<Boxf> &detected_boxes,
                                         std::vector<int64_t>& input_dims,
                                         int img_height,
                                         int img_width,
                                         float score_threshold, 
                                         float iou_threshold, 
                                         unsigned int topk, 
                                         unsigned int nms_type)
{
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::vector<Boxf> bbox_collection;
  Ort::Value &pred = output_tensors.at(0); // batch*13*13*3*85
  auto outputInfo = pred.GetTensorTypeAndShapeInfo();
  auto pred_dims = outputInfo.GetShape();

  const unsigned int num_classes = pred_dims.at(4) - 5; // 20
  const float input_height = static_cast<float>(input_dims.at(2)); // e.g 640
  //std::cout<<"input_height "<<input_height<<std::endl;
  const float input_width = static_cast<float>(input_dims.at(1)); // e.g 640
  //std::cout<<"input_width "<<input_width<<std::endl;
  //std::cout<<"img_height "<<img_height<<std::endl;
  //std::cout<<"img_width "<<img_width<<std::endl;
  const float scale_height = img_height / input_height;
  //std::cout<<"scale_height "<<scale_height<<std::endl;
  const float scale_width = img_width / input_width;
  //std::cout<<"scale_width"<<scale_width<<std::endl;

  bbox_collection.clear();
  unsigned int count = 0;
  //std::cout<<"output_size"<<pred_dims[1]<<std::endl;
  for (unsigned int i = 0; i < pred_dims[1]; ++i)
  {
    for (unsigned int j = 0; j < pred_dims[2]; ++j)
    {
      int grid_x=(416/52)*j;
      //std::cout<<"grid_x"<<grid_x<<std::endl;
      int grid_y=(416/52)*i;      
      //std::cout<<"grid_y"<<grid_y<<std::endl;
      for (unsigned int k = 0; k < pred_dims[3]; ++k)
      {
          float obj_conf = pred.At<float>({0, i, j, k, 4});
          //std::cout<<obj_conf<<std::endl;
          //std::cout<<score_threshold<<std::endl;
          if (obj_conf < score_threshold) continue; // filter first.

          float cls_conf = pred.At<float>({0, i, j, k, 5});
          unsigned int label = 0;
          for (unsigned int h = 0; h < num_classes; ++h)
          {
            float tmp_conf = pred.At<float>({0, i, j, k, h + 5});
            if (tmp_conf > cls_conf)
            {
              cls_conf = tmp_conf;
              label = h;
            }
          }
          float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
          if (conf < score_threshold) continue; // filter

          float cx = sigmoid(pred.At<float>({0, i, j, k, 0})) + grid_x;
          //std::cout<<"cx "<<cx<<std::endl;
          float cy = sigmoid(pred.At<float>({0, i, j, k, 1})) + grid_y;
          //std::cout<<"cy "<<cy<<std::endl;
          float w = exp(pred.At<float>({0, i, j, k, 2}))*36;
          //std::cout<<"w "<<w<<std::endl;
          float h = exp(pred.At<float>({0, i, j, k, 3}))*75;
          //std::cout<<"h "<<h<<std::endl;

          Boxf box;
          box.x1 = (cx - w / 2.f) * scale_width;
          //std::cout<<"box.x1 "<<box.x1<<std::endl;
          box.y1 = (cy - h / 2.f) * scale_height;
          //std::cout<<"box.y1 "<<box.y1<<std::endl;
          box.x2 = (cx + w / 2.f) * scale_width;
          //std::cout<<"box.x2 "<<box.x2<<std::endl;
          box.y2 = (cy + h / 2.f) * scale_height;
          //std::cout<<"box.y2 "<<box.y2<<std::endl;
          box.score = conf;
          //std::cout<<"box.score"<<box.score<<std::endl;
          box.label = label;
          box.label_text = class_names[label];
          box.flag = true;
          bbox_collection.push_back(box);
          //std::cout<<"box.label_text "<<box.label_text<<std::endl;
          count += 1; // limit boxes for nms.
          if (count > max_nms)
            break;
      }
    }
  }
  //std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
  // 4. hard|blend|offset nms with topk.
  nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
  //std::cout << "detected_bboxes num: " << detected_boxes.size()<< "\n";
  std::chrono::steady_clock::time_point end =
  std::chrono::steady_clock::now();
  std::cout << "postprocess including inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
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