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
  //std::cout<<output_tensors.size()<<std::endl;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::vector<Boxf> bbox_collection;
  bbox_collection.clear();
  unsigned int count = 0;
  const float input_height = static_cast<float>(input_dims.at(2)); // e.g 640
  //std::cout<<"input_height "<<input_height<<std::endl;
  const float input_width = static_cast<float>(input_dims.at(1)); // e.g 640
  //std::cout<<"input_width "<<input_width<<std::endl;
  //std::cout<<"img_height "<<img_height<<std::endl;
  //std::cout<<"img_width "<<img_width<<std::endl;
  const float resize_ratio = std::min(input_height/img_height, input_width/img_width);
  //std::cout<<"resize_ratio"<<resize_ratio<<std::endl;
  for(int s=0;s<output_tensors.size();s++)
  {
    Ort::Value &pred = output_tensors.at(s); // batch*13*13*3*85
    auto outputInfo = pred.GetTensorTypeAndShapeInfo();
    auto pred_dims = outputInfo.GetShape();

    const unsigned int num_classes = pred_dims.at(4) - 5; // 20
    //std::cout<<pred_dims.at(2)<<std::endl;
    //std::cout<<"output_size"<<pred_dims[1]<<std::endl;
    for (unsigned int i = 0; i < pred_dims[1]; ++i)
    {
      for (unsigned int j = 0; j < pred_dims[2]; ++j)
      {
        int grid_x=j;
        //std::cout<<"grid_x"<<grid_x<<std::endl;
        int grid_y=i;      
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
            //std::cout<<"XYSCALE[0]"<<XYSCALE[0]<<std::endl;
            //std::cout<<"STRIDES[0]"<<STRIDES[0]<<std::endl;
            //std::cout<<"anchors"<<anchors[0][k][0]<<" "<<anchors[0][k][1]<<std::endl;
            float cx = (sigmoid(pred.At<float>({0, i, j, k, 0}))*XYSCALE[s] - 0.5 * (XYSCALE[s] - 1)+ grid_x)*STRIDES[s];
            //std::cout<<"cx "<<cx<<std::endl;
            float cy = (sigmoid(pred.At<float>({0, i, j, k, 1}))*XYSCALE[s] - 0.5 * (XYSCALE[s] - 1)+ grid_y)*STRIDES[s];
            //std::cout<<"cy "<<cy<<std::endl;
            float w = exp(pred.At<float>({0, i, j, k, 2}))*anchors[s][k][0];
            //std::cout<<"w "<<anchors[s][k][0]<<std::endl;
            float h = exp(pred.At<float>({0, i, j, k, 3}))*anchors[s][k][1];
            //std::cout<<"h "<<anchors[s][k][1]<<std::endl;
            Boxf box;
            float dw = (input_width - resize_ratio * img_width) / 2;
            float dh = (input_height - resize_ratio * img_height) / 2;
            //std::cout<<"dw "<<dw<<std::endl;
            //std::cout<<"dh "<<dh<<std::endl;
            box.x1 = (cx - w / 2.f - dw)/resize_ratio;
            if(box.x1<0) box.x1=0;
            //std::cout<<"box.x1 "<<box.x1<<std::endl;
            box.y1 = (cy - h / 2.f - dh)/resize_ratio;
            //std::cout<<"box.y1 "<<box.y1<<std::endl;
            if(box.y1<0) box.y1=0;
            box.x2 = (cx + w / 2.f - dw)/resize_ratio;
            //std::cout<<"box.x2 "<<box.x2<<std::endl;
            if(box.x2>img_width) box.x2=img_width;
            box.y2 = (cy + h / 2.f - dh)/resize_ratio;
            //std::cout<<"box.y2 "<<box.y2<<std::endl;
            if(box.y2>img_height) box.y2=img_height;
            box.score = conf;
            //std::cout<<"box.score"<<box.score<<std::endl;
            box.label = label;
            box.label_text = class_names[label];
            box.flag = true;
            bbox_collection.push_back(box);
            //std::cout<<"box.label_text "<<box.label_text<<std::endl;
            count += 1; // limit boxes for nms.
            //std::cout<<"count "<<count<<std::endl;
            if (count > max_nms)
              break;
        }
      }
    }
  }
  //std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
  // 4. hard|blend|offset nms with topk.
  nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
  //std::cout<<detected_boxes.size()<<std::endl;
  //std::cout << "detected_bboxes num: " << detected_boxes.size()<< "\n";
  std::chrono::steady_clock::time_point end =
  std::chrono::steady_clock::now();
  std::cout << "postprocess including inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
}



void DetectionPostprocessor::nms(std::vector<Boxf>& input, std::vector<Boxf>& output,
                float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == BLEND) blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == OFFSET) offset_nms(input, output, iou_threshold, topk);
  else hard_nms(input, output, iou_threshold, topk);
}