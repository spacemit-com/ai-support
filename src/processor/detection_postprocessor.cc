#include "src/processor/detection_postprocessor.h"
#include "utils/time.h"

void DetectionPostprocessor::Postprocess(std::vector<Ort::Value> output_tensors,
                                         std::vector<Boxi> &result_boxes,
                                         std::vector<int64_t>& input_dims,
                                         int img_height,
                                         int img_width,
                                         std::vector<std::string> labels,
                                         float score_threshold, 
                                         float iou_threshold, 
                                         unsigned int topk, 
                                         unsigned int nms_type)
{
#ifdef DEBUG
  TimeWatcher t("|-- Postprocess");
#endif
  std::vector<Boxf> bbox_collection;
  bbox_collection.clear();
  unsigned int count = 0;
  const float input_height = static_cast<float>(input_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_dims.at(1)); // e.g 640
  const float resize_ratio = std::min(input_height/img_height, input_width/img_width);
  for(int s=0;s<output_tensors.size();s++)
  {
    Ort::Value &pred = output_tensors.at(s); // batch*13*13*3*85
    auto outputInfo = pred.GetTensorTypeAndShapeInfo();
    auto pred_dims = outputInfo.GetShape();

    const unsigned int num_classes = pred_dims.at(4) - 5; // 20
    for (unsigned int i = 0; i < pred_dims[1]; ++i)
    {
      for (unsigned int j = 0; j < pred_dims[2]; ++j)
      {
        int grid_x=j;
        int grid_y=i;      
        for (unsigned int k = 0; k < pred_dims[3]; ++k)
        {
            float obj_conf = pred.At<float>({0, i, j, k, 4});
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
            float cx = (sigmoid(pred.At<float>({0, i, j, k, 0}))*XYSCALE[s] - 0.5 * (XYSCALE[s] - 1)+ grid_x)*STRIDES[s];
            float cy = (sigmoid(pred.At<float>({0, i, j, k, 1}))*XYSCALE[s] - 0.5 * (XYSCALE[s] - 1)+ grid_y)*STRIDES[s];
            float w = exp(pred.At<float>({0, i, j, k, 2}))*anchors[s][k][0];
            float h = exp(pred.At<float>({0, i, j, k, 3}))*anchors[s][k][1];
            Boxf box;
            float dw = (input_width - resize_ratio * img_width) / 2;
            float dh = (input_height - resize_ratio * img_height) / 2;
            box.x1 = (cx - w / 2.f - dw)/resize_ratio;
            box.x1 = std::max(box.x1,.0f);
            box.y1 = (cy - h / 2.f - dh)/resize_ratio;
            box.y1 = std::max(box.y1,.0f);
            box.x2 = (cx + w / 2.f - dw)/resize_ratio;
            box.x2 = std::min(box.x2,float(img_width-1));
            box.y2 = (cy + h / 2.f - dh)/resize_ratio;
            box.y2 = std::min(box.y2,float(img_height-1));
            box.score = conf;
            box.label = label;
            box.label_text = labels[label].c_str();
            box.flag = true;
            bbox_collection.push_back(box);
            count += 1; // limit boxes for nms.
            if (count > max_nms)
              break;
        }
      }
    }
  }
  std::vector<Boxf> detected_boxes;

  // 4. hard|blend|offset nms with topk.
  nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);

  int detected_boxes_num = detected_boxes.size();
  for(int i = 0; i < detected_boxes_num; i++)
  {
    Boxi result_box;
    result_box.x1 = int(detected_boxes[i].x1);
    result_box.y1 = int(detected_boxes[i].y1);
    result_box.x2 = int(detected_boxes[i].x2);
    result_box.y2 = int(detected_boxes[i].y2);
    result_box.label = detected_boxes[i].label;
    result_box.score = detected_boxes[i].score;
    result_box.label_text = detected_boxes[i].label_text;
    result_box.flag = detected_boxes[i].flag;
    result_boxes.push_back(result_box);
  }
}

void DetectionPostprocessor::Postprocess_Yolov6(std::vector<Ort::Value> output_tensors,
            std::vector<Boxi> &result_boxes,
            std::vector<int64_t>& input_dims,
            int img_height,
            int img_width,
            std::vector<std::string> labels)
{
#ifdef DEBUG
  TimeWatcher t("|-- Postprocess");
#endif
  std::vector<Boxf> bbox_collection;
  bbox_collection.clear();
  unsigned int count = 0;
  const float input_height = static_cast<float>(input_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_dims.at(3)); // e.g 640
  const float resize_ratio = std::min(input_height/img_height, input_width/img_width);
  Ort::Value &pred0 = output_tensors.at(0); // batch*13*13*3*85
  Ort::Value &pred1 = output_tensors.at(1); // batch*13*13*3*85
  Ort::Value &pred2 = output_tensors.at(2);
  auto outputInfo = pred0.GetTensorTypeAndShapeInfo();
  auto pred_dims = outputInfo.GetShape();
  float dw = (input_width - resize_ratio * img_width) / 2;
  float dh = (input_height - resize_ratio * img_height) / 2;

  int num = pred_dims[1];
  for(int i=0;i<num;i++)
  {
    Boxi result_box;
    result_box.x1 = int((pred0.At<float>({0,i,0}) - dw)/resize_ratio);
    result_box.y1 = int((pred0.At<float>({0,i,1}) - dh)/resize_ratio);
    result_box.x2 = int((pred0.At<float>({0,i,2}) - dw)/resize_ratio);
    result_box.y2 = int((pred0.At<float>({0,i,3}) - dh)/resize_ratio);
    if(pred1.At<int>({0,i})<0)
    {
      continue;
    }
    result_box.label = pred1.At<int>({0,i});
    result_box.score = pred2.At<float>({0,i,0});
    result_box.label_text = labels[result_box.label].c_str();
    result_box.flag = true;
    result_boxes.push_back(result_box);
  }
}

void DetectionPostprocessor::Postprocess_NanoDet(std::vector<Ort::Value> output_tensors,
            std::vector<Boxi> &result_boxes,
            std::vector<int64_t>& input_dims,
            int img_height,
            int img_width,
            std::vector<std::string> labels)
{
#ifdef DEBUG
  TimeWatcher t("|-- Postprocess");
#endif
  std::vector<Boxf> bbox_collection;
  bbox_collection.clear();
  const int cls_num = 80;
  const float input_height = static_cast<float>(input_dims.at(2)); // e.g 640
  const float input_width = static_cast<float>(input_dims.at(3)); // e.g 640
  const float resize_ratio = std::min(input_height/img_height, input_width/img_width);
  Ort::Value &pred = output_tensors.at(0); // batch*2125*112
  auto outputInfo = pred.GetTensorTypeAndShapeInfo();
  auto pred_dims = outputInfo.GetShape();
  std::vector<int> hw = {40, 20, 10 ,5};
  std::vector<int> strides = {8, 16, 32, 64};
  int num=-1;
  for (unsigned int i = 0; i < 4; i++)
  {
    for (unsigned int y = 0; y < hw[i]; y++)
    {
      for (unsigned int x = 0; x < hw[i]; x++)
      {
        num++;
        int ct_x = x;
        int ct_y = y;
        float cls_conf = pred.At<float>({0, num, 0});
        float tmp_conf;
        unsigned int label = 0;
        for (unsigned int h = 0; h < cls_num; h++)
        {
          tmp_conf = pred.At<float>({0, num, h});
          if (tmp_conf > cls_conf)
          {
            cls_conf = tmp_conf;
            label = h;
          }
        }
 
        
        if(cls_conf<0.30f) continue;
        std::vector<float> dis_pred(4,.0);
        for(int s=0;s<4;s++)
        {
          float alptha = .0;
          int cur_num = cls_num + s*8;
          std::vector<float> dst(8);
          for(int j=0;j<8;j++)
          {
            alptha = std::max(alptha, pred.At<float>({0, num, cur_num +j}));
          }
          float sum = .0f;
          for(int j=0;j<8;j++)
          {
            dst[j] =  fast_exp(pred.At<float>({0, num, cur_num+j}) - alptha);
            sum = sum + dst[j];
          }
          for(int j=0;j<8;j++)
          {
            dis_pred[s] = dis_pred[s] + j*(dst[j]/sum);
          }
        }
        
        Boxf box;
        float dw = (input_width - resize_ratio * img_width) / 2;
        float dh = (input_height - resize_ratio * img_height) / 2;
        box.label = label;
        box.label_text = labels[label].c_str();
        box.score = cls_conf;
        box.flag = true;
        box.x1 = ((ct_x - dis_pred[0])*strides[i] - dw)/resize_ratio;
        box.x1 = std::max(box.x1,.0f);
        box.y1 = ((ct_y - dis_pred[1])*strides[i] - dh)/resize_ratio;
        box.y1 = std::max(box.y1,.0f);
        box.x2 = ((ct_x + dis_pred[2])*strides[i] - dw)/resize_ratio;
        box.x2 = std::min(box.x2,float(img_width-1));
        box.y2 = ((ct_y + dis_pred[3])*strides[i] - dh)/resize_ratio;
        box.y2 = std::min(box.y2,float(img_height-1));
        bbox_collection.push_back(box);
      }
    }
  }
  std::vector<Boxf> detected_boxes;

  // 4. hard|blend|offset nms with topk.
  nms(bbox_collection, detected_boxes, 0.30f, 100, OFFSET);

  int detected_boxes_num = detected_boxes.size();
  for(int i = 0; i < detected_boxes_num; i++)
  {
    Boxi result_box;
    result_box.x1 = int(detected_boxes[i].x1);
    result_box.y1 = int(detected_boxes[i].y1);
    result_box.x2 = int(detected_boxes[i].x2);
    result_box.y2 = int(detected_boxes[i].y2);
    result_box.label = detected_boxes[i].label;
    result_box.score = detected_boxes[i].score;
    result_box.label_text = detected_boxes[i].label_text;
    result_box.flag = detected_boxes[i].flag;
    result_boxes.push_back(result_box);
  }
}

void DetectionPostprocessor::nms(std::vector<Boxf>& input, std::vector<Boxf>& output,
                float iou_threshold, unsigned int topk, unsigned int nms_type)
{
  if (nms_type == BLEND) blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == OFFSET) offset_nms(input, output, iou_threshold, topk);
  else hard_nms(input, output, iou_threshold, topk);
}