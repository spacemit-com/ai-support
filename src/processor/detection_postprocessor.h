#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>

#include "core/types.h"
#include "src/utils/nms_utils.h"
#include "src/processor/processor.h"

#include "onnxruntime_cxx_api.h"

class DetectionPostprocessor : public Postprocessor{
    public:
    DetectionPostprocessor() {};

    void Postprocess(std::vector<Ort::Value> output_tensors,
                    std::vector<Boxi> &result_boxes,
                    std::vector<int64_t>& input_dims,
                    int img_height,
                    int img_width,
                    float score_threshold = 0.25f, 
                    float iou_threshold = 0.45f, 
                    unsigned int topk = 100, 
                    unsigned int nms_type = OFFSET
                    );

    private: 
    float sigmoid(float x)
    {
        return (1 / (1 + exp(-x)));
    }

    void nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
                float iou_threshold, unsigned int topk, unsigned int nms_type);
    
    int STRIDES[3] = {8, 16, 32};
    float XYSCALE[3] = {1.2, 1.1, 1.05};
    int anchors[3][3][2]= {{{12,16}, {19,36}, {40,28}}, 
                           {{36,75}, {76,55}, {72,146}}, 
                           {{142,110}, {192,243}, {459,401}}};
    
    const char *class_names[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
    };
};
