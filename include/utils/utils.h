#ifndef _INLCUDE_UTILS_UTILS_H_
#define _INLCUDE_UTILS_UTILS_H_

#include "task/vision/object_detection_types.h"

#include "opencv2/opencv.hpp"

#include <sys/stat.h>
#include <string>
#include <iostream>
#include <stdexcept> // To use runtime_error

static bool checkImageFileExtension(const std::string& filename)
{
    size_t pos = filename.rfind('.');
    if (filename.empty())
    {
        std::cout<<"[ ERROR ] The Image file path is empty"<<std::endl;
        return false;
    }
    if (pos == std::string::npos)
        return false;
    std::string ext = filename.substr(pos+1);
    if (ext == "jpeg"||ext == "jpg"||ext == "png") {
        return true;
    } else {
        return false;
    }
}

static void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<Boxi> &boxes)
{
  if (boxes.empty()) return;
  for (const auto &box: boxes)
  {
    if (box.flag)
    {
      cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 255, 0), 2);
      if (box.label_text)
      {
        std::string label_text(box.label_text);
        label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
        cv::putText(mat_inplace, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    .6f, cv::Scalar(0, 0, 255), 1);

      }
    }
  }
}

static bool exists_check(const std::string& name) 
{
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

#endif