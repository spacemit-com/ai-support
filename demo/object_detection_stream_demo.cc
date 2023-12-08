#include "task/vision/object_detection_task.h"
#include "utils/utils.h"


#include <iostream>
#include <thread>
#include <string>
#include <mutex>
#include <queue>
#include <condition_variable>

#include "opencv2/opencv.hpp"

template<typename T>
class ProducerConsumerQueue {
public:
    ProducerConsumerQueue(int size) : m_maxSize(size), m_curSize(0) {}

    void push(T& val) {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_curSize >= m_maxSize) {
            m_notFull.wait(lock);
        }
        m_queue.push(val);
        ++m_curSize;
        m_notEmpty.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (m_curSize <= 0) {
            m_notEmpty.wait(lock);
        }
        T val = m_queue.front();
        m_queue.pop();
        --m_curSize;
        m_notFull.notify_one();
        return val;
    }

    int size() {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_curSize;
    }

private:
    std::queue<T> m_queue;
    int m_maxSize;
    int m_curSize;
    std::mutex m_mutex;
    std::condition_variable m_notEmpty;
    std::condition_variable m_notFull;
};

class Detector {
public:
  Detector()
  {
    objs_array_ = std::unique_ptr<ProducerConsumerQueue<ObjectDetectionResult>> (new ProducerConsumerQueue<ObjectDetectionResult>(1));
  }
  ~Detector() {};
  // 初始化/反初始化
  int init(std::string filePath, std::string labelFilepath) 
  {
    objectdetectiontask_ = std::unique_ptr<objectDetectionTask>(new objectDetectionTask(filePath, labelFilepath));
    return 0;
  }

  int uninit() {return 0;}

  // 推理
  int infer(cv::Mat frame) {
    ObjectDetectionResult objs_temp = objectdetectiontask_->Detect_NanoDet(frame);
    objs_mutex_.lock();
    objs_array_->push(objs_temp);   // 直接替换掉当前的 objs_array_
    objs_mutex_.unlock();
    return objs_array_->size();
  }
  
  // 查询检测结果
  int detected() {
    return objs_array_->size();
  }
  
  // 移走检测结果
  ObjectDetectionResult get_object() {
    ObjectDetectionResult objs_moved;
    objs_mutex_.lock();
    objs_moved = objs_array_->pop();  // 移走后 objs_array_ 为空数组
    objs_mutex_.unlock();
    return objs_moved;
  }
  
private:
  std::mutex objs_mutex_;
  std::unique_ptr<ProducerConsumerQueue<ObjectDetectionResult>> objs_array_;
  std::unique_ptr<objectDetectionTask> objectdetectiontask_;
};


class DataLoader {
public:
  DataLoader() 
  {
    enable = true;
  }
  ~DataLoader() {};
  bool ifenable() {return enable;}
  virtual cv::Mat fetch_frame() = 0;
  virtual cv::Mat peek_frame() = 0;

private:
  bool enable;
};

// 独占式
class ExclusiveDataLoader: public DataLoader {
public:
  ExclusiveDataLoader()
  {
    capture = std::unique_ptr<cv::VideoCapture> (new cv::VideoCapture(0));
    int width = 1280;
    int height = 720;
    capture->set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture->set(cv::CAP_PROP_FRAME_HEIGHT, height);
  }
  ~ExclusiveDataLoader() {};
  cv::Mat fetch_frame()
  {
    cv::Mat frame;
		capture->read(frame);
    return frame; 
  }
  cv::Mat peek_frame() { return fetch_frame(); }

private:
  std::unique_ptr<cv::VideoCapture> capture;
};

// 共享式
class SharedDataLoader: public DataLoader {
public:
  SharedDataLoader() 
  {
    frame_queue_ = std::unique_ptr<ProducerConsumerQueue<cv::Mat>> (new ProducerConsumerQueue<cv::Mat>(1));
    capture = std::unique_ptr<cv::VideoCapture> (new cv::VideoCapture("/home/gexy5/Documents/bianbu-support/data/imgs/test.mp4"));
  }
  ~SharedDataLoader() {};
  cv::Mat fetch_frame() 
  {
    frame_mutex_.lock();
    if(frame_queue_->size())         // 有缓存
      frame = frame_queue_->pop();   // 弹出一帧
    else
      capture->read(frame);       // 无缓存: 新取一帧
    frame_mutex_.unlock();
    return frame;
  }
  cv::Mat peek_frame() { 
    frame_mutex_.lock();
    capture->read(frame);     // 取最新一帧数据
    frame_queue_->push(frame);   // 缓存(预览)
    frame_mutex_.unlock();
    return frame;
  }

private:
  cv::Mat frame;
  std::mutex frame_mutex_;
  std::unique_ptr<cv::VideoCapture> capture;
  std::unique_ptr<ProducerConsumerQueue<cv::Mat>> frame_queue_;
};

// 检测线程
void Detection(ExclusiveDataLoader& dataloader, Detector& detector) {
  cv::Mat frame; 
  while (dataloader.ifenable())
  {
      frame = dataloader.peek_frame(); // 取(拷贝)一帧数据
      int flag = detector.infer(frame);       // 推理并保存检测结果
  }
}

// 预览线程
void Preview(ExclusiveDataLoader& dataloader, Detector& detector) {
  cv::Mat frame;
  while (true)
  {
    frame = dataloader.fetch_frame();  // 取(搬走)一帧数据
    if(detector.detected())// 判断原因: detector.detected 不用锁, detector.get_object 需要锁;
    {
      // 是否有检测结果
      ObjectDetectionResult objs = detector.get_object();  // 取(搬走)检测结果(移动赋值)
      if(objs.result_bboxes.size())
      {
        draw_boxes_inplace(frame, objs.result_bboxes);   // 画框
      }   
      cv::imshow("Detection", frame);
      cv::waitKey(10);
    }   // 调用 detector.detected 和 detector.get_object 期间, 检测结果依然可能被刷新
  }
}

int main(int argc, char* argv[])
{
  Detector detector;
  std::string filePath, labelFilepath;
  if(argc == 3)
  {
    filePath = argv[1];
    labelFilepath = argv[2];
  }
  else
  {
    std::cout<<"run with ./detection_stream_demo <modelFilepath> <labelFilepath>" <<std::endl;
    return 0;
  }
  detector.init(filePath, labelFilepath);
  ExclusiveDataLoader dataloader;
  std::thread t1(Detection, std::ref(dataloader), std::ref(detector));
  std::thread t2(Preview, std::ref(dataloader), std::ref(detector));
  t1.join();
  t2.join();
  return 0;
}



