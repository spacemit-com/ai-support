# support


## Getting started

### install dependence
gcc version: 14

prepare opencv (at least version 4.2.0)
```bash
sudo apt-get install libopencv-dev
```
prepare onnxruntime

visit [onnxruntime release](https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1) and download proper version.


### build demo with bash followed

```bash
ORT_HOME=${PATH_TO_ONNXRUNTIME}
# Note: Add the installation prefix of "OpenCV" to CMAKE_PREFIX_PATH or set
# "OpenCV_DIR" to a directory containing one of the following names:
#   OpenCVConfig.cmake
#   opencv-config.cmake
OPENCV_DIR=${PATH_TO_OPENCV_CMAKE_DIR}

mkdir build && pushd build
cmake .. -DORT_HOME=${ORT_HOME} -DOpenCV_DIR=${OPENCV_DIR} -DCMAKE_BUILD_TYPE=Debug -DTEST=OFF -DDEMO=ON
make install -j`nproc`
popd

# Or with cross compiler:
CROSS_TOOL=${PATH_TO_COMPILER_PREFIX}-
SYSROOT=${PATH_TO_SYSROOT}
cmake .. -DORT_HOME=${ORT_HOME} -DOpenCV_DIR=${OPENCV_DIR} \
   -DCMAKE_C_COMPILER=${CROSS_TOOL}gcc -DCMAKE_CXX_COMPILER=${CROSS_TOOL}g++ -DCMAKE_SYSROOT=${SYSROOT}
```

### run demo

```bash
./classification_demo <model_file_path> <label_file_path> <image_file_path>
or
./classification_demo <config_file_path> <image_file_path>
./detection_demo <model_file_path> <label_file_path> <image_file_path> <save_img_path> 
or 
./detection_demo <config_file_path> <image_file_path> <save_img_path>
./detection_stream_demo [-h <resize_height>] [-w <resize_width>] [-f] <model_file_path> <label_file_path> <input>
or
./detection_stream_demo [-h <resize_height>] [-w <resize_width>] [-f] <config_file_path> <input>
./detection_video_demo <model_file_path> <label_file_path> <video_file_path> <dst_file_path> (end with .avi)
or
./detection_video_demo <config_file_path> <video_file_path> <dst_file_path> (end with .avi)
./estimation_demo <det_model_file_path> <det_label_file_path> <pose_model_file_path> <image_file_path> <save_img_path> 
or
./estimation_demo <det_config_file_path> <pose_config_file_path> <image_file_path> <save_img_path> 
./tracker_stream_demo [-h <resize_height>] [-w <resize_width>] [-f] <det_model_file_path> <det_label_file_path> <pose_model_file_path> <input>
or
./tracker_stream_demo [-h <resize_height>] [-w <resize_width>] [-f] <det_config_file_path> <pose_config_file_path> <input>
```

### Using environment variables to implement functions
| Environment variable name         |                                          Remarks                                                    |
|-----------------------------------|:---------------------------------------------------------------------------------------------------:|
| SUPPORT_SHOW (stream demo)        |                                      -1 means not to display                                        |
| SUPPORT_SHOWFPS (stream demo)     |                                   If there is content, fps will be displayed                        |
| SUPPORT_PROFILING_PROJECTS        |                                 Generated profile file address                                      |
| SUPPORT_LOG_LEVEL                 |                                          The range is 0-4                                           |
| SUPPORT_GRAPH_OPTIMIZATION_LEVEL  | Graph optimization level (ort_disable_all, ort_enable_basic, ort_enable_extended, ort_enable_all)   |
| SUPPORT_OPT_MODEL_PATH            |                                    Optimized model path                                             |
| SUPPORT_DISABLE_SPACEMIT_EP       |                                  1 means to disable spacemit-ep                                     |
| SUPPORT_OPENCV_THREAD_NUM         |                              The number of threads used by opencv(>= 4.x)                           |

### Description of formats related to label files, configuration files, and model files
Model files format: [ONNX(Open Neural Network Exchange)](https://github.com/onnx/onnx)

label files format: using text document, [here](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/OpenVINO_EP/Linux/squeezenet_classification/synset.txt) is a recommended example

configuration files format: using [json](https://github.com/nlohmann/json), the recommended configuration file content is as [here](https://gitlab.dc.com:8443/bianbu/ai/support/-/blob/main/rootfs/usr/share/ai-support/models/yolov6.json)

### Python Support

See [README.md](./python/README.md) for more details.
