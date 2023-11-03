# support


## Getting started

### install dependence
prepare opencv (at least version 4.2.0)
```bash
sudo apt-get install libopencv-dev
```
prepare onnxruntime

visit [onnxruntime release](https://github.com/microsoft/onnxruntime/releases/tag/v1.15.1) and download proper version.

change  {SUPPORT_ROOT_PATH}/support/CMakeLists.txt to set {ROOT_DIR} and {ORT_ROOT_DIR}.